"""Table of Contents generator."""

import time
from collections.abc import Iterator
from typing import cast

import pymupdf
from langchain.chat_models.base import BaseChatModel
from langchain.messages import HumanMessage, SystemMessage
from rich.console import Console
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from tqdm import tqdm

from pdfalive.models.toc import TOC, TOCFeature
from pdfalive.prompts import TOC_GENERATOR_CONTINUATION_SYSTEM_PROMPT, TOC_GENERATOR_SYSTEM_PROMPT
from pdfalive.tokens import TokenUsage, estimate_tokens


# Console for rich output
console = Console()

# Default maximum tokens for features per batch to stay under context window limits
# 200k context window - reserve space for:
#   - System prompt (~2k tokens)
#   - User message template (~500 tokens)
#   - Response/output tokens (~10k reserved)
#   - Safety margin (~7.5k)
# This leaves ~180k for features, but we use 100k to be safe given estimation uncertainty
DEFAULT_MAX_TOKENS_PER_BATCH = 100000

# Default number of blocks to overlap between batches for context continuity
DEFAULT_OVERLAP_BLOCKS = 5

# Estimated token overhead for the prompt template (system + user message excluding features)
PROMPT_OVERHEAD_TOKENS = 3000

# Delay between LLM calls (in seconds) to avoid rate limiting
DEFAULT_REQUEST_DELAY_SECONDS = 2.0

# Maximum retry attempts for rate-limited requests
MAX_RETRY_ATTEMPTS = 5


class TOCGenerator:
    """Class to generate table of contents for a PDF document."""

    def __init__(self, doc: pymupdf.Document, llm: BaseChatModel) -> None:
        self.doc = doc
        self.llm = llm

    def run(self, output_file: str, force: bool = False) -> TokenUsage:
        """Generate the table of contents.

        Args:
            output_file: Path to save the modified PDF with TOC.
            force: If True, overwrite existing TOC. Otherwise raise if TOC exists.

        Returns:
            TokenUsage statistics from the LLM calls.

        Raises:
            ValueError: If document has existing TOC and force=False.
        """
        if self._check_for_existing_toc() and not force:
            # TODO: can also use any existing toc to guide LLM generation.
            raise ValueError("The document already has a Table of Contents. Use `--force` to overwrite.")

        features = self._extract_features(self.doc)
        toc, usage = self._extract_toc(features)

        self.doc.set_toc(toc.to_list())
        self.doc.save(output_file)

        return usage

    def _check_for_existing_toc(self) -> list:
        """Check if the document already has a TOC."""
        return self.doc.get_toc()

    def _extract_features(
        self,
        doc: pymupdf.Document,
        max_pages=None,
        max_blocks_per_page=3,
        max_lines_per_block=5,
        text_snippet_length=25,
    ) -> list:
        """Extract features from the document to generate TOC entries.

        Features are indexed by page, block, line, and span.
        They include attributes such as: font name, size, text length, and a text snippet.

        """

        # hierarchical structure of features detected in the page for (blocks, lines, spans)
        features: list[list] = []

        for ix, page in enumerate(tqdm(self.doc, desc="Processing pages...", total=self.doc.page_count)):
            page_number = ix + 1  # 1-indexed
            page_dict = page.get_text("dict")

            for block_ix, block in enumerate(page_dict["blocks"]):
                if block_ix >= max_blocks_per_page:
                    break

                features.append([])
                if block["type"] == 0:
                    # text block
                    for line_ix, line in enumerate(block["lines"]):
                        if line_ix >= max_lines_per_block:
                            break

                        features[-1].append([])

                        for span in line["spans"]:
                            features[-1][-1].append(
                                TOCFeature(
                                    page_number=page_number,
                                    font_name=span["font"],
                                    font_size=span["size"],
                                    text_length=len(span["text"]),
                                    text_snippet=span["text"][:text_snippet_length],
                                )
                            )

            if max_pages is not None and (ix + 1) >= max_pages:
                break

        return features

    def _extract_toc(
        self,
        features: list,
        max_depth: int = 2,
        max_tokens_per_batch: int = DEFAULT_MAX_TOKENS_PER_BATCH,
    ) -> tuple[TOC, TokenUsage]:
        """Infer TOC entries from extracted features using the LLM.

        This method handles pagination automatically when features exceed the
        token limit, splitting them into batches and merging results.

        Args:
            features: Nested list of TOCFeature objects extracted from the document.
            max_depth: Maximum depth level for TOC entries.
            max_tokens_per_batch: Maximum tokens per LLM call (for pagination).

        Returns:
            A tuple of (TOC, TokenUsage) with the generated TOC and usage statistics.
        """
        return self._extract_toc_paginated(
            features,
            max_depth=max_depth,
            max_tokens_per_batch=max_tokens_per_batch,
        )

    def _batch_features(
        self,
        features: list,
        max_tokens: int = DEFAULT_MAX_TOKENS_PER_BATCH,
        overlap_blocks: int = DEFAULT_OVERLAP_BLOCKS,
    ) -> Iterator[list]:
        """Split features into batches that fit within token limits.

        Args:
            features: Nested list of features (blocks containing lines containing spans).
            max_tokens: Maximum estimated tokens per batch (for features only).
            overlap_blocks: Number of blocks to overlap between batches for context.

        Yields:
            Batches of features, each estimated to be under max_tokens.
        """
        if not features:
            yield []
            return

        # Account for prompt overhead in the effective limit
        effective_max_tokens = max_tokens - PROMPT_OVERHEAD_TOKENS

        current_batch: list = []
        current_tokens = 0

        for block in features:
            block_str = str(block)
            block_tokens = estimate_tokens(block_str)

            # If adding this block would exceed limit and we have content, yield current batch
            if current_tokens + block_tokens > effective_max_tokens and current_batch:
                yield current_batch

                # Start new batch with overlap from end of previous batch
                overlap_start = max(0, len(current_batch) - overlap_blocks)
                current_batch = current_batch[overlap_start:]
                current_tokens = estimate_tokens(str(current_batch))

            current_batch.append(block)
            current_tokens += block_tokens

        # Yield final batch if non-empty
        if current_batch:
            yield current_batch

    def _invoke_with_retry(self, model, messages, batch_description: str, input_tokens: int) -> TOC:
        """Invoke the LLM with retry logic for rate limiting.

        Args:
            model: The LLM model with structured output.
            messages: The messages to send.
            batch_description: Description of the current batch for logging.
            input_tokens: Estimated input tokens for logging.

        Returns:
            The TOC response from the LLM.
        """

        def _log_retry(retry_state) -> None:
            """Log retry attempt information."""
            wait_time = getattr(retry_state.next_action, "sleep", 0) if retry_state.next_action else 0
            console.print(
                f"  [yellow]Rate limited or error. Retrying in {wait_time:.1f}s "
                f"(attempt {retry_state.attempt_number}/{MAX_RETRY_ATTEMPTS})...[/yellow]"
            )

        @retry(
            retry=retry_if_exception_type(Exception),
            stop=stop_after_attempt(MAX_RETRY_ATTEMPTS),
            wait=wait_exponential(multiplier=2, min=4, max=120),
            before_sleep=_log_retry,
            reraise=True,
        )
        def _invoke():
            return model.invoke(messages)

        console.print(f"  [dim]Invoking LLM for {batch_description} (~{input_tokens:,} input tokens)...[/dim]")
        start_time = time.time()

        response = _invoke()

        elapsed = time.time() - start_time
        console.print(f"  [green]Completed {batch_description} in {elapsed:.1f}s[/green]")

        return cast(TOC, response)

    def _extract_toc_paginated(
        self,
        features: list,
        max_depth: int = 2,
        max_tokens_per_batch: int = DEFAULT_MAX_TOKENS_PER_BATCH,
        overlap_blocks: int = DEFAULT_OVERLAP_BLOCKS,
        request_delay: float = DEFAULT_REQUEST_DELAY_SECONDS,
    ) -> tuple[TOC, TokenUsage]:
        """Extract TOC using pagination for large documents.

        Splits features into batches, makes separate LLM calls for each,
        and merges the results.

        Args:
            features: Nested list of TOCFeature objects.
            max_depth: Maximum TOC depth level.
            max_tokens_per_batch: Maximum tokens per LLM call.
            overlap_blocks: Number of blocks to overlap between batches.
            request_delay: Delay in seconds between LLM calls to avoid rate limiting.

        Returns:
            A tuple of (merged TOC, TokenUsage statistics).
        """
        usage = TokenUsage()
        merged_toc = TOC(entries=[])
        model = self.llm.with_structured_output(TOC)

        batches = list(self._batch_features(features, max_tokens_per_batch, overlap_blocks))
        total_batches = len(batches)

        console.print(f"[bold]Processing {total_batches} batch(es) of features...[/bold]")

        for batch_idx, batch in enumerate(batches):
            is_first_batch = batch_idx == 0
            batch_description = f"batch {batch_idx + 1}/{total_batches}"

            # Add delay between requests (except for the first one)
            if not is_first_batch and request_delay > 0:
                console.print(f"  [dim]Waiting {request_delay:.1f}s before next request...[/dim]")
                time.sleep(request_delay)

            # Select appropriate prompt
            system_prompt = TOC_GENERATOR_SYSTEM_PROMPT if is_first_batch else TOC_GENERATOR_CONTINUATION_SYSTEM_PROMPT

            # Build user message with batch context
            batch_context = ""
            if not is_first_batch:
                batch_context = f"\n\nThis is batch {batch_idx + 1} of {total_batches}."

            user_content = f"""
                Generate a table of contents based on the document features given below.
                Limit the TOC to a maximum depth of {max_depth} levels.{batch_context}
                \n\n
                ------------------------
                {str(batch)}
            """

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_content),
            ]

            # Estimate input tokens
            input_text = system_prompt + user_content
            input_tokens = estimate_tokens(input_text)

            # Make LLM call with retry logic
            batch_toc = self._invoke_with_retry(model, messages, batch_description, input_tokens)

            # Estimate output tokens (rough estimate based on response)
            output_tokens = estimate_tokens(str(batch_toc.entries))

            # Record token usage
            usage.add_call(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                description=f"Batch {batch_idx + 1}/{total_batches}",
            )

            # Log entries found
            entries_found = len(batch_toc.entries)
            if entries_found > 0:
                console.print(f"  [cyan]Found {entries_found} TOC entries in this batch[/cyan]")

            # Merge results
            merged_toc = merged_toc.merge(batch_toc)

        console.print(f"[bold green]All batches processed. Total TOC entries: {len(merged_toc.entries)}[/bold green]")

        return merged_toc, usage
