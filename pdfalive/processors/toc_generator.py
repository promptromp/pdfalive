"""Table of Contents generator."""

import re
import time
import warnings
from collections import Counter
from collections.abc import Iterator
from multiprocessing import Pool, cpu_count
from typing import cast

import pymupdf
from langchain.chat_models.base import BaseChatModel
from langchain.messages import HumanMessage, SystemMessage
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from pdfalive.models.toc import TOC, TOCFeature
from pdfalive.prompts import (
    TOC_GENERATOR_CONTINUATION_SYSTEM_PROMPT,
    TOC_GENERATOR_SYSTEM_PROMPT,
    TOC_POSTPROCESSOR_SYSTEM_PROMPT,
)
from pdfalive.tokens import TokenUsage, estimate_tokens


# Suppress PydanticSerializationUnexpectedValue warnings emitted by LangChain's
# with_structured_output() wrapper. The parsed output is correct; the warning is a
# known LangChain + Pydantic compatibility issue (the union serializer for the
# response type warns "Expected `none`" even though the value is valid).
# A context-manager approach (warnings.catch_warnings) doesn't work here because
# pydantic-core emits the warning from Rust, bypassing Python-level scoped filters.
warnings.filterwarnings(
    "ignore", message="Pydantic serializer warnings", category=UserWarning, module=r"pydantic\.main"
)

# Sub-pattern for Roman numerals (I through XXXIX covers typical chapter counts)
_ROMAN_NUMERAL_RE = r"(?:X{0,3}(?:IX|IV|V?I{0,3}))"

# Regex pattern for section numbering (e.g. "1.", "1.2", "Chapter 1", "Appendix A",
# Roman numerals like "I ", "XIV.", and letter-spaced "C H A P T E R")
_SECTION_NUMBER_PATTERN = re.compile(
    r"^\s*("
    r"\d+\.|\d+\.\d+"  # Arabic: "1.", "1.2"
    r"|(?:Chapter|Section|Part|Appendix)\s"  # Named prefixes
    r"|C\s+H\s+A\s+P\s+T\s+E\s+R"  # Letter-spaced CHAPTER
    r"|" + _ROMAN_NUMERAL_RE + r"\.?\s"  # Roman numeral + optional dot + space
    r")",
    re.IGNORECASE,
)

# Pattern for letter-spaced ALL-CAPS text (e.g. "C H A P T E R  I" or "P R E F A C E")
_LETTERSPACED_PATTERN = re.compile(r"^[A-Z](\s+[A-Z]){3,}")

# Minimum/maximum text length for heading candidates
_HEADING_MIN_LENGTH = 3
_HEADING_MAX_LENGTH = 200

# Minimum font size ratio vs body text to be considered a heading candidate
_HEADING_FONT_SIZE_RATIO = 1.15


def apply_toc_to_document(doc: pymupdf.Document, toc: list, output_file: str) -> None:
    """Apply a TOC (bookmarks) to a document and save it.

    This helper function is useful when you want to apply a TOC generated
    from one document (e.g., an OCR'd version) to another document
    (e.g., the original without OCR text layer).

    Args:
        doc: PyMuPDF Document to apply the TOC to.
        toc: TOC list in PyMuPDF format (list of [level, title, page] entries).
        output_file: Path to save the modified document.
    """
    doc.set_toc(toc)
    doc.save(output_file)


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
# Default is 10s to stay under typical rate limits (e.g., 30k input tokens/minute)
DEFAULT_REQUEST_DELAY_SECONDS = 10.0

# Retry configuration for rate-limited requests
MAX_RETRY_ATTEMPTS = 5
RETRY_MULTIPLIER = 2  # Exponential backoff multiplier
RETRY_MIN_WAIT_SECONDS = 10  # Minimum wait time between retries
RETRY_MAX_WAIT_SECONDS = 120  # Maximum wait time between retries


def _is_bold_font(span: dict) -> bool:
    """Check if a span uses a bold font.

    Uses both the PyMuPDF flags bitmask (bit 4 = bold) and font name heuristics.

    Args:
        span: A PyMuPDF span dict with 'flags' and 'font' keys.

    Returns:
        True if the span is bold.
    """
    flags = span.get("flags", 0)
    font_name = span.get("font", "")
    return bool(flags & 16) or "bold" in font_name.lower()


def _compute_body_font_profile(features: list) -> tuple[str, float]:
    """Determine the most common (font_name, font_size) pair across all features.

    This represents the "body text" baseline used to identify heading candidates.

    Args:
        features: Nested list of TOCFeature objects (blocks > lines > spans).

    Returns:
        Tuple of (font_name, font_size) for the most common pair.
        Falls back to ("", 0.0) if no features are found.
    """
    counter: Counter[tuple[str, float]] = Counter()
    for block in features:
        for line in block:
            for span in line:
                if isinstance(span, TOCFeature):
                    counter[(span.font_name, span.font_size)] += 1

    if not counter:
        return ("", 0.0)

    return counter.most_common(1)[0][0]


def _is_heading_candidate(span: dict, body_font_name: str, body_font_size: float) -> bool:
    """Determine if a span is likely a heading based on font characteristics.

    A span is a heading candidate if:
    - Font size >= 1.15x body font size (15% larger), OR
    - Bold font AND font size >= body font size, OR
    - Text matches section numbering pattern (e.g. "1.", "Chapter", "Section")
    AND text length is between 3-200 chars.

    Args:
        span: A PyMuPDF span dict.
        body_font_name: The most common font name (body text baseline).
        body_font_size: The most common font size (body text baseline).

    Returns:
        True if the span looks like a heading.
    """
    text = span.get("text", "").strip()
    text_len = len(text)

    if text_len < _HEADING_MIN_LENGTH or text_len > _HEADING_MAX_LENGTH:
        return False

    font_size = span.get("size", 0.0)
    is_bold = _is_bold_font(span)

    # Size-based: significantly larger than body text
    if body_font_size > 0 and font_size >= body_font_size * _HEADING_FONT_SIZE_RATIO:
        return True

    # Bold + at least body size
    if is_bold and body_font_size > 0 and font_size >= body_font_size:
        return True

    # Section numbering pattern
    if _SECTION_NUMBER_PATTERN.match(text):
        return True

    # Letter-spaced ALL-CAPS text (e.g. "C H A P T E R  I")
    return bool(_LETTERSPACED_PATTERN.match(text))


def _extract_features_from_page_range(args: tuple) -> tuple[int, int, list, list]:
    """Worker function to extract features from a range of pages.

    This function is designed to be called in a separate process.
    It opens the document independently and processes its assigned pages.

    Args:
        args: Tuple of (process_index, total_processes, input_path,
                       max_blocks_per_page, max_lines_per_block, text_snippet_length)

    Returns:
        Tuple of (start_page, end_page, features_list, remaining_spans_buffer)
        for the processed range. remaining_spans_buffer contains
        (page_number, page_height, insert_index, spans) tuples for blocks beyond
        the max_blocks_per_page limit.
    """
    (
        process_idx,
        total_processes,
        input_path,
        max_blocks_per_page,
        max_lines_per_block,
        text_snippet_length,
    ) = args

    doc = pymupdf.open(input_path)
    page_count = doc.page_count

    # Calculate page range for this process
    pages_per_process = page_count // total_processes
    start_page = process_idx * pages_per_process
    end_page = start_page + pages_per_process if process_idx < total_processes - 1 else page_count

    features: list[list] = []
    remaining_spans_buffer: list[tuple[int, float, int, list[dict]]] = []

    for page_idx in range(start_page, end_page):
        page = doc[page_idx]
        page_number = page_idx + 1  # 1-indexed
        page_dict = page.get_text("dict")
        page_height = page_dict.get("height", page.rect.height) if hasattr(page, "rect") else 1.0

        for block_ix, block in enumerate(page_dict["blocks"]):
            if block_ix >= max_blocks_per_page:
                # Buffer remaining blocks for heading candidate scanning
                if block["type"] == 0:
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            remaining_spans_buffer.append((page_number, page_height, len(features), [span]))
                continue

            features.append([])
            if block["type"] == 0:
                # text block
                for line_ix, line in enumerate(block["lines"]):
                    if line_ix >= max_lines_per_block:
                        break

                    features[-1].append([])

                    for span in line["spans"]:
                        bbox = span.get("bbox", None)
                        y_pos = round(bbox[1] / page_height, 2) if bbox and page_height > 0 else None
                        features[-1][-1].append(
                            TOCFeature(
                                page_number=page_number,
                                font_name=span["font"],
                                font_size=span["size"],
                                text_length=len(span["text"]),
                                text_snippet=span["text"][:text_snippet_length],
                                y_position=y_pos,
                                is_bold=_is_bold_font(span),
                            )
                        )

    doc.close()
    return start_page, end_page, features, remaining_spans_buffer


class TOCGenerator:
    """Class to generate table of contents for a PDF document."""

    def __init__(
        self,
        doc: pymupdf.Document,
        llm: BaseChatModel,
        num_processes: int | None = None,
    ) -> None:
        """Initialize the TOC generator.

        Args:
            doc: PyMuPDF Document object.
            llm: LangChain chat model for TOC inference.
            num_processes: Number of parallel processes for feature extraction.
                          Defaults to CPU count - 1.
        """
        self.doc = doc
        self.llm = llm
        self.num_processes = num_processes or max(1, cpu_count() - 1)

    def run(
        self,
        output_file: str,
        force: bool = False,
        request_delay: float = DEFAULT_REQUEST_DELAY_SECONDS,
        postprocess: bool = False,
    ) -> TokenUsage:
        """Generate the table of contents.

        Args:
            output_file: Path to save the modified PDF with TOC.
            force: If True, overwrite existing TOC. Otherwise raise if TOC exists.
            request_delay: Delay in seconds between LLM calls to avoid rate limiting.
            postprocess: If True, run a postprocessing step to clean up and improve the TOC.

        Returns:
            TokenUsage statistics from the LLM calls.

        Raises:
            ValueError: If document has existing TOC and force=False.
        """
        if self._check_for_existing_toc() and not force:
            # TODO: can also use any existing toc to guide LLM generation.
            raise ValueError(
                "The input document already has a Table of Contents. Use `--force` to force TOC generation."
            )

        features = self._extract_features(self.doc)
        toc, usage = self._extract_toc(features, request_delay=request_delay)

        # Optionally postprocess the TOC to clean up duplicates, fix errors, etc.
        if postprocess:
            toc, postprocess_usage = self._postprocess_toc(toc, features)
            usage = usage + postprocess_usage

        toc = toc.sanitize_hierarchy()
        self.doc.set_toc(toc.to_list())
        self.doc.save(output_file)

        return usage

    def _check_for_existing_toc(self) -> list:
        """Check if the document already has a TOC."""
        return self.doc.get_toc()

    def _extract_features(
        self,
        doc: pymupdf.Document,
        max_pages: int | None = None,
        max_blocks_per_page: int = 3,
        max_lines_per_block: int = 5,
        text_snippet_length: int = 25,
        show_progress: bool = True,
    ) -> list:
        """Extract features from the document to generate TOC entries.

        Features are indexed by page, block, line, and span.
        They include attributes such as: font name, size, text length, and a text snippet.

        Uses multiprocessing for large documents to speed up extraction.

        Args:
            doc: PyMuPDF Document object.
            max_pages: Maximum number of pages to process (None for all).
            max_blocks_per_page: Maximum blocks to extract per page.
            max_lines_per_block: Maximum lines to extract per block.
            text_snippet_length: Maximum characters for text snippets.
            show_progress: Whether to show progress indicator.

        Returns:
            Nested list of TOCFeature objects.
        """
        # For documents opened in memory (not from file), use sequential processing
        if not doc.name:
            return self._extract_features_sequential(
                doc, max_pages, max_blocks_per_page, max_lines_per_block, text_snippet_length, show_progress
            )

        page_count = doc.page_count if max_pages is None else min(max_pages, doc.page_count)
        num_processes = min(self.num_processes, page_count)

        if num_processes <= 1:
            return self._extract_features_sequential(
                doc, max_pages, max_blocks_per_page, max_lines_per_block, text_snippet_length, show_progress
            )

        return self._extract_features_parallel(
            doc.name,
            page_count,
            num_processes,
            max_blocks_per_page,
            max_lines_per_block,
            text_snippet_length,
            show_progress,
        )

    def _extract_features_sequential(
        self,
        doc: pymupdf.Document,
        max_pages: int | None = None,
        max_blocks_per_page: int = 3,
        max_lines_per_block: int = 5,
        text_snippet_length: int = 25,
        show_progress: bool = True,
    ) -> list:
        """Extract features sequentially (single process) using a two-phase approach.

        Phase 1: Extract first N blocks per page (enriched with y_position and is_bold),
        and buffer raw span data from remaining blocks.
        Phase 2: Compute body font profile, filter buffered spans for heading candidates,
        and insert them in page order.

        Args:
            doc: PyMuPDF Document object.
            max_pages: Maximum number of pages to process.
            max_blocks_per_page: Maximum blocks per page.
            max_lines_per_block: Maximum lines per block.
            text_snippet_length: Maximum characters for snippets.
            show_progress: Whether to show progress indicator.

        Returns:
            Nested list of TOCFeature objects.
        """
        features: list[list] = []
        page_count = doc.page_count if max_pages is None else min(max_pages, doc.page_count)

        # Buffer for remaining-block spans: list of (page_number, page_height, insert_index, spans)
        remaining_spans_buffer: list[tuple[int, float, int, list[dict]]] = []

        def _process_page(page, page_number: int) -> None:
            page_dict = page.get_text("dict")
            page_height = page_dict.get("height", page.rect.height) if hasattr(page, "rect") else 1.0

            for block_ix, block in enumerate(page_dict["blocks"]):
                if block_ix >= max_blocks_per_page:
                    # Buffer remaining blocks for heading candidate scanning
                    if block["type"] == 0:
                        for line in block.get("lines", []):
                            for span in line.get("spans", []):
                                remaining_spans_buffer.append((page_number, page_height, len(features), [span]))
                    continue

                features.append([])
                if block["type"] == 0:
                    # text block
                    for line_ix, line in enumerate(block["lines"]):
                        if line_ix >= max_lines_per_block:
                            break

                        features[-1].append([])

                        for span in line["spans"]:
                            bbox = span.get("bbox", None)
                            y_pos = round(bbox[1] / page_height, 2) if bbox and page_height > 0 else None
                            features[-1][-1].append(
                                TOCFeature(
                                    page_number=page_number,
                                    font_name=span["font"],
                                    font_size=span["size"],
                                    text_length=len(span["text"]),
                                    text_snippet=span["text"][:text_snippet_length],
                                    y_position=y_pos,
                                    is_bold=_is_bold_font(span),
                                )
                            )

        if show_progress:
            console.print(f"[cyan]Extracting features from {page_count} page(s)...[/cyan]")
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=console,
            ) as progress:
                task = progress.add_task("Extracting features...", total=page_count)

                for ix, page in enumerate(doc):
                    if max_pages is not None and ix >= max_pages:
                        break
                    _process_page(page, ix + 1)
                    progress.advance(task)
        else:
            for ix, page in enumerate(doc):
                if max_pages is not None and ix >= max_pages:
                    break
                _process_page(page, ix + 1)

        # Phase 2: Compute body font profile and scan remaining blocks for heading candidates
        if remaining_spans_buffer:
            body_font_name, body_font_size = _compute_body_font_profile(features)

            # Group heading candidates by their insert_index for in-order insertion
            # Process in reverse order so insert indices remain valid
            candidates_by_index: dict[int, list[list]] = {}
            for page_number, page_height, insert_idx, spans in remaining_spans_buffer:
                for span in spans:
                    if _is_heading_candidate(span, body_font_name, body_font_size):
                        bbox = span.get("bbox", None)
                        y_pos = round(bbox[1] / page_height, 2) if bbox and page_height > 0 else None
                        feature = TOCFeature(
                            page_number=page_number,
                            font_name=span["font"],
                            font_size=span["size"],
                            text_length=len(span["text"]),
                            text_snippet=span["text"][:text_snippet_length],
                            y_position=y_pos,
                            is_bold=_is_bold_font(span),
                        )
                        if insert_idx not in candidates_by_index:
                            candidates_by_index[insert_idx] = []
                        candidates_by_index[insert_idx].append([[feature]])

            # Insert heading candidate blocks at the right positions (reverse order to preserve indices)
            for idx in sorted(candidates_by_index.keys(), reverse=True):
                for block in reversed(candidates_by_index[idx]):
                    features.insert(idx, block)

        return features

    def _extract_features_parallel(
        self,
        input_path: str,
        page_count: int,
        num_processes: int,
        max_blocks_per_page: int = 3,
        max_lines_per_block: int = 5,
        text_snippet_length: int = 25,
        show_progress: bool = True,
    ) -> list:
        """Extract features in parallel using multiprocessing.

        Args:
            input_path: Path to the PDF file.
            page_count: Total number of pages to process.
            num_processes: Number of parallel processes.
            max_blocks_per_page: Maximum blocks per page.
            max_lines_per_block: Maximum lines per block.
            text_snippet_length: Maximum characters for snippets.
            show_progress: Whether to show progress indicator.

        Returns:
            Merged list of features from all processes.
        """
        if show_progress:
            console.print(
                f"[cyan]Extracting features from {page_count} page(s) "
                f"(using {num_processes} parallel processes)...[/cyan]"
            )

        # Prepare arguments for worker processes
        args_list = [
            (i, num_processes, input_path, max_blocks_per_page, max_lines_per_block, text_snippet_length)
            for i in range(num_processes)
        ]

        # Process in parallel
        with Pool(processes=num_processes) as pool:
            results = []
            if show_progress:
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TaskProgressColumn(),
                    console=console,
                ) as progress:
                    task = progress.add_task("Extracting features...", total=num_processes)

                    for result in pool.imap_unordered(_extract_features_from_page_range, args_list):
                        results.append(result)
                        progress.advance(task)
            else:
                results = pool.map(_extract_features_from_page_range, args_list)

        # Sort results by start page to maintain order
        results = sorted(results, key=lambda x: x[0])

        # Merge features and remaining spans buffers from all processes
        if show_progress:
            console.print("[cyan]Merging extracted features...[/cyan]")

        all_features: list = []
        all_remaining_spans: list[tuple[int, float, int, list[dict]]] = []
        feature_offset = 0

        for _, _, features, remaining_spans in results:
            all_features.extend(features)
            # Adjust insert indices by the cumulative feature offset
            for page_number, page_height, insert_idx, spans in remaining_spans:
                all_remaining_spans.append((page_number, page_height, insert_idx + feature_offset, spans))
            feature_offset += len(features)

        # Phase 2: Compute body font profile and scan for heading candidates
        if all_remaining_spans:
            body_font_name, body_font_size = _compute_body_font_profile(all_features)

            candidates_by_index: dict[int, list[list]] = {}
            for page_number, page_height, insert_idx, spans in all_remaining_spans:
                for span in spans:
                    if _is_heading_candidate(span, body_font_name, body_font_size):
                        bbox = span.get("bbox", None)
                        y_pos = round(bbox[1] / page_height, 2) if bbox and page_height > 0 else None
                        feature = TOCFeature(
                            page_number=page_number,
                            font_name=span["font"],
                            font_size=span["size"],
                            text_length=len(span["text"]),
                            text_snippet=span["text"][:text_snippet_length],
                            y_position=y_pos,
                            is_bold=_is_bold_font(span),
                        )
                        if insert_idx not in candidates_by_index:
                            candidates_by_index[insert_idx] = []
                        candidates_by_index[insert_idx].append([[feature]])

            for idx in sorted(candidates_by_index.keys(), reverse=True):
                for block in reversed(candidates_by_index[idx]):
                    all_features.insert(idx, block)

        return all_features

    def _extract_toc(
        self,
        features: list,
        max_depth: int = 2,
        max_tokens_per_batch: int = DEFAULT_MAX_TOKENS_PER_BATCH,
        request_delay: float = DEFAULT_REQUEST_DELAY_SECONDS,
    ) -> tuple[TOC, TokenUsage]:
        """Infer TOC entries from extracted features using the LLM.

        This method handles pagination automatically when features exceed the
        token limit, splitting them into batches and merging results.

        Args:
            features: Nested list of TOCFeature objects extracted from the document.
            max_depth: Maximum depth level for TOC entries.
            max_tokens_per_batch: Maximum tokens per LLM call (for pagination).
            request_delay: Delay in seconds between LLM calls to avoid rate limiting.

        Returns:
            A tuple of (TOC, TokenUsage) with the generated TOC and usage statistics.
        """
        return self._extract_toc_paginated(
            features,
            max_depth=max_depth,
            max_tokens_per_batch=max_tokens_per_batch,
            request_delay=request_delay,
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

            # Extract exception details if available
            exception_info = ""
            if retry_state.outcome is not None:
                exc = retry_state.outcome.exception()
                if exc is not None:
                    exc_name = type(exc).__name__
                    exc_msg = str(exc) or getattr(exc, "message", "")
                    exception_info = f" ({exc_name}: {exc_msg})" if exc_msg else f" ({exc_name})"

            console.print(
                f"  [yellow]Error encountered{exception_info}. Retrying in {wait_time:.1f}s "
                f"(attempt {retry_state.attempt_number}/{MAX_RETRY_ATTEMPTS})...[/yellow]"
            )

        @retry(
            retry=retry_if_exception_type(Exception),
            stop=stop_after_attempt(MAX_RETRY_ATTEMPTS),
            wait=wait_exponential(
                multiplier=RETRY_MULTIPLIER,
                min=RETRY_MIN_WAIT_SECONDS,
                max=RETRY_MAX_WAIT_SECONDS,
            ),
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

    def _extract_reference_toc_text(
        self,
        max_pages: int = 10,
    ) -> str:
        """Extract text from the first few pages that may contain a printed TOC.

        Scans the first N pages of the document looking for text that might
        be a printed table of contents, which can be used as a reference
        for postprocessing.

        Args:
            max_pages: Maximum number of pages to scan from the beginning.

        Returns:
            Concatenated text from the first pages, with page markers.
        """
        pages_to_scan = min(max_pages, self.doc.page_count)
        reference_texts = []

        for page_idx in range(pages_to_scan):
            page = self.doc[page_idx]
            page_text = page.get_text("text")
            if page_text.strip():
                reference_texts.append(f"--- Page {page_idx + 1} ---\n{page_text}")

        return "\n\n".join(reference_texts)

    def _postprocess_toc(
        self,
        toc: TOC,
        features: list,
        max_pages_for_reference_toc: int = 20,
    ) -> tuple[TOC, TokenUsage]:
        """Postprocess a generated TOC using LLM to clean up and improve entries.

        This method takes a previously generated TOC and refines it by:
        - Removing duplicate entries
        - Fixing typos in titles
        - Adjusting page numbers based on any printed TOC found in the document
        - Adding missing entries discovered from a printed TOC
        - Removing false positives

        Args:
            toc: The previously generated TOC to postprocess.
            features: The document features used for the original extraction.
            max_pages_for_reference_toc: Maximum pages to scan for a printed TOC.

        Returns:
            A tuple of (refined TOC, TokenUsage) with the improved TOC.
        """
        usage = TokenUsage()
        model = self.llm.with_structured_output(TOC)

        # Extract reference text from first pages (may contain printed TOC)
        reference_text = self._extract_reference_toc_text(max_pages=max_pages_for_reference_toc)

        # Format the generated TOC for the prompt
        toc_entries_str = "\n".join(
            f"- {entry.title} (page {entry.page_number}, level {entry.level}, confidence {entry.confidence:.2f})"
            for entry in toc.entries
        )

        # Build a compact representation of features for context
        # Only include a summary to keep token count reasonable
        features_summary = self._summarize_features_for_postprocessing(features)

        # Detect front matter offset to warn the LLM
        offset_note = self._detect_page_offset_note(toc, reference_text)

        user_content = f"""
Please review and refine the following automatically generated Table of Contents.

## Generated TOC (to be refined)

The page numbers below are **PDF page numbers** (physical position in the PDF file).
For existing entries, prefer keeping these page numbers unchanged.

{toc_entries_str if toc_entries_str else "(No entries were generated)"}

## Reference Text from First Pages (may contain printed TOC)

{offset_note}

{reference_text if reference_text else "(No text extracted from first pages)"}

## Document Features Summary

{features_summary}

Please return a cleaned and improved TOC. It is very important that you add any missing \
sections or chapters visible in the printed TOC above. For new entries, use whatever page \
numbers seem most appropriate — page numbers will be verified and corrected downstream.
"""

        messages = [
            SystemMessage(content=TOC_POSTPROCESSOR_SYSTEM_PROMPT),
            HumanMessage(content=user_content),
        ]

        # Estimate input tokens
        input_text = TOC_POSTPROCESSOR_SYSTEM_PROMPT + user_content
        input_tokens = estimate_tokens(input_text)

        # Make LLM call with retry logic
        console.print("[bold]Postprocessing TOC...[/bold]")
        refined_toc = self._invoke_with_retry(model, messages, "TOC postprocessing", input_tokens)

        # Validate and correct page numbers if the LLM shifted to printed page numbers
        refined_toc = self._correct_postprocessed_page_numbers(toc, refined_toc)

        # Estimate output tokens
        output_tokens = estimate_tokens(str(refined_toc.entries))

        # Record token usage
        usage.add_call(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            description="TOC postprocessing",
        )

        console.print(
            f"[bold green]Postprocessing complete. "
            f"Refined TOC has {len(refined_toc.entries)} entries "
            f"(was {len(toc.entries)})[/bold green]"
        )

        return refined_toc, usage

    def _detect_front_matter_offset(self, toc: TOC) -> int:
        """Detect the number of front matter pages from Phase 1 TOC data.

        Finds the first level-1 entry past page 5 and assumes it corresponds
        to the start of the main content (approximately printed page 1).

        Args:
            toc: The generated TOC with correct PDF page numbers.

        Returns:
            Estimated front matter offset (0 if no significant front matter detected).
        """
        for entry in toc.entries:
            if entry.level == 1 and entry.page_number > 5:
                return entry.page_number - 1
        return 0

    def _detect_page_offset_note(self, toc: TOC, reference_text: str) -> str:
        """Detect the offset between PDF page numbers and printed page numbers.

        Looks at the first chapter-level entry in the generated TOC to estimate
        the front matter offset, then generates an explicit warning for the LLM.

        Args:
            toc: The generated TOC with correct PDF page numbers.
            reference_text: The extracted reference text from first pages.

        Returns:
            A warning string to include in the postprocessor prompt, or empty string
            if no offset is detected.
        """
        estimated_offset = self._detect_front_matter_offset(toc)

        if estimated_offset < 3:
            return ""

        first_chapter_page = estimated_offset + 1

        return (
            f"**Note**: This document has approximately {estimated_offset} pages of front matter. "
            f"The printed page numbers below (if any) start counting AFTER the front matter, "
            f'so printed page "1" corresponds to approximately PDF page {first_chapter_page}. '
            f"For existing entries, keep their PDF page numbers. For new entries, use your best estimate."
        )

    def _page_contains_heading(self, page_number: int, title: str, window: int = 1) -> bool:
        """Check if a heading's text appears on or near the given page.

        Args:
            page_number: 1-indexed PDF page number to check.
            title: The heading title to search for.
            window: Number of pages before/after to also check.

        Returns:
            True if the title text is found on or near the page.
        """
        # Normalize: strip common prefixes, lowercase, collapse whitespace
        search_text = re.sub(r"^\d+\.\s*", "", title)  # Strip "1. " prefixes
        search_text = re.sub(r"\s+", " ", search_text.strip().lower())

        if len(search_text) < 3:
            return False  # Too short to match reliably

        start = max(0, page_number - 1 - window)
        end = min(self.doc.page_count, page_number + window)

        for page_idx in range(start, end):
            page_text = self.doc[page_idx].get_text("text")
            page_text_normalized = re.sub(r"\s+", " ", page_text.lower())
            if search_text in page_text_normalized:
                return True

        return False

    def _resolve_page_number(self, title: str, page_number: int, offset: int) -> int:
        """Determine the correct PDF page number for a new postprocessed entry.

        Checks whether the entry's page number is a printed page number (needs offset)
        or already a correct PDF page number by searching the actual PDF text.

        Args:
            title: The TOC entry title to search for.
            page_number: The page number from the postprocessor output.
            offset: The detected front matter offset.

        Returns:
            The corrected PDF page number.
        """
        corrected_page = page_number + offset

        # Validate corrected page is in range
        if corrected_page < 1 or corrected_page > self.doc.page_count:
            return page_number  # Can't apply offset, keep original

        # Check if heading text appears at the offset-corrected page (±1 page window)
        if self._page_contains_heading(corrected_page, title):
            return corrected_page

        # Check if heading text appears at the original page
        if 1 <= page_number <= self.doc.page_count and self._page_contains_heading(page_number, title):
            return page_number

        # Neither matched — default to offset-corrected (most common case for printed TOC entries)
        return corrected_page

    def _correct_postprocessed_page_numbers(self, original_toc: TOC, refined_toc: TOC) -> TOC:
        """Correct page numbers for entries added or modified by the postprocessor.

        Uses a deterministic strategy:
        1. Detect the front matter offset from Phase 1 data.
        2. For matched entries (exist in both original and refined): restore the
           original PDF page number.
        3. For new entries (only in refined) with significant front matter: verify
           and apply the offset by checking the actual PDF text.

        Args:
            original_toc: The pre-postprocessing TOC with correct PDF page numbers.
            refined_toc: The postprocessed TOC that may have shifted page numbers.

        Returns:
            The corrected TOC with PDF page numbers restored.
        """
        if not refined_toc.entries:
            return refined_toc

        # Step 1: Detect front matter offset from Phase 1 data
        offset = self._detect_front_matter_offset(original_toc)

        # Step 2: Build map of original entries (normalized title -> page_number)
        def normalize(title: str) -> str:
            return re.sub(r"\s+", " ", title.strip().lower())

        original_map: dict[str, int] = {}
        for entry in original_toc.entries:
            original_map[normalize(entry.title)] = entry.page_number

        corrected = []
        restored_count = 0
        offset_applied_count = 0

        for entry in refined_toc.entries:
            key = normalize(entry.title)

            if key in original_map:
                # MATCHED: restore original PDF page number
                orig_page = original_map[key]
                if entry.page_number != orig_page:
                    restored_count += 1
                corrected.append(entry.model_copy(update={"page_number": orig_page}))

            elif offset >= 3:
                # NEW entry with significant front matter — verify and apply offset
                corrected_page = self._resolve_page_number(entry.title, entry.page_number, offset)
                if corrected_page != entry.page_number:
                    offset_applied_count += 1
                corrected.append(entry.model_copy(update={"page_number": corrected_page}))

            else:
                # No significant front matter — keep as-is
                corrected.append(entry)

        # Log corrections
        if restored_count > 0:
            console.print(f"  [yellow]Restored page numbers for {restored_count} existing entries[/yellow]")
        if offset_applied_count > 0:
            console.print(
                f"  [yellow]Applied front matter offset (+{offset}) to {offset_applied_count} new entries[/yellow]"
            )

        return TOC(entries=corrected)

    def _summarize_features_for_postprocessing(self, features: list, max_entries: int = 150) -> str:
        """Create a compact summary of features for postprocessing context.

        Samples spans evenly across the entire document rather than taking only
        the first N sequentially. This ensures the postprocessor sees features
        from chapter boundaries throughout the book, not just early pages.

        Args:
            features: Nested list of TOCFeature objects.
            max_entries: Maximum number of feature entries to include.

        Returns:
            A string summary of the most relevant features.
        """
        # Flatten all TOCFeature spans into a single list
        all_spans: list[TOCFeature] = []
        for block in features:
            for line in block:
                for span in line:
                    if isinstance(span, TOCFeature):
                        all_spans.append(span)

        if not all_spans:
            return "(No features available)"

        # Sample evenly across the document
        total = len(all_spans)
        if total <= max_entries:
            sampled = all_spans
        else:
            step = total / max_entries
            sampled = [all_spans[int(i * step)] for i in range(max_entries)]

        summary_lines = []
        for span in sampled:
            parts = [f"Page {span.page_number}"]
            if span.y_position is not None:
                parts[0] += f" @y={span.y_position}"
            parts.append(f"{span.font_name} {span.font_size}pt")
            if span.is_bold:
                parts.append("[bold]")
            parts.append(f'- "{span.text_snippet}"')
            summary_lines.append(" ".join(parts))

        return "\n".join(summary_lines)
