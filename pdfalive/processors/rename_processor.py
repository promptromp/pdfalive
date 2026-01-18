"""File rename processor using LLM inference."""

import time
from collections.abc import Iterator
from pathlib import Path
from typing import cast

from langchain.chat_models.base import BaseChatModel
from langchain.messages import HumanMessage, SystemMessage
from rich.console import Console
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from pdfalive.models.rename import ApplyRenamesResult, RenameError, RenameOp, RenameResult
from pdfalive.prompts import RENAME_CONTINUATION_SYSTEM_PROMPT, RENAME_SYSTEM_PROMPT
from pdfalive.tokens import TokenUsage, estimate_tokens


# Console for rich output
console = Console()

# Default maximum tokens for filenames per batch to stay under context window limits
# Use conservative limits since filenames can be long and we need output space
DEFAULT_MAX_TOKENS_PER_BATCH = 50000

# Estimated token overhead for the prompt template (system + user message excluding filenames)
PROMPT_OVERHEAD_TOKENS = 2000

# Delay between LLM calls (in seconds) to avoid rate limiting
DEFAULT_REQUEST_DELAY_SECONDS = 2.0

# Retry configuration for rate-limited requests
MAX_RETRY_ATTEMPTS = 5
RETRY_MULTIPLIER = 2  # Exponential backoff multiplier
RETRY_MIN_WAIT_SECONDS = 10  # Minimum wait time between retries
RETRY_MAX_WAIT_SECONDS = 120  # Maximum wait time between retries


class RenameProcessor:
    """Processor for intelligent file renaming using LLM."""

    def __init__(self, llm: BaseChatModel) -> None:
        """Initialize the rename processor.

        Args:
            llm: LangChain chat model for rename inference.
        """
        self.llm = llm

    def _extract_filenames(self, paths: list[Path]) -> list[str]:
        """Extract filenames from full paths.

        Args:
            paths: List of file paths.

        Returns:
            List of filenames (without directory components).
        """
        return [path.name for path in paths]

    def _build_path_mapping(self, paths: list[Path]) -> dict[str, Path]:
        """Build a mapping from filename to full path.

        Args:
            paths: List of file paths.

        Returns:
            Dictionary mapping filename to full path.

        Raises:
            ValueError: If duplicate filenames are found.
        """
        mapping: dict[str, Path] = {}
        for path in paths:
            filename = path.name
            if filename in mapping:
                raise ValueError(
                    f"Found duplicate filename '{filename}' in different directories. "
                    "All input files must have unique filenames."
                )
            mapping[filename] = path
        return mapping

    def _batch_filenames(
        self,
        filenames: list[str],
        max_tokens: int = DEFAULT_MAX_TOKENS_PER_BATCH,
    ) -> Iterator[list[str]]:
        """Split filenames into batches that fit within token limits.

        Args:
            filenames: List of filenames to batch.
            max_tokens: Maximum estimated tokens per batch (for filenames only).

        Yields:
            Batches of filenames, each estimated to be under max_tokens.
        """
        if not filenames:
            yield []
            return

        # Account for prompt overhead in the effective limit
        effective_max_tokens = max_tokens - PROMPT_OVERHEAD_TOKENS

        current_batch: list[str] = []
        current_tokens = 0

        for filename in filenames:
            # Estimate tokens for this filename (including list formatting overhead)
            filename_str = f"- {filename}\n"
            filename_tokens = estimate_tokens(filename_str)

            # If adding this filename would exceed limit and we have content, yield current batch
            if current_tokens + filename_tokens > effective_max_tokens and current_batch:
                yield current_batch
                current_batch = []
                current_tokens = 0

            current_batch.append(filename)
            current_tokens += filename_tokens

        # Yield final batch if non-empty
        if current_batch:
            yield current_batch

    def _invoke_with_retry(
        self,
        model,
        messages,
        batch_description: str,
        input_tokens: int,
    ) -> RenameResult:
        """Invoke the LLM with retry logic for rate limiting.

        Args:
            model: The LLM model with structured output.
            messages: The messages to send.
            batch_description: Description of the current batch for logging.
            input_tokens: Estimated input tokens for logging.

        Returns:
            The RenameResult response from the LLM.
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

        return cast(RenameResult, response)

    def generate_renames(
        self,
        paths: list[Path],
        query: str,
        max_tokens_per_batch: int = DEFAULT_MAX_TOKENS_PER_BATCH,
        request_delay: float = DEFAULT_REQUEST_DELAY_SECONDS,
    ) -> tuple[RenameResult, TokenUsage]:
        """Generate rename suggestions using LLM.

        This method handles batching automatically when filenames exceed the
        token limit, splitting them into batches and merging results.

        Args:
            paths: List of file paths to rename.
            query: User's renaming instruction/query.
            max_tokens_per_batch: Maximum tokens per LLM call (for batching).
            request_delay: Delay in seconds between LLM calls to avoid rate limiting.

        Returns:
            Tuple of (RenameResult containing suggested rename operations, TokenUsage statistics).
        """
        if not paths:
            return RenameResult(operations=[]), TokenUsage()

        filenames = self._extract_filenames(paths)

        return self._generate_renames_batched(
            filenames=filenames,
            query=query,
            max_tokens_per_batch=max_tokens_per_batch,
            request_delay=request_delay,
        )

    def _generate_renames_batched(
        self,
        filenames: list[str],
        query: str,
        max_tokens_per_batch: int = DEFAULT_MAX_TOKENS_PER_BATCH,
        request_delay: float = DEFAULT_REQUEST_DELAY_SECONDS,
    ) -> tuple[RenameResult, TokenUsage]:
        """Generate rename suggestions with batching support.

        Args:
            filenames: List of filenames to rename.
            query: User's renaming instruction/query.
            max_tokens_per_batch: Maximum tokens per LLM call.
            request_delay: Delay in seconds between LLM calls.

        Returns:
            Tuple of (merged RenameResult, TokenUsage statistics).
        """
        usage = TokenUsage()
        merged_result = RenameResult(operations=[])
        model = self.llm.with_structured_output(RenameResult)

        batches = list(self._batch_filenames(filenames, max_tokens_per_batch))
        total_batches = len(batches)

        if total_batches > 1:
            console.print(f"[bold]Processing {len(filenames)} files in {total_batches} batch(es)...[/bold]")

        for batch_idx, batch in enumerate(batches):
            is_first_batch = batch_idx == 0
            batch_description = f"batch {batch_idx + 1}/{total_batches}" if total_batches > 1 else "rename request"

            # Add delay between requests (except for the first one)
            if not is_first_batch and request_delay > 0:
                console.print(f"  [dim]Waiting {request_delay:.1f}s before next request...[/dim]")
                time.sleep(request_delay)

            # Select appropriate prompt
            system_prompt = RENAME_SYSTEM_PROMPT if is_first_batch else RENAME_CONTINUATION_SYSTEM_PROMPT

            # Build user message with batch context
            filenames_list = "\n".join(f"- {filename}" for filename in batch)

            batch_context = ""
            if total_batches > 1:
                batch_context = f"\n\n**Batch information**: This is batch {batch_idx + 1} of {total_batches}."
                if batch_idx > 0:
                    # For sequential numbering, tell the model where to start
                    files_processed = sum(len(batches[i]) for i in range(batch_idx))
                    batch_context += f" ({files_processed} files have been processed in previous batches.)"

            user_content = f"""Please rename the following files according to the instruction below.

## Files to rename:
{filenames_list}

## Renaming instruction:
{query}{batch_context}
"""

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_content),
            ]

            # Estimate input tokens
            input_text = system_prompt + user_content
            input_tokens = estimate_tokens(input_text)

            # Make LLM call with retry logic
            batch_result = self._invoke_with_retry(model, messages, batch_description, input_tokens)

            # Estimate output tokens (rough estimate based on response)
            output_tokens = estimate_tokens(str([op.model_dump() for op in batch_result.operations]))

            # Record token usage
            usage.add_call(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                description=f"Batch {batch_idx + 1}/{total_batches}" if total_batches > 1 else "Rename request",
            )

            # Log operations found
            ops_found = len(batch_result.operations)
            if ops_found > 0 and total_batches > 1:
                console.print(f"  [cyan]Generated {ops_found} rename operation(s) in this batch[/cyan]")

            # Merge results
            merged_result = merged_result.merge(batch_result)

        if total_batches > 1:
            ops_count = len(merged_result.operations)
            console.print(f"[bold green]All batches processed. Total rename operations: {ops_count}[/bold green]")

        return merged_result, usage

    def _resolve_full_paths(
        self,
        operations: list[RenameOp],
        original_paths: list[Path],
    ) -> list[tuple[Path, Path]]:
        """Resolve rename operations to full source/target paths.

        Args:
            operations: List of rename operations with filenames only.
            original_paths: Original list of input paths.

        Returns:
            List of (source_path, target_path) tuples. No-op renames (where
            source equals target) are filtered out.
        """
        path_mapping = self._build_path_mapping(original_paths)
        resolved: list[tuple[Path, Path]] = []

        for op in operations:
            if op.input_filename not in path_mapping:
                # Skip operations for files that don't exist in our input
                continue

            source_path = path_mapping[op.input_filename]
            target_path = source_path.parent / op.output_filename

            # Skip no-op renames where source and target are the same
            if source_path == target_path:
                continue

            resolved.append((source_path, target_path))

        return resolved

    def apply_renames(self, renames: list[tuple[Path, Path]]) -> ApplyRenamesResult:
        """Apply rename operations to files.

        This method is resilient: it attempts all renames and continues even if
        some fail. Errors are collected and returned in the result.

        Args:
            renames: List of (source_path, target_path) tuples.

        Returns:
            ApplyRenamesResult containing successful and failed operations.
        """
        result = ApplyRenamesResult()

        for source, target in renames:
            # Validate and apply each rename individually
            if not source.exists():
                result.failed.append(
                    RenameError(
                        source=source,
                        target=target,
                        error=f"Source file not found: {source}",
                    )
                )
                continue

            if target.exists():
                result.failed.append(
                    RenameError(
                        source=source,
                        target=target,
                        error=f"Target file already exists: {source} -> {target}",
                    )
                )
                continue

            try:
                source.rename(target)
                result.successful.append((source, target))
            except OSError as e:
                result.failed.append(
                    RenameError(
                        source=source,
                        target=target,
                        error=f"Failed to rename {source} -> {target}: {e}",
                    )
                )

        return result
