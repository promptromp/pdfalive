"""File rename processor using LLM inference."""

import ntpath
import os
import time
from collections import Counter
from collections.abc import Iterable, Iterator
from pathlib import Path, PurePath, PureWindowsPath
from typing import cast

from langchain.chat_models.base import BaseChatModel
from langchain.messages import HumanMessage, SystemMessage
from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from pdfalive.console import contains_unsafe_characters, create_console
from pdfalive.models.rename import (
    ApplyRenamesResult,
    PlannedRename,
    RejectedRename,
    RenameError,
    RenameOp,
    RenameResult,
    ResolvedRenames,
)
from pdfalive.paths import entry_key, is_case_only_rename, name_key, unique_input_paths
from pdfalive.prompts import RENAME_CONTINUATION_SYSTEM_PROMPT, RENAME_SYSTEM_PROMPT
from pdfalive.retry import retry_warning
from pdfalive.tokens import TokenUsage, estimate_tokens


# Console for rich output
console = create_console()

# Default maximum tokens for filenames per batch to stay under context window limits
# Use conservative limits since filenames can be long and we need output space
DEFAULT_MAX_TOKENS_PER_BATCH = 50000

# Estimated token overhead for the prompt template (system + user message excluding filenames)
PROMPT_OVERHEAD_TOKENS = 2000

# Why a rename suggestion is rejected rather than applied.
REJECTED_UNKNOWN_INPUT = "it names a file that is not one of the inputs"
REJECTED_UNSAFE_CHARACTER = "the new name contains a control, bidi-control, invisible or undecodable character"
REJECTED_EMPTY_NAME = "the new name is empty"
REJECTED_SURROUNDING_WHITESPACE = "the new name starts or ends with whitespace"
REJECTED_DIRECTORY_NAME = "the new name refers to a directory"
REJECTED_PATH = "the new name is a path, not a plain file name"
REJECTED_RESERVED_NAME = (
    "the new name is not allowed on Windows (a reserved device name, a trailing dot or space, or a forbidden character)"
)
REJECTED_HIDDEN_NAME = "the new name starts with a dot, which would hide the file"
REJECTED_DUPLICATE_TARGET = "another suggestion gives a file the same new name"
REJECTED_TARGET_IS_INPUT = "the new name is another input's current name (swaps and chains are not supported)"

# Names that refer to a directory rather than a file.
_DIRECTORY_REFERENCES = frozenset({".", ".."})

# The running OS's path flavor, which decides what a plain file name is: on POSIX
# "X: The Unknown.pdf" and "a\b.pdf" are plain names, on Windows they are paths.
_RUNNING_PATH_FLAVOR: type[PurePath] = type(PurePath())

# The start of a name that hides the file on POSIX systems.
_HIDDEN_NAME_PREFIX = "."

# Invisible formatting and direction marks that would hide or disguise part of a file
# name (zero-width space, byte order mark, word joiner, left-to-right, right-to-left and
# Arabic letter marks), beyond the characters the console already escapes. The zero-width
# joiner and non-joiner (U+200D, U+200C) are kept: ordinary words in scripts such as
# Persian and Devanagari need them.
_INVISIBLE_NAME_MARKS = frozenset({"\u200b", "\ufeff", "\u2060", "\u200e", "\u200f", "\u061c"})


def _target_name_problem(
    output_filename: str, source: Path, path_flavor: type[PurePath] = _RUNNING_PATH_FLAVOR
) -> str | None:
    """Why output_filename cannot be source's new name in its directory, or None if it is a plain file name.

    The target is joined to the source's directory, so anything but a plain
    file name on this OS (a path, an absolute path, "..", a Windows drive)
    could move the file elsewhere. On Windows, names the OS reserves or forbids
    (ntpath.isreserved()) are rejected too. Characters that act on the terminal
    or the display, invisible marks, surrounding whitespace and a hiding
    leading dot make a name that is hard to see, type or use.

    Args:
        output_filename: The suggested new name.
        source: The file it would rename.
        path_flavor: The path rules to apply; the running OS's by default.
    """
    if contains_unsafe_characters(output_filename) or not _INVISIBLE_NAME_MARKS.isdisjoint(output_filename):
        return REJECTED_UNSAFE_CHARACTER
    if not output_filename.strip():
        return REJECTED_EMPTY_NAME
    if output_filename != output_filename.strip():
        return REJECTED_SURROUNDING_WHITESPACE
    if output_filename in _DIRECTORY_REFERENCES:
        return REJECTED_DIRECTORY_NAME
    as_path = path_flavor(output_filename)
    if as_path.anchor or as_path.parts != (output_filename,):
        return REJECTED_PATH
    if issubclass(path_flavor, PureWindowsPath) and ntpath.isreserved(output_filename):
        return REJECTED_RESERVED_NAME
    if output_filename.startswith(_HIDDEN_NAME_PREFIX) and not source.name.startswith(_HIDDEN_NAME_PREFIX):
        return REJECTED_HIDDEN_NAME
    return None


def _collision_problem(
    target_key: str, source_key: str, suggestions_per_target: Counter[str], inputs_per_key: Counter[str]
) -> str | None:
    """Why a planned rename collides, or None: compared by directory entry (pdfalive.paths.entry_key).

    Two suggestions for one target collide. So does a target that is another
    input's entry; the source's own entry (a case-only rename) is not another
    input, while an input that differs from the source only in case (two
    files on a case-sensitive volume) is.
    """
    if suggestions_per_target[target_key] > 1:
        return REJECTED_DUPLICATE_TARGET
    other_inputs = inputs_per_key[target_key] - (1 if target_key == source_key else 0)
    if other_inputs:
        return REJECTED_TARGET_IS_INPUT
    return None


def _rename_blocker(source: Path, target: Path) -> str | None:
    """Why source cannot be renamed to target now, or None if it can.

    Known limitation: another process could create the exact target between
    this check and the rename, which would then replace it. The standard
    library has no portable rename that refuses to replace (renameat2 with
    RENAME_NOREPLACE on Linux, renamex_np with RENAME_EXCL on macOS), so the
    window is narrowed, not closed.
    """
    if not os.path.lexists(source):
        return f"Source file not found: {source}"
    if os.path.lexists(target) and not is_case_only_rename(source, target):
        return f"Target file already exists: {source} -> {target}"
    return None


class DuplicateInputNameError(ValueError):
    """Different input files share a file name, and suggestions are matched to inputs by name."""


def build_input_mapping(paths: Iterable[Path]) -> dict[str, Path]:
    """Map each input's name key (pdfalive.paths.name_key) to its path, in input order.

    A file given more than once, in any spelling, is listed once. Built once
    per run and passed to both generate_renames() and resolve_renames().

    Raises:
        DuplicateInputNameError: If different files share a file name.
    """
    mapping: dict[str, Path] = {}
    for path in unique_input_paths(paths):
        key = name_key(path.name)
        if key in mapping:
            raise DuplicateInputNameError(
                f"Found duplicate filename '{path.name}' in different directories. "
                "All input files must have unique filenames."
            )
        mapping[key] = path
    return mapping


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

        def _log_retry(retry_state: RetryCallState) -> None:
            console.print(retry_warning(retry_state, MAX_RETRY_ATTEMPTS))

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

        console.print(f"  Invoking LLM for {batch_description} (~{input_tokens:,} input tokens)...", style="dim")
        start_time = time.time()

        response = _invoke()

        elapsed = time.time() - start_time
        console.print(f"  Completed {batch_description} in {elapsed:.1f}s", style="green")

        return cast(RenameResult, response)

    def generate_renames(
        self,
        inputs: dict[str, Path],
        query: str,
        *,
        usage: TokenUsage,
        max_tokens_per_batch: int = DEFAULT_MAX_TOKENS_PER_BATCH,
        request_delay: float = DEFAULT_REQUEST_DELAY_SECONDS,
    ) -> RenameResult:
        """Generate rename suggestions using LLM.

        This method handles batching automatically when filenames exceed the
        token limit, splitting them into batches and merging results.

        Args:
            inputs: The files to rename, from build_input_mapping().
            query: User's renaming instruction/query.
            usage: Where each completed LLM call is recorded as it happens, so the
                caller keeps the usage of the batches that completed if a later one fails.
            max_tokens_per_batch: Maximum tokens per LLM call (for batching).
            request_delay: Delay in seconds between LLM calls to avoid rate limiting.

        Returns:
            The suggested rename operations, from all batches.
        """
        if not inputs:
            return RenameResult(operations=[])

        # Only file names are sent, never directories.
        filenames = [path.name for path in inputs.values()]
        merged_result = RenameResult(operations=[])
        model = self.llm.with_structured_output(RenameResult)

        batches = list(self._batch_filenames(filenames, max_tokens_per_batch))
        total_batches = len(batches)

        if total_batches > 1:
            console.print(f"Processing {len(filenames)} files in {total_batches} batch(es)...", style="bold")

        for batch_idx, batch in enumerate(batches):
            is_first_batch = batch_idx == 0
            batch_description = f"batch {batch_idx + 1}/{total_batches}" if total_batches > 1 else "rename request"

            # Add delay between requests (except for the first one)
            if not is_first_batch and request_delay > 0:
                console.print(f"  Waiting {request_delay:.1f}s before next request...", style="dim")
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
                console.print(f"  Generated {ops_found} rename operation(s) in this batch", style="cyan")

            # Merge results
            merged_result = merged_result.merge(batch_result)

        if total_batches > 1:
            ops_count = len(merged_result.operations)
            console.print(f"All batches processed. Total rename operations: {ops_count}", style="bold green")

        return merged_result

    def resolve_renames(self, operations: list[RenameOp], inputs: dict[str, Path]) -> ResolvedRenames:
        """Resolve rename suggestions against the input files.

        A suggestion is planned when it names an input file (by name_key) and
        gives it a new plain file name, which keeps the file in its directory,
        that no other suggestion also gives and that is not another input's
        current name (swaps and chains would depend on the order renames are
        applied in). A suggestion that keeps the current name is dropped before
        any check, so it is never reported; other suggestions are rejected with
        the reason.

        Args:
            operations: Rename suggestions, naming files without directories.
            inputs: The input files, from build_input_mapping().

        Returns:
            The planned and the rejected renames.
        """
        resolved = ResolvedRenames()

        # Each candidate with its target's entry key, computed once.
        candidates: list[tuple[PlannedRename, str]] = []
        for op in operations:
            source = inputs.get(name_key(op.input_filename))
            if source is None:
                resolved.rejected.append(RejectedRename(op=op, reason=REJECTED_UNKNOWN_INPUT))
                continue
            if op.output_filename == source.name:
                continue
            problem = _target_name_problem(op.output_filename, source)
            if problem is not None:
                resolved.rejected.append(RejectedRename(op=op, reason=problem))
                continue
            target = source.parent / op.output_filename
            candidates.append((PlannedRename(op=op, source=source, target=target), entry_key(target)))

        # Compare directory entries, not spellings: "./a.pdf" and "/abs/a.pdf", or
        # "A.pdf" and "a.pdf", name the same entry.
        input_keys = {path: entry_key(path) for path in inputs.values()}
        inputs_per_key = Counter(input_keys.values())
        suggestions_per_target = Counter(target_key for _candidate, target_key in candidates)
        for candidate, target_key in candidates:
            problem = _collision_problem(
                target_key, input_keys[candidate.source], suggestions_per_target, inputs_per_key
            )
            if problem is None:
                resolved.planned.append(candidate)
            else:
                resolved.rejected.append(RejectedRename(op=candidate.op, reason=problem))

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
            try:
                error = _rename_blocker(source, target)
                if error is None:
                    source.rename(target)
                    result.successful.append((source, target))
                    continue
            except Exception as e:
                # One file's failure, expected or not, must not stop the others.
                error = f"Failed to rename {source} -> {target}: {e}"
            result.failed.append(RenameError(source=source, target=target, error=error))

        return result
