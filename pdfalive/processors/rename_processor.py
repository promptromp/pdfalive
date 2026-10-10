"""File rename processor using LLM inference."""

import json
import ntpath
import os
import time
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
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

from pdfalive.console import contains_unsafe_characters, create_console, visible
from pdfalive.models.rename import (
    FIRST_FILE_ID,
    ApplyRenamesResult,
    PlannedRename,
    RejectedRename,
    RenameBatch,
    RenameError,
    RenameInputs,
    RenameOp,
    RenameResult,
    ResolvedRenames,
)
from pdfalive.paths import entry_key, is_case_only_rename, name_key
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

# A file's estimated share of the prompt beyond its name: the JSON entry around it,
# '{"file_id": 12, "input_filename": "…"},' and the newline (13 with o200k_base, rounded up).
ENTRY_FRAMING_TOKENS = 16
# A file's estimated share of the answer: its id, its name echoed back, its new name, a
# confidence and a short reasoning. Counted against the batch's token budget and its
# output cap.
ESTIMATED_OUTPUT_TOKENS_PER_FILE = 80
# The most output one request may ask for, within the output limits of current models,
# so a batch never asks for more suggestions than the model can write.
DEFAULT_MAX_OUTPUT_TOKENS_PER_BATCH = 16_000

# Why a rename suggestion is rejected rather than applied.
REJECTED_UNKNOWN_INPUT = "its file id names none of the files its request listed"
REJECTED_NAME_MISMATCH = "its file name is not the name of the file its id names"
REJECTED_MULTIPLE_SUGGESTIONS = "there are differing suggestions for this file"
REJECTED_COPIED_ESCAPE = "the new name copies an escape that stood for an unreadable character in the current name"
REJECTED_UNSAFE_CHARACTER = "the new name contains a control, bidi-control, invisible or undecodable character"
REJECTED_EMPTY_NAME = "the new name is empty"
REJECTED_SURROUNDING_WHITESPACE = "the new name starts or ends with whitespace"
REJECTED_DIRECTORY_NAME = "the new name refers to a directory"
REJECTED_PATH = "the new name is a path, not a plain file name"
REJECTED_RESERVED_NAME = (
    "the new name is not allowed on Windows (a reserved device name, a trailing dot or space, or a forbidden character)"
)
REJECTED_HIDDEN_NAME = "the new name starts with a dot, which would hide the file"
REJECTED_DUPLICATE_TARGET = "another suggestion gives a file in the same directory the same new name"
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


def _match_problem(op: RenameOp, source: Path, suggestions_for_file: int) -> str | None:
    """Why a suggestion cannot be trusted to name source, the file its id names in its request, or None.

    The suggestion must be the only one for that file (identical repeats are
    counted once), and its echoed name must be the file's name as sent: the
    visible form, with unsafe characters escaped, compared by name key.
    """
    if suggestions_for_file > 1:
        return REJECTED_MULTIPLE_SUGGESTIONS
    if name_key(op.input_filename) != name_key(visible(source.name)):
        return REJECTED_NAME_MISMATCH
    return None


# What starts each escape the visible form of a name shows an unreadable character as.
_ESCAPE_PREFIX = "\\"


def _copied_escape_problem(output_filename: str, source: Path) -> str | None:
    """Why output_filename looks like it copies an escape from source's name as sent, or None.

    A name with unsafe characters is sent in its visible form, where each one
    is an escape such as \\n or \\xe9. A new name with a backslash for such a
    file most likely copies one: on POSIX that is a legal name, but the
    character would become literal text that the preview shows the same way.
    """
    if visible(source.name) != source.name and _ESCAPE_PREFIX in output_filename:
        return REJECTED_COPIED_ESCAPE
    return None


# How the prompt's JSON array of files opens, separates and closes its entries.
_FILES_ARRAY_OPEN = "[\n"
_FILES_ARRAY_SEPARATOR = ",\n"
_FILES_ARRAY_CLOSE = "\n]"


def _file_entry(file_id: int, sent_name: str) -> str:
    """A file's entry in the prompt's JSON array of files, keyed like the RenameOp fields the LLM echoes.

    JSON quotes the name, so a name cannot break out of the list (with a
    newline, say) and read as instructions. Non-ASCII names stay readable
    text (ensure_ascii=False); sent_name already has its unsafe characters
    (controls, bidi controls, undecodable bytes) in their visible escaped
    form, so the model never has to write a lone-surrogate escape back.
    """
    return json.dumps({"file_id": file_id, "input_filename": sent_name}, ensure_ascii=False)


@dataclass(frozen=True)
class _Request:
    """The files one LLM request lists, in id order, each with its name as sent."""

    paths: tuple[Path, ...]
    sent_names: tuple[str, ...]

    def files_array(self) -> str:
        """The request's JSON array of files, ids counting from FIRST_FILE_ID; each entry serialized once."""
        entries = (_file_entry(file_id, name) for file_id, name in enumerate(self.sent_names, start=FIRST_FILE_ID))
        return _FILES_ARRAY_OPEN + _FILES_ARRAY_SEPARATOR.join(entries) + _FILES_ARRAY_CLOSE


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

    def _batch_files(
        self,
        paths: Iterable[Path],
        max_tokens: int = DEFAULT_MAX_TOKENS_PER_BATCH,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS_PER_BATCH,
    ) -> list[_Request]:
        """Split the files into requests that fit the token budget and the output cap.

        Each file costs its sent name's tokens, ENTRY_FRAMING_TOKENS and
        ESTIMATED_OUTPUT_TOKENS_PER_FILE against max_tokens (less the prompt
        overhead), and ESTIMATED_OUTPUT_TOKENS_PER_FILE against max_output_tokens.
        A request always takes at least one file. Entries are serialized later,
        once, with their ids in their request (_Request.files_array()).
        """
        input_budget = max_tokens - PROMPT_OVERHEAD_TOKENS
        requests: list[_Request] = []
        current_paths: list[Path] = []
        current_names: list[str] = []
        current_tokens = 0
        for path in paths:
            sent_name = visible(path.name)
            file_tokens = estimate_tokens(sent_name) + ENTRY_FRAMING_TOKENS + ESTIMATED_OUTPUT_TOKENS_PER_FILE
            output_tokens = (len(current_paths) + 1) * ESTIMATED_OUTPUT_TOKENS_PER_FILE
            if current_paths and (current_tokens + file_tokens > input_budget or output_tokens > max_output_tokens):
                requests.append(_Request(tuple(current_paths), tuple(current_names)))
                current_paths, current_names, current_tokens = [], [], 0
            current_paths.append(path)
            current_names.append(sent_name)
            current_tokens += file_tokens
        if current_paths:
            requests.append(_Request(tuple(current_paths), tuple(current_names)))
        return requests

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
        inputs: RenameInputs,
        query: str,
        *,
        usage: TokenUsage,
        max_tokens_per_batch: int = DEFAULT_MAX_TOKENS_PER_BATCH,
        max_output_tokens_per_batch: int = DEFAULT_MAX_OUTPUT_TOKENS_PER_BATCH,
        request_delay: float = DEFAULT_REQUEST_DELAY_SECONDS,
    ) -> list[RenameBatch]:
        """Generate rename suggestions using LLM.

        This method handles batching automatically, splitting the files into
        requests that fit the token budget and the output cap. File ids count
        from FIRST_FILE_ID in every request, and each request's suggestions are
        kept with the files it listed, for resolve_renames().

        Args:
            inputs: The files to rename.
            query: User's renaming instruction/query.
            usage: Where each completed LLM call is recorded as it happens, so the
                caller keeps the usage of the batches that completed if a later one fails.
            max_tokens_per_batch: Maximum tokens per LLM call, its expected answer included.
            max_output_tokens_per_batch: Maximum expected answer tokens per LLM call.
            request_delay: Delay in seconds between LLM calls to avoid rate limiting.

        Returns:
            Each request's suggestions, with the files it listed.
        """
        if not inputs:
            return []

        results: list[RenameBatch] = []
        model = self.llm.with_structured_output(RenameResult)

        # Only file names are sent, never directories.
        requests = self._batch_files(inputs.paths, max_tokens_per_batch, max_output_tokens_per_batch)
        total_batches = len(requests)

        if total_batches > 1:
            console.print(f"Processing {len(inputs)} files in {total_batches} batch(es)...", style="bold")

        for batch_idx, request in enumerate(requests):
            is_first_batch = batch_idx == 0
            batch_description = f"batch {batch_idx + 1}/{total_batches}" if total_batches > 1 else "rename request"

            # Add delay between requests (except for the first one)
            if not is_first_batch and request_delay > 0:
                console.print(f"  Waiting {request_delay:.1f}s before next request...", style="dim")
                time.sleep(request_delay)

            # Select appropriate prompt
            system_prompt = RENAME_SYSTEM_PROMPT if is_first_batch else RENAME_CONTINUATION_SYSTEM_PROMPT

            # Build user message with batch context
            batch_context = ""
            if total_batches > 1:
                batch_context = f"\n\n**Batch information**: This is batch {batch_idx + 1} of {total_batches}."
                if batch_idx > 0:
                    # For sequential numbering, tell the model where to start
                    files_processed = sum(len(requests[i].paths) for i in range(batch_idx))
                    batch_context += f" ({files_processed} files have been processed in previous batches.)"

            user_content = f"""Please rename the following files according to the instruction below.

## Files to rename (a JSON array of file_id and input_filename; ids count from {FIRST_FILE_ID} here):
{request.files_array()}

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

            results.append(RenameBatch(files=request.paths, operations=tuple(batch_result.operations)))

        if total_batches > 1:
            ops_count = sum(len(result.operations) for result in results)
            console.print(f"All batches processed. Total rename operations: {ops_count}", style="bold green")

        return results

    def resolve_renames(self, batches: Iterable[RenameBatch], inputs: RenameInputs) -> ResolvedRenames:
        """Resolve rename suggestions against the input files, matching each to its file by its request's id.

        A suggestion is planned when it can be trusted to name a file its request
        listed (_match_problem) and gives the file a new plain file name, which
        keeps it in its directory, that no other suggestion gives a file in the
        same directory and that is not another input's current name (swaps and
        chains would depend on the order renames are applied in). A suggestion
        that keeps the current name is dropped before the name checks, so it is
        never reported; other suggestions are rejected with the reason.

        Args:
            batches: Each LLM request's suggestions, with the files it listed.
            inputs: The files to rename.

        Returns:
            The planned and the rejected renames.
        """
        resolved = ResolvedRenames()

        # Each candidate with its target's entry key, computed once.
        candidates: list[tuple[PlannedRename, str]] = []
        for batch in batches:
            # An identical suggestion repeated is one suggestion; only differing ones are ambiguous.
            operations = tuple(dict.fromkeys(batch.operations))
            suggestions_per_file = Counter(op.file_id for op in operations)
            for op in operations:
                source = batch.path(op.file_id)
                if source is None:
                    resolved.rejected.append(RejectedRename(op=op, reason=REJECTED_UNKNOWN_INPUT, source=None))
                    continue
                problem = _match_problem(op, source, suggestions_per_file[op.file_id])
                if problem is None and op.output_filename == source.name:
                    continue
                problem = (
                    problem
                    or _target_name_problem(op.output_filename, source)
                    or _copied_escape_problem(op.output_filename, source)
                )
                if problem is not None:
                    resolved.rejected.append(RejectedRename(op=op, reason=problem, source=source))
                    continue
                target = source.parent / op.output_filename
                candidates.append((PlannedRename(op=op, source=source, target=target), entry_key(target)))

        # Compare directory entries, not spellings: "./a.pdf" and "/abs/a.pdf", or
        # "A.pdf" and "a.pdf", name the same entry.
        input_keys = {path: entry_key(path) for path in inputs.paths}
        inputs_per_key = Counter(input_keys.values())
        suggestions_per_target = Counter(target_key for _candidate, target_key in candidates)
        for candidate, target_key in candidates:
            problem = _collision_problem(
                target_key, input_keys[candidate.source], suggestions_per_target, inputs_per_key
            )
            if problem is None:
                resolved.planned.append(candidate)
            else:
                resolved.rejected.append(RejectedRename(op=candidate.op, reason=problem, source=candidate.source))

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
