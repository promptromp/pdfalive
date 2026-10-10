"""The retry policy for LLM calls, shared by every processor that makes them.

An LLM call is retried only for conditions that can succeed on a second try
(an allowlist), with exponential backoff: up to MAX_RETRY_ATTEMPTS attempts,
waiting RETRY_INITIAL_WAIT_SECONDS before the first retry and doubling the wait
before each later one (10, 20, 40 and 80 s for the four retries of five
attempts). A provider's Retry-After hint lengthens a wait, up to
RETRY_MAX_WAIT_SECONDS. Each retry is reported on the caller's console.

Retryable:
- a LangChain ``ModelError`` whose ``is_retryable`` is true (rate limits, server
  errors, connection failures, timeouts). The OpenAI and Anthropic integrations
  raise these, so their own classification is used whenever it is available.
  Those SDKs make their requests with httpx2 and wrap its transport errors
  (``APIConnectionError``, ``APITimeoutError``), which LangChain wraps again
  as ``ModelConnectionError`` / ``ModelTimeoutError``, so no httpx2 error
  reaches this policy unwrapped.
- an error carrying an HTTP status of 408, 409, 429 or 5xx, read from its
  ``status_code`` or its response's.
- a connection failure or timeout without a status: built-in ``ConnectionError``
  or ``TimeoutError``, or an httpx network/timeout error (the Ollama client
  uses httpx and raises these).
- an Ollama ``ResponseError`` with no HTTP status (its -1 sentinel), which the
  client raises for a server error in the middle of a streamed answer.
- a ``StructuredOutputError``: the model's answer did not parse into the
  requested schema (see pdfalive.structured_output), and the next one may. Such
  a failure is likely to repeat, so a call gets at most MAX_PARSE_ATTEMPTS of
  them, whatever other failures come between.

Everything else surfaces on the first attempt: client errors (other 4xx such as
context overflow, an unsupported reasoning effort or bad credentials), a 429
that means the account's quota is used up (``insufficient_quota``: waiting does
not refill it), programming errors (``TypeError``, a pydantic
``ValidationError`` raised outside the structured-output boundary), a
truncated completion (``LengthFinishReasonError``), a model refusal, a cassette
replay mismatch (``CassetteMissError``), and exceptions that are not
``Exception``s (Ctrl-C, ``SystemExit``). For OpenAI and Anthropic this is the
only retry layer: ``build_chat_model`` disables those SDKs' own retries.

``invoke_with_retry()`` raises ``LLMCallError`` for any ``Exception`` the policy
gives up on, so a command can report a failed call in one place; Ctrl-C and
``SystemExit`` pass through unwrapped.
"""

import sys
import time
from collections.abc import Callable
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime

import httpx
from langchain_core.exceptions import ModelError
from rich.text import Text
from tenacity import RetryCallState, retry, retry_if_exception
from tenacity.stop import stop_base
from tenacity.wait import wait_exponential

from pdfalive.console import ConsoleType, safe_text


MAX_RETRY_ATTEMPTS = 5
# Attempts for an answer that does not parse: a malformed answer tends to repeat.
MAX_PARSE_ATTEMPTS = 3
RETRY_INITIAL_WAIT_SECONDS = 10.0
RETRY_MAX_WAIT_SECONDS = 120.0


class LLMCallError(Exception):
    """An LLM call failed for good: its error is not retryable, or it kept failing through the policy's attempts.

    The failure is the exception's ``__cause__``.
    """


class StructuredOutputError(Exception):
    """The model's answer did not parse into the requested schema, or held no parsed result.

    The next answer may parse, so the policy retries this, up to MAX_PARSE_ATTEMPTS attempts.
    """


# HTTP statuses of transient conditions: request timeout, conflict, rate limit, and server errors.
_RETRYABLE_STATUSES = frozenset({408, 409, 429})
_SERVER_ERROR_STATUSES = range(500, 600)
# Statuses outside this range (e.g. Ollama's -1 sentinel) mean the error has no HTTP status.
_HTTP_STATUSES = range(100, 600)
# The error code of a 429 that means the account's quota is used up, not a rate limit (OpenAI).
_INSUFFICIENT_QUOTA_CODE = "insufficient_quota"
# The Ollama client's module and its error type, looked up only once something has imported
# them (an Ollama error cannot exist before), and the status it reports without an HTTP one.
_OLLAMA_MODULE = "ollama"
_OLLAMA_RESPONSE_ERROR = "ResponseError"
_OLLAMA_NO_STATUS = -1

# Errors without an HTTP status that are retried: transient connection failures and timeouts,
# and answers that did not parse.
_RETRYABLE_ERRORS: tuple[type[Exception], ...] = (
    ConnectionError,
    TimeoutError,
    httpx.NetworkError,
    httpx.TimeoutException,
    httpx.RemoteProtocolError,
    StructuredOutputError,
)

# Response headers through which providers suggest a wait, in milliseconds or seconds.
_RETRY_AFTER_MS_HEADER = "retry-after-ms"
_RETRY_AFTER_HEADER = "retry-after"
_MILLISECONDS_PER_SECOND = 1000


def _http_status(exception: Exception) -> int | None:
    """The error's HTTP status, from its ``status_code`` or its response's; None if it has none. Never raises.

    SDK errors carry ``status_code`` themselves; ``httpx.HTTPStatusError`` keeps it on its response. An
    out-of-range value (Ollama's -1 sentinel) counts as no status.
    """
    try:
        status = getattr(exception, "status_code", None)
        if status is None:
            status = getattr(getattr(exception, "response", None), "status_code", None)
    except Exception:
        return None
    return status if isinstance(status, int) and status in _HTTP_STATUSES else None


def _is_quota_exhausted(exception: Exception) -> bool:
    """Whether the error reports a used-up quota (OpenAI's ``insufficient_quota`` code); never raises."""
    try:
        return getattr(exception, "code", None) == _INSUFFICIENT_QUOTA_CODE
    except Exception:
        return False


def _is_ollama_stream_error(exception: Exception) -> bool:
    """Whether the error is Ollama's for a server error mid-stream: a ResponseError with no HTTP status."""
    module = sys.modules.get(_OLLAMA_MODULE)
    response_error = getattr(module, _OLLAMA_RESPONSE_ERROR, None)
    if not isinstance(response_error, type) or not isinstance(exception, response_error):
        return False
    return getattr(exception, "status_code", None) == _OLLAMA_NO_STATUS


def is_retryable_error(exception: BaseException) -> bool:
    """Whether a failed LLM call may succeed if retried; never raises (see the module docstring)."""
    if not isinstance(exception, Exception):
        return False
    if _is_quota_exhausted(exception):
        return False
    if isinstance(exception, ModelError):
        return exception.is_retryable
    if _is_ollama_stream_error(exception):
        return True
    status = _http_status(exception)
    if status is not None:
        return status in _RETRYABLE_STATUSES or status in _SERVER_ERROR_STATUSES
    return isinstance(exception, _RETRYABLE_ERRORS)


def _header(exception: BaseException | None, name: str) -> str | None:
    """A header of the error's HTTP response, if it has one; never raises."""
    try:
        headers = getattr(getattr(exception, "response", None), "headers", None)
        value = headers.get(name) if headers is not None else None
    except Exception:
        return None
    return value if isinstance(value, str) else None


def _milliseconds_hint(value: str) -> float | None:
    """Seconds from a ``retry-after-ms`` value, or None if it is not a number."""
    try:
        return float(value) / _MILLISECONDS_PER_SECOND
    except ValueError:
        return None


def _retry_after_hint(value: str) -> float | None:
    """Seconds from a ``retry-after`` value (a number of seconds or an HTTP date), or None if it is neither."""
    try:
        return float(value)
    except ValueError:
        pass
    try:
        return (parsedate_to_datetime(value) - datetime.now(UTC)).total_seconds()
    except (TypeError, ValueError):
        return None


# Wait hints in the order they are consulted, each header parsed on its own, so a
# malformed retry-after-ms does not hide a usable retry-after.
_RETRY_AFTER_HINTS: tuple[tuple[str, Callable[[str], float | None]], ...] = (
    (_RETRY_AFTER_MS_HEADER, _milliseconds_hint),
    (_RETRY_AFTER_HEADER, _retry_after_hint),
)


def _retry_after_seconds(exception: BaseException | None) -> float | None:
    """The wait a provider suggested in the error's response headers, if any; never raises."""
    for name, parse in _RETRY_AFTER_HINTS:
        value = _header(exception, name)
        seconds = parse(value) if value is not None else None
        if seconds is not None:
            return seconds
    return None


def _last_exception(retry_state: RetryCallState) -> BaseException | None:
    """The exception the last attempt raised, or None if it has not failed."""
    return retry_state.outcome.exception() if retry_state.outcome is not None else None


_backoff = wait_exponential(multiplier=RETRY_INITIAL_WAIT_SECONDS, max=RETRY_MAX_WAIT_SECONDS)


def retry_wait_seconds(retry_state: RetryCallState) -> float:
    """The wait before the next attempt: the backoff, lengthened to a provider's Retry-After hint up to the cap."""
    backoff = _backoff(retry_state)
    hint = _retry_after_seconds(_last_exception(retry_state))
    if hint is None:
        return backoff
    return min(max(backoff, hint), RETRY_MAX_WAIT_SECONDS)


class AttemptLimit(stop_base):
    """The policy's stop condition: MAX_RETRY_ATTEMPTS attempts, of which at most MAX_PARSE_ATTEMPTS fail to parse.

    Answers that failed to parse are counted across the call's attempts, so
    the parse limit holds however transient failures interleave with them,
    and a parse failure after transient ones still gets its retries. The
    count is per call: it restarts at a call's first attempt.
    """

    def __init__(self) -> None:
        self._parse_failures = 0

    def __call__(self, retry_state: RetryCallState) -> bool:
        if retry_state.attempt_number == 1:
            self._parse_failures = 0
        if isinstance(_last_exception(retry_state), StructuredOutputError):
            self._parse_failures += 1
        return retry_state.attempt_number >= MAX_RETRY_ATTEMPTS or self._parse_failures >= MAX_PARSE_ATTEMPTS

    def progress(self, retry_state: RetryCallState) -> str:
        """Where the call stands against its limits, for the retry warning: "attempt 2/5", plus unparsed answers."""
        progress = f"attempt {retry_state.attempt_number}/{MAX_RETRY_ATTEMPTS}"
        if self._parse_failures:
            progress += f", unparsed answer {self._parse_failures}/{MAX_PARSE_ATTEMPTS}"
        return progress


def _exception_message(exc: BaseException) -> str:
    """The exception's text: ``str(exc)``, else its ``message`` attribute, else ``""``; never raises."""
    try:
        message = str(exc)
        if not message:
            # Some SDK errors stringify to "" and keep their text in a `message`
            # attribute of any type. Compare it with None only: the truth value
            # of some types (e.g. arrays) raises.
            attribute = getattr(exc, "message", None)
            message = "" if attribute is None else str(attribute)
    except Exception:
        # A broken __str__ must not replace the error being retried.
        return ""
    return message


def describe_exception(exc: BaseException) -> str:
    """Describe an exception in one line: ``TypeName: message``, or ``TypeName`` without a message.

    Never raises, since it runs inside tenacity's ``before_sleep`` callback.
    """
    message = _exception_message(exc)
    name = type(exc).__name__
    return f"{name}: {message}" if message else name


def retry_warning(retry_state: RetryCallState) -> Text:
    """The console warning shown before tenacity sleeps ahead of the next attempt.

    Under the policy's AttemptLimit the progress shows the attempt total and any unparsed answers.
    """
    wait_seconds = retry_state.next_action.sleep if retry_state.next_action is not None else 0.0
    exception = _last_exception(retry_state)
    details = f" ({describe_exception(exception)})" if exception is not None else ""
    stop = retry_state.retry_object.stop
    progress = stop.progress(retry_state) if isinstance(stop, AttemptLimit) else f"attempt {retry_state.attempt_number}"
    message = f"Error encountered{details}. Retrying in {wait_seconds:.1f}s ({progress})..."
    return safe_text("  ", (message, "yellow"), multiline=True)


def llm_retry[**P, R](console: ConsoleType) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorate an LLM call with the retry policy, reporting each retry on console.

    The last error is re-raised once the attempts run out or it is not retryable.
    """

    def report_retry(retry_state: RetryCallState) -> None:
        console.print(retry_warning(retry_state))

    return retry(
        retry=retry_if_exception(is_retryable_error),
        stop=AttemptLimit(),
        wait=retry_wait_seconds,
        before_sleep=report_retry,
        reraise=True,
    )


def invoke_with_retry[R](console: ConsoleType, call: Callable[[], R], *, description: str, input_tokens: int) -> R:
    """Make an LLM call under the retry policy, reporting it and its duration on console.

    Args:
        console: Where to report the call, its retries and its completion.
        call: The LLM call; it runs again on each retry.
        description: What the call is for, e.g. "batch 1/3".
        input_tokens: Estimated input tokens, for the report.

    Returns:
        The call's result.

    Raises:
        LLMCallError: The call failed for good, from the last error.
    """
    console.print(f"  Invoking LLM for {description} (~{input_tokens:,} input tokens)...", style="dim")
    started = time.monotonic()
    try:
        result = llm_retry(console)(call)()
    except Exception as error:
        raise LLMCallError(f"LLM call for {description} failed: {describe_exception(error)}") from error
    console.print(f"  Completed {description} in {time.monotonic() - started:.1f}s", style="green")
    return result
