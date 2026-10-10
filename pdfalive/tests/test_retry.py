"""Tests for the LLM retry policy and its reporting."""

from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from email.utils import format_datetime
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import httpx2
import ollama
import pytest
from langchain_anthropic.chat_models import AnthropicConnectionError, AnthropicTimeoutError
from langchain_core.exceptions import (
    ContextOverflowError,
    ModelAPIError,
    ModelAuthenticationError,
    ModelConnectionError,
    ModelInvalidRequestError,
    ModelRateLimitError,
    ModelTimeoutError,
)
from langchain_openai.chat_models.base import OpenAIConnectionError, OpenAIRateLimitError, OpenAITimeoutError
from openai import LengthFinishReasonError
from openai.types.chat import ChatCompletion
from rich.console import Console
from tenacity import RetryAction, RetryCallState, Retrying, stop_after_delay
from tenacity.stop import stop_base

from pdfalive.evaluation.cassette import CassetteMissError
from pdfalive.retry import (
    MAX_PARSE_ATTEMPTS,
    MAX_RETRY_ATTEMPTS,
    RETRY_INITIAL_WAIT_SECONDS,
    RETRY_MAX_WAIT_SECONDS,
    AttemptLimit,
    LLMCallError,
    StructuredOutputError,
    describe_exception,
    invoke_with_retry,
    is_retryable_error,
    llm_retry,
    retry_wait_seconds,
    retry_warning,
)


ERROR_TEXT = "rate limited"
SLEEP_SECONDS = 12.5
ATTEMPT_NUMBER = 2
ARRAY_LIKE_TEXT = "[429 503]"
# What a successful call returns.
RESULT = "parsed response"
# Ollama's ResponseError reports this when there is no HTTP status.
OLLAMA_NO_STATUS = -1
# A client error and a server error status, for errors built with one.
NON_RETRYABLE_STATUS = 400
SERVER_ERROR_STATUS = 500
# The attempts after which the policy waits: every attempt but the last.
BACKOFF_ATTEMPTS = range(1, MAX_RETRY_ATTEMPTS)
# A Retry-After hint longer than the first backoff wait and within the cap, in both header units.
RETRY_AFTER_SECONDS = 30.0
RETRY_AFTER_MILLISECONDS = timedelta(seconds=RETRY_AFTER_SECONDS) / timedelta(milliseconds=1)
# A Retry-After hint shorter than the first backoff wait.
SHORT_RETRY_AFTER_SECONDS = RETRY_INITIAL_WAIT_SECONDS / 10
UNPARSABLE_HINT = "not a number or date"
# A provider endpoint, for real SDK exceptions (nothing is sent).
API_URL = "https://api.example.com/v1/chat"
# OpenAI's error codes for a used-up quota and for an ordinary rate limit, both sent with HTTP 429.
INSUFFICIENT_QUOTA_CODE = "insufficient_quota"
RATE_LIMIT_CODE = "rate_limit_exceeded"
RATE_LIMITED = 429


class _UnprintableError(Exception):
    """An exception whose __str__ raises."""

    def __str__(self) -> str:
        raise RuntimeError("cannot render this error")


class _AmbiguousTruthValue:
    """Array-like: printable, but asking for its truth value raises, like a numpy array's."""

    def __bool__(self) -> bool:
        raise ValueError("The truth value of an array with more than one element is ambiguous")

    def __str__(self) -> str:
        return ARRAY_LIKE_TEXT


class _SentinelStatusTimeout(TimeoutError):
    """A timeout carrying an out-of-range status, as Ollama's errors do when there is no HTTP status."""

    def __init__(self, status_code: int) -> None:
        super().__init__(ERROR_TEXT)
        self.status_code = status_code


class _RaisingStatusError(Exception):
    """An error whose status_code property raises."""

    @property
    def status_code(self) -> int:
        raise RuntimeError("no status available")


class _ResponseError(Exception):
    """A provider error with an HTTP response carrying headers, as OpenAI / Anthropic SDK errors have."""

    def __init__(self, headers: dict[str, str]) -> None:
        super().__init__(ERROR_TEXT)
        self.response = SimpleNamespace(headers=headers)


def _length_finish_reason_error() -> LengthFinishReasonError:
    """OpenAI's error for a completion cut off at the token limit."""
    return LengthFinishReasonError(completion=ChatCompletion.model_construct(usage=None))


def _httpx2_request() -> httpx2.Request:
    """A request as the OpenAI and Anthropic SDKs make them (with httpx2)."""
    return httpx2.Request("POST", API_URL)


def _openai_rate_limit_error(code: str) -> OpenAIRateLimitError:
    """The 429 LangChain raises for OpenAI, carrying OpenAI's error code."""
    response = httpx2.Response(RATE_LIMITED, request=_httpx2_request())
    return OpenAIRateLimitError(ERROR_TEXT, response=response, body={"code": code})


def _http_status_error(status: int) -> httpx.HTTPStatusError:
    """httpx's error for a bad status, which keeps the status on its response rather than on itself."""
    request = httpx.Request("POST", API_URL)
    return httpx.HTTPStatusError(ERROR_TEXT, request=request, response=httpx.Response(status, request=request))


def _retry_state(
    exception: BaseException | None,
    next_sleep: float | None,
    *,
    attempt: int = ATTEMPT_NUMBER,
    stop: stop_base | None = None,
) -> RetryCallState:
    """The state tenacity hands to its wait and before_sleep callbacks after a failed attempt."""
    policy = Retrying(stop=stop if stop is not None else AttemptLimit())
    state = RetryCallState(retry_object=policy, fn=None, args=(), kwargs={})
    state.attempt_number = attempt
    if exception is not None:
        state.set_exception((type(exception), exception, None))
    if next_sleep is not None:
        state.next_action = RetryAction(next_sleep)
    return state


class TestDescribeException:
    @pytest.mark.parametrize(
        ("exception", "expected"),
        [
            (RuntimeError(ERROR_TEXT), f"RuntimeError: {ERROR_TEXT}"),
            (RuntimeError(), "RuntimeError"),
        ],
    )
    def test_uses_the_exception_text(self, exception: BaseException, expected: str) -> None:
        assert describe_exception(exception) == expected

    @pytest.mark.parametrize(
        ("message", "expected_message"),
        [(ERROR_TEXT, ERROR_TEXT), ("", None), (None, None)],
    )
    def test_falls_back_to_a_message_attribute(
        self,
        make_message_attribute_error: Callable[[object], Exception],
        message: object,
        expected_message: str | None,
    ) -> None:
        error = make_message_attribute_error(message)
        name = type(error).__name__

        expected = f"{name}: {expected_message}" if expected_message is not None else name
        assert describe_exception(error) == expected

    def test_converts_a_non_string_message_attribute(
        self, non_string_message_error: Exception, non_string_error_message: dict[str, dict[str, str]]
    ) -> None:
        description = describe_exception(non_string_message_error)

        assert description == f"{type(non_string_message_error).__name__}: {non_string_error_message}"

    def test_unprintable_exception_falls_back_to_its_type_name(self) -> None:
        """describe_exception runs inside tenacity's before_sleep, so a broken __str__ must not raise."""
        error = _UnprintableError()

        assert describe_exception(error) == type(error).__name__

    def test_array_like_message_attribute_is_not_tested_for_truth(
        self, make_message_attribute_error: Callable[[object], Exception]
    ) -> None:
        error = make_message_attribute_error(_AmbiguousTruthValue())

        assert describe_exception(error) == f"{type(error).__name__}: {ARRAY_LIKE_TEXT}"

    def test_unprintable_message_attribute_falls_back_to_the_type_name(
        self, make_message_attribute_error: Callable[[object], Exception]
    ) -> None:
        error = make_message_attribute_error(_UnprintableError())

        assert describe_exception(error) == type(error).__name__


class TestRetryWarning:
    def test_describes_the_exception_verbatim(self, recording_console: Console, markup_like_text: str) -> None:
        error = RuntimeError(markup_like_text)

        recording_console.print(retry_warning(_retry_state(error, SLEEP_SECONDS)))

        assert recording_console.export_text() == (
            f"  Error encountered ({type(error).__name__}: {markup_like_text}). "
            f"Retrying in {SLEEP_SECONDS:.1f}s (attempt {ATTEMPT_NUMBER}/{MAX_RETRY_ATTEMPTS})...\n"
        )

    def test_without_an_exception_or_next_action(self) -> None:
        warning = retry_warning(_retry_state(exception=None, next_sleep=None))

        assert (
            warning.plain == f"  Error encountered. Retrying in 0.0s (attempt {ATTEMPT_NUMBER}/{MAX_RETRY_ATTEMPTS})..."
        )

    def test_progress_shows_the_attempt_total(self) -> None:
        state = _retry_state(TimeoutError(ERROR_TEXT), SLEEP_SECONDS)

        assert retry_warning(state).plain.endswith(f"(attempt {ATTEMPT_NUMBER}/{MAX_RETRY_ATTEMPTS})...")

    def test_progress_counts_unparsed_answers_against_their_own_limit(self) -> None:
        stop = AttemptLimit()
        state = _retry_state(StructuredOutputError(ERROR_TEXT), SLEEP_SECONDS, attempt=1, stop=stop)
        stop(state)  # the policy's stop condition sees the failure first, as tenacity calls it

        assert retry_warning(state).plain.endswith(
            f"(attempt 1/{MAX_RETRY_ATTEMPTS}, unparsed answer 1/{MAX_PARSE_ATTEMPTS})..."
        )

    def test_attempt_total_is_omitted_without_an_attempt_limit(self) -> None:
        state = _retry_state(RuntimeError(ERROR_TEXT), SLEEP_SECONDS, stop=stop_after_delay(SLEEP_SECONDS))

        assert retry_warning(state).plain.endswith(f"(attempt {ATTEMPT_NUMBER})...")


class TestIsRetryableError:
    def test_transient_statuses_are_retryable(
        self, make_status_error: Callable[[int], Exception], retryable_status: int
    ) -> None:
        assert is_retryable_error(make_status_error(retryable_status)) is True

    def test_client_error_statuses_are_not_retryable(
        self, make_status_error: Callable[[int], Exception], non_retryable_status: int
    ) -> None:
        assert is_retryable_error(make_status_error(non_retryable_status)) is False

    def test_a_status_on_the_response_is_read_too(self, retryable_status: int, non_retryable_status: int) -> None:
        """httpx.HTTPStatusError has no status_code of its own; its response has."""
        assert is_retryable_error(_http_status_error(retryable_status)) is True
        assert is_retryable_error(_http_status_error(non_retryable_status)) is False

    @pytest.mark.parametrize(
        ("exception", "expected"),
        [
            (ModelRateLimitError(), True),
            (ModelAPIError(), True),
            (ModelConnectionError(), True),
            (ModelTimeoutError(), True),
            (ModelInvalidRequestError(), False),
            (ModelAuthenticationError(), False),
            (ContextOverflowError(), False),
        ],
        ids=lambda value: type(value).__name__ if isinstance(value, BaseException) else str(value),
    )
    def test_langchain_model_errors_use_their_own_classification(self, exception: Exception, expected: bool) -> None:
        assert is_retryable_error(exception) is expected

    def test_model_error_classification_takes_precedence_over_status(
        self, make_status_error: Callable[[int], Exception], retryable_status: int
    ) -> None:
        """LangChain's subclasses also inherit the SDK error, status included; the classification wins."""
        error = ModelInvalidRequestError()
        error.status_code = retryable_status  # type: ignore[attr-defined]

        assert is_retryable_error(error) is False

    @pytest.mark.parametrize(
        "make_error",
        [
            lambda: OpenAIConnectionError(message=ERROR_TEXT, request=_httpx2_request()),
            lambda: OpenAITimeoutError(request=_httpx2_request()),
            lambda: AnthropicConnectionError(message=ERROR_TEXT, request=_httpx2_request()),
            lambda: AnthropicTimeoutError(request=_httpx2_request()),
        ],
        ids=["openai-connection", "openai-timeout", "anthropic-connection", "anthropic-timeout"],
    )
    def test_sdk_transport_failures_arrive_as_retryable_model_errors(self, make_error: Callable[[], Exception]) -> None:
        """The SDKs' httpx2 failures reach the policy wrapped by the SDK and then LangChain, never unwrapped."""
        assert is_retryable_error(make_error()) is True

    @pytest.mark.parametrize(
        "exception",
        [
            ConnectionError(),
            TimeoutError(),
            httpx.ConnectError(ERROR_TEXT, request=httpx.Request("POST", API_URL)),
            httpx.ReadTimeout(ERROR_TEXT, request=httpx.Request("POST", API_URL)),
        ],
        ids=lambda value: type(value).__name__,
    )
    def test_status_less_connection_failures_and_timeouts_are_retryable(self, exception: Exception) -> None:
        """Ollama's client uses httpx and lets these through."""
        assert is_retryable_error(exception) is True

    def test_answers_that_did_not_parse_are_retryable(self) -> None:
        assert is_retryable_error(StructuredOutputError(ERROR_TEXT)) is True

    def test_parsing_errors_raised_outside_the_structured_output_boundary_are_not(
        self, parsing_failure: Exception
    ) -> None:
        """Retrying is decided by pdfalive's own type, not by any ValidationError, which may be a bug of ours."""
        assert is_retryable_error(parsing_failure) is False

    def test_a_used_up_quota_is_not_retried(self) -> None:
        """Waiting does not refill a quota, though OpenAI reports it as a 429 like a rate limit."""
        assert is_retryable_error(_openai_rate_limit_error(INSUFFICIENT_QUOTA_CODE)) is False
        assert is_retryable_error(_openai_rate_limit_error(RATE_LIMIT_CODE)) is True

    @pytest.mark.parametrize(
        "exception",
        [
            TypeError(ERROR_TEXT),
            KeyError(ERROR_TEXT),
            RuntimeError(ERROR_TEXT),
            _length_finish_reason_error(),
            CassetteMissError(ERROR_TEXT),
        ],
        ids=lambda value: type(value).__name__,
    )
    def test_other_errors_are_not_retryable(self, exception: Exception) -> None:
        """An allowlist: programming errors, truncated completions and replay mismatches surface immediately."""
        assert is_retryable_error(exception) is False

    def test_an_out_of_range_status_counts_as_no_status(self, make_status_error: Callable[[int], Exception]) -> None:
        """Ollama reports -1 when it has no HTTP status; that must not read as a client error."""
        sentinel_error = make_status_error(OLLAMA_NO_STATUS)

        assert is_retryable_error(sentinel_error) is False  # status-less and not on the allowlist
        assert is_retryable_error(_SentinelStatusTimeout(OLLAMA_NO_STATUS)) is True

    @pytest.mark.parametrize(
        ("status_code", "expected"),
        [(OLLAMA_NO_STATUS, True), (NON_RETRYABLE_STATUS, False), (SERVER_ERROR_STATUS, True)],
        ids=["mid-stream-sentinel", "client-error", "server-error"],
    )
    def test_ollama_response_errors(self, status_code: int, expected: bool) -> None:
        """Ollama raises ResponseError with its -1 sentinel for a server error in the middle of a streamed answer."""
        assert is_retryable_error(ollama.ResponseError(ERROR_TEXT, status_code=status_code)) is expected

    def test_a_raising_status_property_counts_as_no_status(self) -> None:
        assert is_retryable_error(_RaisingStatusError(ERROR_TEXT)) is False

    @pytest.mark.parametrize("exception", [KeyboardInterrupt(), SystemExit(), GeneratorExit()])
    def test_exceptions_that_stop_the_program_are_not_retryable(self, exception: BaseException) -> None:
        """Regression: Ctrl-C during an LLM call was retried like a transient error."""
        assert is_retryable_error(exception) is False


class TestRetryWaitSeconds:
    def test_backoff_doubles_from_the_initial_wait(self) -> None:
        waits = [retry_wait_seconds(_retry_state(None, None, attempt=attempt)) for attempt in BACKOFF_ATTEMPTS]

        assert waits == [RETRY_INITIAL_WAIT_SECONDS * 2 ** (attempt - 1) for attempt in BACKOFF_ATTEMPTS]
        assert max(waits) <= RETRY_MAX_WAIT_SECONDS

    @pytest.mark.parametrize(
        ("headers", "expected"),
        [
            ({"retry-after": str(RETRY_AFTER_SECONDS)}, RETRY_AFTER_SECONDS),
            ({"retry-after-ms": str(RETRY_AFTER_MILLISECONDS)}, RETRY_AFTER_SECONDS),
            ({"retry-after-ms": UNPARSABLE_HINT, "retry-after": str(RETRY_AFTER_SECONDS)}, RETRY_AFTER_SECONDS),
            ({"retry-after": str(RETRY_MAX_WAIT_SECONDS * 10)}, RETRY_MAX_WAIT_SECONDS),
            ({"retry-after": str(SHORT_RETRY_AFTER_SECONDS)}, RETRY_INITIAL_WAIT_SECONDS),
            ({"retry-after": UNPARSABLE_HINT}, RETRY_INITIAL_WAIT_SECONDS),
            ({}, RETRY_INITIAL_WAIT_SECONDS),
        ],
        ids=[
            "seconds",
            "milliseconds",
            "malformed-milliseconds-then-seconds",
            "capped",
            "shorter-than-backoff",
            "unparsable",
            "no-hint",
        ],
    )
    def test_a_provider_retry_after_hint_lengthens_the_wait_up_to_the_cap(
        self, headers: dict[str, str], expected: float
    ) -> None:
        state = _retry_state(_ResponseError(headers), None, attempt=1)

        assert retry_wait_seconds(state) == expected

    def test_an_http_date_hint_is_honored(self) -> None:
        when = datetime.now(UTC) + timedelta(seconds=RETRY_AFTER_SECONDS * 2)
        state = _retry_state(_ResponseError({"retry-after": format_datetime(when, usegmt=True)}), None, attempt=1)

        assert RETRY_INITIAL_WAIT_SECONDS < retry_wait_seconds(state) <= RETRY_AFTER_SECONDS * 2


class TestLlmRetry:
    """The retry decorator, exercised with real tenacity and instant backoff waits."""

    def _retrying(self, console: Console, outcomes: list[object]) -> tuple[Callable[[], object], MagicMock]:
        """A call raising or returning each outcome in turn, wrapped in llm_retry, and the mock behind it."""
        call = MagicMock(side_effect=outcomes)

        @llm_retry(console)
        def invoke() -> object:
            return call()

        return invoke, call

    def test_transient_errors_are_retried_and_reported(
        self,
        recording_console: Console,
        instant_retries: None,
        make_status_error: Callable[[int], Exception],
        retryable_status: int,
    ) -> None:
        error = make_status_error(retryable_status)
        invoke, call = self._retrying(recording_console, [error, RESULT])

        assert invoke() == RESULT
        assert call.call_count == 2
        assert f"Error encountered ({describe_exception(error)})" in recording_console.export_text()

    def test_answers_that_did_not_parse_are_retried(self, recording_console: Console, instant_retries: None) -> None:
        invoke, call = self._retrying(recording_console, [StructuredOutputError(ERROR_TEXT), RESULT])

        assert invoke() == RESULT
        assert call.call_count == 2

    def test_answers_that_keep_failing_to_parse_get_fewer_attempts(
        self, recording_console: Console, instant_retries: None
    ) -> None:
        invoke, call = self._retrying(recording_console, [StructuredOutputError(ERROR_TEXT)] * MAX_RETRY_ATTEMPTS)

        with pytest.raises(StructuredOutputError):
            invoke()
        assert call.call_count == MAX_PARSE_ATTEMPTS

    def test_a_parse_failure_after_transient_ones_still_gets_its_retries(
        self,
        recording_console: Console,
        instant_retries: None,
        make_status_error: Callable[[int], Exception],
        retryable_status: int,
    ) -> None:
        """Regression: the limit followed the last failure only, so a first parse failure at attempt 3 ended a call."""
        transient = [make_status_error(retryable_status)] * (MAX_PARSE_ATTEMPTS - 1)
        invoke, call = self._retrying(recording_console, [*transient, StructuredOutputError(ERROR_TEXT), RESULT])

        assert invoke() == RESULT
        assert call.call_count == len(transient) + 2

    def test_parse_failures_are_bounded_across_interleaved_transient_ones(
        self,
        recording_console: Console,
        instant_retries: None,
        make_status_error: Callable[[int], Exception],
        retryable_status: int,
    ) -> None:
        unparsed = StructuredOutputError(ERROR_TEXT)
        outcomes = [unparsed, make_status_error(retryable_status)] * MAX_PARSE_ATTEMPTS
        invoke, call = self._retrying(recording_console, [*outcomes, RESULT])

        with pytest.raises(StructuredOutputError):
            invoke()
        assert sum(outcome is unparsed for outcome in outcomes[: call.call_count]) == MAX_PARSE_ATTEMPTS

    def test_the_parse_count_restarts_with_each_call(self, recording_console: Console, instant_retries: None) -> None:
        """One decorated function called twice: the second call does not inherit the first one's failures."""
        unparsed = [StructuredOutputError(ERROR_TEXT)] * (MAX_PARSE_ATTEMPTS - 1)
        invoke, _call = self._retrying(recording_console, [*unparsed, RESULT, *unparsed, RESULT])

        assert invoke() == RESULT
        assert invoke() == RESULT

    def test_client_errors_surface_immediately(
        self,
        recording_console: Console,
        instant_retries: None,
        make_status_error: Callable[[int], Exception],
        non_retryable_status: int,
    ) -> None:
        error = make_status_error(non_retryable_status)
        invoke, call = self._retrying(recording_console, [error, RESULT])

        with pytest.raises(type(error)):
            invoke()
        assert call.call_count == 1
        assert recording_console.export_text() == ""

    @pytest.mark.parametrize("exception_type", [KeyboardInterrupt, SystemExit])
    def test_stopping_the_program_is_not_retried(
        self, recording_console: Console, instant_retries: None, exception_type: type[BaseException]
    ) -> None:
        """Regression: Ctrl-C during an LLM call was swallowed and the call retried."""
        invoke, call = self._retrying(recording_console, [exception_type(), RESULT])

        with pytest.raises(exception_type):
            invoke()
        assert call.call_count == 1

    def test_gives_up_after_the_last_attempt(
        self,
        recording_console: Console,
        instant_retries: None,
        make_status_error: Callable[[int], Exception],
        retryable_status: int,
    ) -> None:
        error = make_status_error(retryable_status)
        invoke, call = self._retrying(recording_console, [error] * MAX_RETRY_ATTEMPTS)

        with pytest.raises(type(error)):
            invoke()
        assert call.call_count == MAX_RETRY_ATTEMPTS
        warnings = [line for line in recording_console.export_text().splitlines() if "Error encountered" in line]
        assert len(warnings) == MAX_RETRY_ATTEMPTS - 1


class TestInvokeWithRetry:
    DESCRIPTION = "batch 1/1"
    INPUT_TOKENS = 1234

    def test_reports_the_call_and_its_completion(self, recording_console: Console) -> None:
        result = invoke_with_retry(
            recording_console, lambda: RESULT, description=self.DESCRIPTION, input_tokens=self.INPUT_TOKENS
        )

        assert result == RESULT
        lines = recording_console.export_text().splitlines()
        assert lines[0] == f"  Invoking LLM for {self.DESCRIPTION} (~{self.INPUT_TOKENS:,} input tokens)..."
        assert lines[1].startswith(f"  Completed {self.DESCRIPTION} in ")

    def test_a_call_that_fails_for_good_raises_an_llm_call_error(
        self, recording_console: Console, make_status_error: Callable[[int], Exception], non_retryable_status: int
    ) -> None:
        """One type for every failure the policy gives up on, so a command reports it in one place."""
        error = make_status_error(non_retryable_status)

        def fail() -> object:
            raise error

        with pytest.raises(LLMCallError, match=self.DESCRIPTION) as raised:
            invoke_with_retry(recording_console, fail, description=self.DESCRIPTION, input_tokens=self.INPUT_TOKENS)

        assert raised.value.__cause__ is error

    @pytest.mark.parametrize("exception_type", [KeyboardInterrupt, SystemExit])
    def test_stopping_the_program_is_not_wrapped(
        self, recording_console: Console, exception_type: type[BaseException]
    ) -> None:
        def stop() -> object:
            raise exception_type()

        with pytest.raises(exception_type):
            invoke_with_retry(recording_console, stop, description=self.DESCRIPTION, input_tokens=self.INPUT_TOKENS)
