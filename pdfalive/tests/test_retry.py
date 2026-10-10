"""Tests for retry reporting."""

from collections.abc import Callable

import pytest
from rich.console import Console
from tenacity import RetryAction, RetryCallState, Retrying

from pdfalive.retry import describe_exception, retry_warning


ERROR_TEXT = "rate limited"
SLEEP_SECONDS = 12.5
ATTEMPT_NUMBER = 2
MAX_ATTEMPTS = 5
ARRAY_LIKE_TEXT = "[429 503]"


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


def _retry_state(exception: BaseException | None, next_sleep: float | None) -> RetryCallState:
    """The state tenacity hands to before_sleep after a failed attempt."""
    state = RetryCallState(retry_object=Retrying(), fn=None, args=(), kwargs={})
    state.attempt_number = ATTEMPT_NUMBER
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

        recording_console.print(retry_warning(_retry_state(error, SLEEP_SECONDS), MAX_ATTEMPTS))

        assert recording_console.export_text() == (
            f"  Error encountered ({type(error).__name__}: {markup_like_text}). "
            f"Retrying in {SLEEP_SECONDS:.1f}s (attempt {ATTEMPT_NUMBER}/{MAX_ATTEMPTS})...\n"
        )

    def test_without_an_exception_or_next_action(self) -> None:
        warning = retry_warning(_retry_state(exception=None, next_sleep=None), MAX_ATTEMPTS)

        assert warning.plain == f"  Error encountered. Retrying in 0.0s (attempt {ATTEMPT_NUMBER}/{MAX_ATTEMPTS})..."
