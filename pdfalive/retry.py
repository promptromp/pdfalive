"""Retry reporting shared by the processors that call LLMs."""

from rich.text import Text
from tenacity import RetryCallState

from pdfalive.console import safe_text


def _exception_message(exc: BaseException) -> str:
    """The exception's text: ``str(exc)``, else its ``message`` attribute, else ``""``; never raises."""
    try:
        message = str(exc)
        if not message:
            # Some SDK errors stringify to "" and keep their text in a `message`
            # attribute of any type. Test it against None only: the truth value
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


def retry_warning(retry_state: RetryCallState, max_attempts: int) -> Text:
    """The console warning shown before tenacity sleeps ahead of the next attempt.

    Args:
        retry_state: The state tenacity passes to its ``before_sleep`` callback.
        max_attempts: The retry policy's attempt limit, for the "attempt n/N" counter.

    Returns:
        The warning, with the exception description shown verbatim.
    """
    wait_seconds = retry_state.next_action.sleep if retry_state.next_action is not None else 0.0
    exception = retry_state.outcome.exception() if retry_state.outcome is not None else None
    details = f" ({describe_exception(exception)})" if exception is not None else ""
    message = (
        f"Error encountered{details}. Retrying in {wait_seconds:.1f}s "
        f"(attempt {retry_state.attempt_number}/{max_attempts})..."
    )
    return safe_text("  ", (message, "yellow"), multiline=True)
