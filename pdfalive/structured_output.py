"""Where a structured-output LLM answer becomes the parsed model, shared by every processor that asks for one.

Processors ask for structured output with ``include_raw=True``, so an answer
arrives as ``{"raw": AIMessage, "parsed": model or None, "parsing_error":
exception or None}``. ``invoke_structured()`` makes one call and returns the
parsed model, or raises:

- ``StructuredOutputError`` (pdfalive.retry) when the answer did not parse or
  held no parsed result, including a pydantic ``ValidationError`` raised by the
  call itself (the OpenAI SDK validates its own structured outputs). The next
  answer may parse, so the retry policy retries it, up to ``MAX_PARSE_ATTEMPTS``
  times in a call.
- ``ModelRefusalError`` when the model declined to answer: OpenAI's
  structured-output refusal, or an Anthropic ``refusal`` stop reason.
- ``TruncatedAnswerError`` when the answer was cut off at the output token limit
  (an Anthropic ``max_tokens`` stop reason; OpenAI's SDK raises its own
  ``LengthFinishReasonError``).

Asking again would get the same refusal or the same truncation, so neither is
retried. Any other error the call raises passes through, to be classified by
the policy.

Every answer is billed whether or not it parses, so each one's token usage is
recorded as it arrives: the provider's own counts from the raw message's
``usage_metadata``, or tiktoken estimates where it has none (a cassette
recorded without usage, a provider that does not report it, an answer the SDK
rejected before handing it over).
"""

import sys
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from pydantic import BaseModel, ValidationError

from pdfalive.console import ConsoleType
from pdfalive.retry import StructuredOutputError, describe_exception, invoke_with_retry
from pdfalive.tokens import TokenUsage, estimate_tokens


class ModelRefusalError(Exception):
    """The model declined to answer the request; not retried, since asking again gets the same answer."""


class TruncatedAnswerError(Exception):
    """The answer was cut off at the output token limit; not retried, since it would be cut off again."""


# The keys of an include_raw=True structured-output answer.
_RAW_KEY = "raw"
_PARSED_KEY = "parsed"
_PARSING_ERROR_KEY = "parsing_error"
# The keys of a raw message's usage_metadata.
_INPUT_TOKENS_KEY = "input_tokens"
_OUTPUT_TOKENS_KEY = "output_tokens"
# The module and type of OpenAI's structured-output refusal, looked up only once something
# has imported them (so pdfalive need not import the OpenAI integration to recognize it).
_OPENAI_CHAT_MODULE = "langchain_openai.chat_models.base"
_OPENAI_REFUSAL_ERROR = "OpenAIRefusalError"
# The response_metadata key and values of an answer's stop reason (Anthropic).
_STOP_REASON_KEY = "stop_reason"
_REFUSAL_STOP_REASON = "refusal"
_TRUNCATED_STOP_REASON = "max_tokens"
# The output tokens recorded for an answer the SDK rejected before handing it over: its size is unknown.
_UNKNOWN_OUTPUT_TOKENS = 0


class StructuredModel(Protocol):
    """What ``with_structured_output(..., include_raw=True)`` returns: something to invoke with the messages."""

    def invoke(self, messages: Any, /) -> Any: ...


def _is_openai_refusal(error: BaseException | None) -> bool:
    module = sys.modules.get(_OPENAI_CHAT_MODULE)
    refusal_error = getattr(module, _OPENAI_REFUSAL_ERROR, None)
    return isinstance(refusal_error, type) and isinstance(error, refusal_error)


@dataclass(frozen=True)
class _Answer:
    """A structured-output answer: the raw message (if any), the parsed model (if any), and the parsing error."""

    raw: Any
    parsed: Any
    parsing_error: BaseException | None

    @classmethod
    def from_response(cls, response: Any) -> "_Answer":
        """The answer an include_raw=True response holds.

        Raises:
            TypeError: The response is not an include_raw=True answer, so the model was asked wrongly.
        """
        if not isinstance(response, Mapping) or _PARSED_KEY not in response:
            raise TypeError(f"Expected an include_raw=True structured-output answer, got {type(response).__name__}")
        return cls(response.get(_RAW_KEY), response[_PARSED_KEY], response.get(_PARSING_ERROR_KEY))

    def usage_metadata(self) -> Mapping[str, Any]:
        """The provider's reported token usage, or an empty mapping."""
        metadata = getattr(self.raw, "usage_metadata", None)
        return metadata if isinstance(metadata, Mapping) else {}

    def _stop_reason(self) -> Any:
        metadata = getattr(self.raw, "response_metadata", None)
        return metadata.get(_STOP_REASON_KEY) if isinstance(metadata, Mapping) else None

    def is_refusal(self) -> bool:
        return _is_openai_refusal(self.parsing_error) or self._stop_reason() == _REFUSAL_STOP_REASON

    def is_truncated(self) -> bool:
        return self._stop_reason() == _TRUNCATED_STOP_REASON

    def estimated_output_tokens(self) -> int:
        """Output tokens estimated from the parsed model, or from the raw message's text when nothing parsed.

        The parsed model is measured by its repr, not its JSON: an answer can hold lone surrogates
        (a file name's undecodable bytes), which JSON serialization rejects.
        """
        if isinstance(self.parsed, BaseModel):
            return estimate_tokens(repr(self.parsed.model_dump()))
        return estimate_tokens(str(getattr(self.raw, "content", "")))


def invoke_structured[T: BaseModel](
    model: StructuredModel,
    messages: list[Any],
    schema: type[T],
    *,
    usage: TokenUsage,
    description: str,
    estimated_input_tokens: int,
) -> T:
    """Make one structured-output call and return its parsed answer, recording the call's token usage in usage.

    Args:
        model: What ``with_structured_output(schema, include_raw=True)`` returned.
        messages: The messages to send.
        schema: The model the answer must parse into.
        usage: Where the call's token usage is recorded, whether or not the answer parses.
        description: What the call is for, recorded with its usage.
        estimated_input_tokens: The input tokens to record when the provider reports none.

    Raises:
        StructuredOutputError: The answer did not parse into schema, or held no parsed result.
        ModelRefusalError: The model declined to answer.
        TruncatedAnswerError: The answer was cut off at the output token limit.
    """
    try:
        response = model.invoke(messages)
    except ValidationError as error:
        # The SDK validated the answer itself and rejected it: billed, but its size is unknown.
        usage.add_call(
            input_tokens=estimated_input_tokens, output_tokens=_UNKNOWN_OUTPUT_TOKENS, description=description
        )
        raise StructuredOutputError(
            f"The answer did not parse as {schema.__name__}: {describe_exception(error)}"
        ) from error

    answer = _Answer.from_response(response)
    reported = answer.usage_metadata()
    usage.add_call(
        input_tokens=reported.get(_INPUT_TOKENS_KEY) or estimated_input_tokens,
        output_tokens=reported.get(_OUTPUT_TOKENS_KEY) or answer.estimated_output_tokens(),
        description=description,
    )
    if answer.parsing_error is None and isinstance(answer.parsed, schema):
        return answer.parsed
    if answer.is_refusal():
        details = f": {describe_exception(answer.parsing_error)}" if answer.parsing_error is not None else ""
        raise ModelRefusalError(f"The model declined to answer{details}") from answer.parsing_error
    if answer.is_truncated():
        raise TruncatedAnswerError(
            f"The answer was cut off at the output token limit before it held a {schema.__name__}"
        ) from answer.parsing_error
    if answer.parsing_error is not None:
        raise StructuredOutputError(
            f"The answer did not parse as {schema.__name__}: {describe_exception(answer.parsing_error)}"
        ) from answer.parsing_error
    raise StructuredOutputError(f"The answer held no {schema.__name__}")


def invoke_structured_with_retry[T: BaseModel](
    console: ConsoleType,
    model: StructuredModel,
    messages: list[Any],
    schema: type[T],
    *,
    usage: TokenUsage,
    description: str,
    input_tokens: int,
) -> T:
    """invoke_structured() under the retry policy (pdfalive.retry), reported on console.

    Each attempt's token usage is recorded in usage as the attempt completes, so a
    retried call shows up once per billed attempt.

    Raises:
        LLMCallError: The call failed for good (see pdfalive.retry.invoke_with_retry).
    """
    return invoke_with_retry(
        console,
        lambda: invoke_structured(
            model, messages, schema, usage=usage, description=description, estimated_input_tokens=input_tokens
        ),
        description=description,
        input_tokens=input_tokens,
    )
