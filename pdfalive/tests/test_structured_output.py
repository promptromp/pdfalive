"""Tests for the structured-output boundary: parsing, refusals, truncation, and per-attempt token usage."""

import subprocess
import sys
from collections.abc import Callable
from typing import Any
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage
from langchain_openai.chat_models.base import OpenAIRefusalError
from pydantic import BaseModel, ValidationError
from rich.console import Console

from pdfalive.retry import MAX_PARSE_ATTEMPTS, LLMCallError, StructuredOutputError, is_retryable_error
from pdfalive.structured_output import (
    ModelRefusalError,
    TruncatedAnswerError,
    invoke_structured,
    invoke_structured_with_retry,
)
from pdfalive.tokens import TokenUsage


class _Schema(BaseModel):
    """A structured answer to ask for."""

    text: str


class _OtherSchema(BaseModel):
    """A structured answer of the wrong kind."""

    number: int


PARSED = _Schema(text="the answer")
MESSAGES: list[Any] = []
DESCRIPTION = "batch 1/1"
ESTIMATED_INPUT_TOKENS = 321
REPORTED_USAGE = {"input_tokens": 1234, "output_tokens": 56, "total_tokens": 1290}
REFUSAL_TEXT = "I can't help with that."
REFUSAL_STOP_REASON = {"stop_reason": "refusal"}
TRUNCATED_STOP_REASON = {"stop_reason": "max_tokens"}
UNDECODABLE_TEXT = "caf\udce9.pdf"
# Where OpenAI's refusal error lives; pdfalive looks it up only once something has imported it.
OPENAI_CHAT_MODULE = "langchain_openai.chat_models.base"


def _raw(**fields: Any) -> AIMessage:
    """A raw provider message, reporting REPORTED_USAGE unless told otherwise."""
    return AIMessage(content="", **{"usage_metadata": REPORTED_USAGE, **fields})


def _model(*responses: Any) -> MagicMock:
    """What with_structured_output() returns, answering (or raising) each response in turn."""
    model = MagicMock()
    model.invoke.side_effect = list(responses)
    return model


def _schema_violation() -> ValidationError:
    """The error the OpenAI SDK raises from the call itself for an answer that breaks the schema."""
    try:
        _OtherSchema.model_validate({"number": "not a number"})
    except ValidationError as error:
        return error
    raise AssertionError("the payload above must not validate")


@pytest.fixture
def usage() -> TokenUsage:
    return TokenUsage()


@pytest.fixture
def answer(make_structured_answer: Callable[..., dict[str, Any]]) -> Callable[..., dict[str, Any]]:
    """An answer with a raw message reporting REPORTED_USAGE, unless another raw message is given."""

    def _answer(parsed: Any = None, *, parsing_error: BaseException | None = None, **raw_fields: Any) -> dict[str, Any]:
        return make_structured_answer(parsed, parsing_error=parsing_error, raw=_raw(**raw_fields))

    return _answer


def _invoke(model: MagicMock, usage: TokenUsage) -> _Schema:
    return invoke_structured(
        model, MESSAGES, _Schema, usage=usage, description=DESCRIPTION, estimated_input_tokens=ESTIMATED_INPUT_TOKENS
    )


class TestInvokeStructured:
    def test_returns_the_parsed_answer_and_records_the_reported_usage(self, answer, usage: TokenUsage) -> None:
        assert _invoke(_model(answer(PARSED)), usage) == PARSED
        assert (usage.llm_calls, usage.input_tokens, usage.output_tokens) == (
            1,
            REPORTED_USAGE["input_tokens"],
            REPORTED_USAGE["output_tokens"],
        )

    def test_without_reported_usage_records_estimates(self, make_structured_answer, usage: TokenUsage) -> None:
        assert _invoke(_model(make_structured_answer(PARSED, raw=AIMessage(content=""))), usage) == PARSED
        assert usage.input_tokens == ESTIMATED_INPUT_TOKENS
        assert usage.output_tokens > 0

    def test_an_answer_holding_a_lone_surrogate_is_measured(self, make_structured_answer, usage: TokenUsage) -> None:
        """An undecodable byte in a file name reaches the answer as a lone surrogate, which JSON cannot encode."""
        parsed = _Schema(text=UNDECODABLE_TEXT)

        assert _invoke(_model(make_structured_answer(parsed)), usage) == parsed
        assert usage.output_tokens > 0

    def test_a_response_that_is_not_an_include_raw_answer_is_a_programming_error(self, usage: TokenUsage) -> None:
        """Every processor asks for include_raw=True; a bare model means the model was asked wrongly."""
        with pytest.raises(TypeError, match="include_raw"):
            _invoke(_model(PARSED), usage)

    def test_a_recorded_parsing_error_becomes_a_structured_output_error(
        self, answer, usage: TokenUsage, parsing_failure: Exception
    ) -> None:
        with pytest.raises(StructuredOutputError) as raised:
            _invoke(_model(answer(parsing_error=parsing_failure)), usage)

        assert raised.value.__cause__ is parsing_failure
        assert usage.llm_calls == 1  # billed, though it did not parse

    def test_a_schema_violation_raised_by_the_call_becomes_a_structured_output_error(self, usage: TokenUsage) -> None:
        """Regression: the OpenAI SDK validates its own structured output and raises ValidationError from the call
        (a confidence of 1.5, say), which was neither retried nor counted.
        """
        violation = _schema_violation()

        with pytest.raises(StructuredOutputError) as raised:
            _invoke(_model(violation), usage)

        assert raised.value.__cause__ is violation
        assert (usage.llm_calls, usage.input_tokens) == (1, ESTIMATED_INPUT_TOKENS)

    @pytest.mark.parametrize("parsed", [None, _OtherSchema(number=1)], ids=["no-parsed-result", "wrong-schema"])
    def test_an_answer_without_a_parsed_schema_is_a_structured_output_error(
        self, answer, usage: TokenUsage, parsed: Any
    ) -> None:
        """Regression: rename crashed with AttributeError on a None answer instead of retrying."""
        with pytest.raises(StructuredOutputError):
            _invoke(_model(answer(parsed)), usage)

    @pytest.mark.parametrize(
        ("parsing_error", "raw_fields"),
        [(OpenAIRefusalError(REFUSAL_TEXT), {}), (None, {"response_metadata": REFUSAL_STOP_REASON})],
        ids=["openai-refusal", "anthropic-refusal-stop-reason"],
    )
    def test_a_refusal_is_a_model_refusal_error(
        self, answer, usage: TokenUsage, parsing_error: BaseException | None, raw_fields: dict[str, Any]
    ) -> None:
        with pytest.raises(ModelRefusalError) as raised:
            _invoke(_model(answer(parsing_error=parsing_error, **raw_fields)), usage)

        assert is_retryable_error(raised.value) is False

    def test_an_answer_cut_off_at_the_token_limit_is_a_truncated_answer_error(self, answer, usage: TokenUsage) -> None:
        """Like OpenAI's LengthFinishReasonError, an Anthropic max_tokens stop is not retried: it would recur."""
        with pytest.raises(TruncatedAnswerError) as raised:
            _invoke(_model(answer(response_metadata=TRUNCATED_STOP_REASON)), usage)

        assert is_retryable_error(raised.value) is False

    def test_the_openai_refusal_is_recognized_only_once_its_module_is_loaded(
        self, answer, usage: TokenUsage, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No OpenAI refusal can exist before something imports the OpenAI integration."""
        monkeypatch.delitem(sys.modules, OPENAI_CHAT_MODULE)

        with pytest.raises(StructuredOutputError):
            _invoke(_model(answer(parsing_error=OpenAIRefusalError(REFUSAL_TEXT))), usage)

    def test_importing_the_boundary_does_not_import_the_openai_integration(self) -> None:
        """Regression: the module imported langchain_openai at startup just for its refusal error type."""
        probe = f"import sys, pdfalive.structured_output; print({OPENAI_CHAT_MODULE!r} in sys.modules)"

        completed = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True)

        assert completed.stdout.strip() == "False"

    def test_errors_raised_by_the_call_pass_through(self, usage: TokenUsage) -> None:
        error = TimeoutError(DESCRIPTION)

        with pytest.raises(TimeoutError) as raised:
            _invoke(_model(error), usage)

        assert raised.value is error
        assert usage.llm_calls == 0  # no answer arrived, so nothing was billed


class TestInvokeStructuredWithRetry:
    def _invoke(self, console: Console, model: MagicMock, usage: TokenUsage) -> _Schema:
        return invoke_structured_with_retry(
            console, model, MESSAGES, _Schema, usage=usage, description=DESCRIPTION, input_tokens=ESTIMATED_INPUT_TOKENS
        )

    def test_every_billed_attempt_is_recorded(
        self, recording_console: Console, instant_retries: None, answer, usage: TokenUsage, parsing_failure: Exception
    ) -> None:
        """Regression: an attempt whose answer failed to parse was billed but missing from the token usage."""
        model = _model(answer(parsing_error=parsing_failure), _schema_violation(), answer(PARSED))

        assert self._invoke(recording_console, model, usage) == PARSED
        assert usage.llm_calls == model.invoke.call_count

    def test_answers_that_keep_failing_to_parse_stop_at_the_parse_limit(
        self, recording_console: Console, instant_retries: None, answer, usage: TokenUsage
    ) -> None:
        model = _model(*[answer()] * (MAX_PARSE_ATTEMPTS + 1))

        with pytest.raises(LLMCallError) as raised:
            self._invoke(recording_console, model, usage)

        assert isinstance(raised.value.__cause__, StructuredOutputError)
        assert model.invoke.call_count == MAX_PARSE_ATTEMPTS
        assert usage.llm_calls == MAX_PARSE_ATTEMPTS

    @pytest.mark.parametrize(
        ("parsing_error", "raw_fields", "error_type"),
        [
            (OpenAIRefusalError(REFUSAL_TEXT), {}, ModelRefusalError),
            (None, {"response_metadata": TRUNCATED_STOP_REASON}, TruncatedAnswerError),
        ],
        ids=["refusal", "truncation"],
    )
    def test_refusals_and_truncations_fail_on_the_first_attempt(
        self,
        recording_console: Console,
        instant_retries: None,
        answer,
        usage: TokenUsage,
        parsing_error: BaseException | None,
        raw_fields: dict[str, Any],
        error_type: type[Exception],
    ) -> None:
        model = _model(answer(parsing_error=parsing_error, **raw_fields), answer(PARSED))

        with pytest.raises(LLMCallError) as raised:
            self._invoke(recording_console, model, usage)

        assert isinstance(raised.value.__cause__, error_type)
        assert model.invoke.call_count == 1
