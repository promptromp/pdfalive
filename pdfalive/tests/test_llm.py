"""Tests for chat model construction."""

from collections.abc import Callable, Iterator
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from pdfalive.llm import build_chat_model


# pdfalive.retry is the only retry layer for OpenAI and Anthropic, so their SDKs get no retries of their own.
NO_SDK_RETRIES = {"max_retries": 0}

# Model identifiers that resolve to each provider, and the API key variable each reads.
OPENAI_MODEL = "gpt-5.6"
ANTHROPIC_MODEL = "claude-sonnet-5"
OLLAMA_MODEL = "ollama:llama3"
API_KEY_VARIABLES = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY")
DUMMY_API_KEY = "test-key"


@pytest.fixture
def mock_init_chat_model() -> Iterator[MagicMock]:
    """Patch the LangChain initializer and expose the recorded call."""
    with patch("pdfalive.llm.init_chat_model") as mock:
        yield mock


@pytest.fixture
def dummy_api_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """Let provider clients be constructed without real credentials (nothing is sent)."""
    for variable in API_KEY_VARIABLES:
        monkeypatch.setenv(variable, DUMMY_API_KEY)


def _call_kwargs(mock: MagicMock) -> dict[str, Any]:
    return dict(mock.call_args.kwargs)


class TestBuildChatModel:
    def test_omits_reasoning_effort_when_unset(self, mock_init_chat_model: MagicMock) -> None:
        """An unset effort must not reach the provider at all, not even as None.

        Providers treat an explicit None as a meaningful value, so passing it
        would change behavior for users who never opted in.
        """
        build_chat_model(OPENAI_MODEL)

        assert "reasoning_effort" not in _call_kwargs(mock_init_chat_model)

    @pytest.mark.parametrize("effort", ["none", "low", "medium", "high", "xhigh", "max"])
    def test_passes_effort_through_unvalidated(self, mock_init_chat_model: MagicMock, effort: str) -> None:
        """Accepted levels differ per model, so the provider validates, not us."""
        build_chat_model(OPENAI_MODEL, effort)

        assert _call_kwargs(mock_init_chat_model) == {
            "model": OPENAI_MODEL,
            "reasoning_effort": effort,
            **NO_SDK_RETRIES,
        }

    def test_returns_the_initialized_model(self, mock_init_chat_model: MagicMock) -> None:
        assert build_chat_model(OPENAI_MODEL) is mock_init_chat_model.return_value


class TestSdkRetries:
    """OpenAI's and Anthropic's own retries are switched off; other providers get no extra setting."""

    @pytest.mark.parametrize(
        "model_identifier",
        [OPENAI_MODEL, ANTHROPIC_MODEL, "openai:gpt-6-luna", "anthropic:claude-opus-5", "o3-mini", "CLAUDE-HAIKU-5"],
    )
    def test_openai_and_anthropic_models_get_no_sdk_retries(
        self, mock_init_chat_model: MagicMock, model_identifier: str
    ) -> None:
        build_chat_model(model_identifier)

        assert _call_kwargs(mock_init_chat_model) == {"model": model_identifier, **NO_SDK_RETRIES}

    @pytest.mark.parametrize("model_identifier", [OLLAMA_MODEL, "mistral-large-latest", "groq:llama-3.3-70b"])
    def test_other_providers_get_no_extra_setting(self, mock_init_chat_model: MagicMock, model_identifier: str) -> None:
        """They keep their SDK's own behavior, retries included: no setting is assumed to exist."""
        build_chat_model(model_identifier)

        assert _call_kwargs(mock_init_chat_model) == {"model": model_identifier}

    @pytest.mark.parametrize(
        ("model_identifier", "sdk_client"),
        [(OPENAI_MODEL, lambda model: model.root_client), (ANTHROPIC_MODEL, lambda model: model._client)],
        ids=["openai", "anthropic"],
    )
    def test_the_setting_reaches_the_sdk_client(
        self, dummy_api_keys: None, model_identifier: str, sdk_client: Callable[[Any], Any]
    ) -> None:
        """Built for real (nothing is sent): the SDK client itself is configured not to retry."""
        assert sdk_client(build_chat_model(model_identifier)).max_retries == NO_SDK_RETRIES["max_retries"]
