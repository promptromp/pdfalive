"""Tests for chat model construction."""

from collections.abc import Iterator
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from pdfalive.llm import build_chat_model


MODEL_IDENTIFIER = "gpt-6-luna"


@pytest.fixture
def mock_init_chat_model() -> Iterator[MagicMock]:
    """Patch the LangChain initializer and expose the recorded call."""
    with patch("pdfalive.llm.init_chat_model") as mock:
        yield mock


def _call_kwargs(mock: MagicMock) -> dict[str, Any]:
    return dict(mock.call_args.kwargs)


class TestBuildChatModel:
    def test_omits_reasoning_effort_when_unset(self, mock_init_chat_model: MagicMock) -> None:
        """An unset effort must not reach the provider at all, not even as None.

        Providers treat an explicit None as a meaningful value, so passing it
        would change behavior for users who never opted in.
        """
        build_chat_model(MODEL_IDENTIFIER)

        assert _call_kwargs(mock_init_chat_model) == {"model": MODEL_IDENTIFIER}

    @pytest.mark.parametrize("effort", ["none", "low", "medium", "high", "xhigh", "max"])
    def test_passes_effort_through_unvalidated(self, mock_init_chat_model: MagicMock, effort: str) -> None:
        """Accepted levels differ per model, so the provider validates, not us."""
        build_chat_model(MODEL_IDENTIFIER, effort)

        assert _call_kwargs(mock_init_chat_model) == {
            "model": MODEL_IDENTIFIER,
            "reasoning_effort": effort,
        }

    def test_returns_the_initialized_model(self, mock_init_chat_model: MagicMock) -> None:
        assert build_chat_model(MODEL_IDENTIFIER) is mock_init_chat_model.return_value
