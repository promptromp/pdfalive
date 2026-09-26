"""Chat model construction shared by the CLI and the evaluation runner."""

from typing import Any

from langchain.chat_models import init_chat_model
from langchain.chat_models.base import BaseChatModel


def build_chat_model(model_identifier: str, reasoning_effort: str | None = None) -> BaseChatModel:
    """Initialize a LangChain chat model, optionally constraining reasoning effort.

    ``reasoning_effort`` is the parameter name OpenAI, Anthropic and LangChain
    all use for this control, so one value covers every provider that supports
    it. Providers that have no such parameter (Ollama, for instance) accept and
    ignore it.

    The value is passed through without client-side validation because the
    accepted levels differ per model, not per provider: ``gpt-6-astra`` takes
    low/medium/high/xhigh while ``gpt-6-sol`` and ``gpt-6-luna`` also take
    ``none``, and Anthropic additionally offers ``max``. Validating here would
    mean rejecting levels that some model does accept, or accepting levels it
    does not. Providers reject unsupported levels with an error that names the
    ones that model allows, which is a better message than a hard-coded list
    could give, and it is a client error so the retry policy surfaces it
    immediately rather than retrying.

    Args:
        model_identifier: LangChain model identifier, e.g. ``gpt-6-luna``.
        reasoning_effort: Reasoning effort level, or None to leave the
            provider's own default in place.

    Returns:
        The initialized chat model.
    """
    # Omitted rather than passed as None so providers keep their own defaults:
    # an explicit None is a meaningful value to some of them.
    extra_kwargs: dict[str, Any] = {}
    if reasoning_effort is not None:
        extra_kwargs["reasoning_effort"] = reasoning_effort
    return init_chat_model(model=model_identifier, **extra_kwargs)
