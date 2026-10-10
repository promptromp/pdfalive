"""Chat model construction shared by the CLI and the evaluation runner."""

from typing import Any

from langchain.chat_models import init_chat_model
from langchain.chat_models.base import BaseChatModel


# Settings that switch a provider SDK's own retries off (OpenAI and Anthropic retry twice by
# default), by LangChain provider name, so pdfalive.retry is the only retry layer. Other
# providers get no extra setting and keep whatever their SDK does: Ollama's client has no
# retries to switch off, and a provider whose SDK retries is retried by both layers.
# Trade-off: the SDKs' own retries come within a second (useful for a dropped keep-alive
# connection), while the policy's first wait is 10 s; one predictable layer is worth the
# slower recovery from such blips.
_SDK_RETRIES_OFF: dict[str, dict[str, Any]] = {
    "openai": {"max_retries": 0},
    "anthropic": {"max_retries": 0},
}

# A model identifier names its provider explicitly as "provider:model" ...
_PROVIDER_SEPARATOR = ":"
# ... or LangChain infers it from the model name. These are the prefixes it infers the
# providers above from, mirrored from LangChain's private _attempt_infer_model_provider()
# because it has no public way to ask. Known limitation: if LangChain adds a prefix for
# these providers, such a model is unknown here and keeps its SDK's retries (both layers
# retry) until this table follows; it is never given a wrong setting.
_PROVIDER_MODEL_PREFIXES: dict[str, tuple[str, ...]] = {
    "openai": ("gpt-", "o1", "o3", "chatgpt", "text-davinci"),
    "anthropic": ("claude",),
}


def _provider(model_identifier: str) -> str | None:
    """The LangChain provider of a model identifier, as far as _SDK_RETRIES_OFF needs it; None if not known."""
    provider, separator, _model = model_identifier.partition(_PROVIDER_SEPARATOR)
    if separator:
        return provider.lower()
    name = model_identifier.lower()
    return next(
        (provider for provider, prefixes in _PROVIDER_MODEL_PREFIXES.items() if name.startswith(prefixes)), None
    )


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

    For OpenAI and Anthropic models the SDK's own retries are switched off, so
    pdfalive.retry is the only retry layer and a failure is not retried twice
    over (see _SDK_RETRIES_OFF).

    Args:
        model_identifier: LangChain model identifier, e.g. ``gpt-6-luna``.
        reasoning_effort: Reasoning effort level, or None to leave the
            provider's own default in place.

    Returns:
        The initialized chat model.
    """
    provider = _provider(model_identifier)
    extra_kwargs: dict[str, Any] = dict(_SDK_RETRIES_OFF.get(provider, {})) if provider is not None else {}
    # Omitted when unset rather than passed as None: an explicit None is a
    # meaningful value to some providers, which keep their default otherwise.
    if reasoning_effort is not None:
        extra_kwargs["reasoning_effort"] = reasoning_effort
    return init_chat_model(model=model_identifier, **extra_kwargs)
