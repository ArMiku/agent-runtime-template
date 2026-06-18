"""Example :class:`Provider` for the driver demo.

``ProviderOpenAIOfficial`` is already the OpenAI-compatible base that covers
DeepSeek / zhipu / groq / xai / openrouter / longcat / xiaomi / minimax-OpenAI and
plain OpenAI (design.md §11). Most consumers need no new adapter code — just a
configured instance. :func:`make_openai_compat_provider` is the ~one-liner way to
get one; subclass :class:`ProviderOpenAIOfficial` directly when you need to override
behaviour.
"""

from __future__ import annotations

from agent_runtime.provider.sources.openai_source import ProviderOpenAIOfficial

__all__ = ["OpenAICompatProvider", "make_openai_compat_provider"]


class OpenAICompatProvider(ProviderOpenAIOfficial):
    """Thin example subclass of the OpenAI-compatible base.

    Identical behaviour to ``ProviderOpenAIOfficial``; exists to show the subclass
    seam and give consumers a named type to extend.
    """


def make_openai_compat_provider(
    api_key: str,
    model: str,
    api_base: str = "https://api.openai.com/v1",
    *,
    provider_id: str = "openai",
    **extra,
) -> ProviderOpenAIOfficial:
    """Build a configured OpenAI-compatible provider.

    Covers OpenAI, DeepSeek, zhipu, groq, xai, openrouter, etc. — pass the vendor's
    ``api_base`` and ``model``. Extra ``provider_config`` keys (``timeout``,
    ``custom_headers``, ``api_version`` for Azure, …) flow through ``**extra``.
    """
    provider_config = {
        "id": provider_id,
        "type": "openai_chat_completion",
        "key": [api_key] if api_key else [],
        "api_base": api_base,
        "model": model,
        **extra,
    }
    return ProviderOpenAIOfficial(provider_config, {})
