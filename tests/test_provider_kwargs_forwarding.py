"""Provider kwargs forwarding: ``text_chat``'s ``**kwargs`` used to be silently dropped, so
per-call ``max_tokens`` / ``reasoning_effort`` / ``extra_body`` never reached the SDK. The fix
lives in ``ProviderOpenAIOfficial._prepare_chat_payload`` (the only place kwargs enter the
payload that ``_query`` sends to the API). These tests guard:

* genuine API params (``max_tokens`` / ``reasoning_effort``) are forwarded into the payload;
* ``extra_body`` is unpacked to top-level payload keys (not nested) so ``_query``'s existing
  routing sends default-param keys direct and the rest via ``extra_body``;
* internal control keys (``abort_signal`` from ``ToolLoopAgentRunner``) are dropped and can
  never reach the API.
"""

from __future__ import annotations

import asyncio

from agent_runtime.provider.sources.openai_source import ProviderOpenAIOfficial


def _make_source() -> ProviderOpenAIOfficial:
    # Construction is lazy — no network call. ``default_params`` is read from the real SDK
    # signature, so the allowlist self-adapts to the installed openai version.
    return ProviderOpenAIOfficial(
        {
            "id": "t",
            "type": "openai_chat_completion",
            "key": ["k"],
            "api_base": "http://localhost",
            "model": "gpt-4o-mini",
        },
        {},
    )


async def test_api_params_forwarded_into_payload():
    src = _make_source()
    payloads, _ = await src._prepare_chat_payload(
        prompt="hi",
        max_tokens=100,
        reasoning_effort="low",
    )
    assert payloads.get("max_tokens") == 100
    assert payloads.get("reasoning_effort") == "low"


async def test_extra_body_unpacked_to_top_level_keys():
    src = _make_source()
    payloads, _ = await src._prepare_chat_payload(
        prompt="hi",
        extra_body={"thinking": False},
    )
    # Unpacked (not nested under "extra_body") — _query routes non-default keys to extra_body.
    assert payloads.get("thinking") is False
    assert "extra_body" not in payloads


async def test_internal_control_keys_dropped():
    """abort_signal (sent by ToolLoopAgentRunner) must never reach the API payload."""
    src = _make_source()
    payloads, _ = await src._prepare_chat_payload(
        prompt="hi",
        max_tokens=100,
        extra_body={"thinking": False},
        abort_signal=asyncio.Event(),
        session_id="s",
    )
    assert "abort_signal" not in payloads
    assert "session_id" not in payloads
    # The real params still made it through alongside the dropped internals.
    assert payloads.get("max_tokens") == 100
    assert payloads.get("thinking") is False


def test_source_constructs_without_network():
    # Smoke: the source builds (sets default_params from the SDK signature) without any API.
    src = _make_source()
    # The allowlist self-adapts to the installed SDK; these are the keys the planner relies on.
    assert "max_tokens" in src.default_params
    assert "reasoning_effort" in src.default_params


async def test_explicit_kwarg_overrides_extra_body_precedence():
    """An explicit typed kwarg wins over the same key supplied via extra_body — the per-call
    value is the more deliberate signal, and the outcome must not depend on dict ordering."""
    src = _make_source()
    payloads, _ = await src._prepare_chat_payload(
        prompt="hi",
        max_tokens=8192,
        extra_body={"max_tokens": 4096, "thinking": False},
    )
    # Explicit max_tokens wins; the thinking extension still flows through.
    assert payloads.get("max_tokens") == 8192
    assert payloads.get("thinking") is False
