"""Tests for ``ChainedAgentRunHooks``: the core primitive that chains N
``BaseAgentRunHooks`` into one (design.md §3).

Covers the three contractual behaviors:

* empty chain is a legal no-op (every event does nothing, never raises);
* N hooks fan out to every child in construction order, once each;
* a child hook raising is isolated per-hook (caught, logged) — siblings still
  run, nothing propagates to the caller.
"""

from __future__ import annotations

import pytest

from agent_runtime.core.hooks import BaseAgentRunHooks
from agent_runtime.core.hooks_chain import ChainedAgentRunHooks
from agent_runtime.core.run_context import ContextWrapper
from agent_runtime.core.session_context import SessionContext


def _ctx() -> ContextWrapper[SessionContext]:
    return ContextWrapper(context=SessionContext(session_id="test"), messages=[])


class _RecordingHooks(BaseAgentRunHooks):
    """Records, into a shared ordered log, the label of each event it sees."""

    def __init__(self, label: str, log: list[str]) -> None:
        self._label = label
        self._log = log

    async def on_agent_begin(self, run_context) -> None:
        self._log.append(f"{self._label}:on_agent_begin")

    async def on_llm_request(self, run_context) -> None:
        self._log.append(f"{self._label}:on_llm_request")

    async def on_tool_start(self, run_context, tool, tool_args) -> None:
        self._log.append(f"{self._label}:on_tool_start")

    async def on_tool_end(self, run_context, tool, tool_args, tool_result) -> None:
        self._log.append(f"{self._label}:on_tool_end")

    async def on_agent_done(self, run_context, llm_response) -> None:
        self._log.append(f"{self._label}:on_agent_done")


class _BoomHooks(BaseAgentRunHooks):
    """Raises on ``on_llm_request`` to exercise per-hook isolation."""

    async def on_llm_request(self, run_context) -> None:
        raise RuntimeError("boom")


@pytest.mark.asyncio
async def test_empty_chain_is_noop():
    """0.1: a zero-hook chain runs every event without side effects or raising,
    and is a valid ``BaseAgentRunHooks``."""
    chain = ChainedAgentRunHooks()
    ctx = _ctx()

    assert isinstance(chain, BaseAgentRunHooks)
    # None of these should raise.
    await chain.on_agent_begin(ctx)
    await chain.on_llm_request(ctx)
    await chain.on_tool_start(ctx, None, None)
    await chain.on_tool_end(ctx, None, None, None)
    await chain.on_agent_done(ctx, None)


@pytest.mark.asyncio
async def test_hooks_fire_in_construction_order():
    """0.2: every child's event fires exactly once, in construction order."""
    log: list[str] = []
    a = _RecordingHooks("a", log)
    b = _RecordingHooks("b", log)
    c = _RecordingHooks("c", log)
    chain = ChainedAgentRunHooks(a, b, c)
    ctx = _ctx()

    await chain.on_llm_request(ctx)
    assert log == ["a:on_llm_request", "b:on_llm_request", "c:on_llm_request"]

    # Other events fan out the same way.
    log.clear()
    await chain.on_agent_begin(ctx)
    assert log == ["a:on_agent_begin", "b:on_agent_begin", "c:on_agent_begin"]

    log.clear()
    await chain.on_tool_start(ctx, None, None)
    assert log == ["a:on_tool_start", "b:on_tool_start", "c:on_tool_start"]

    log.clear()
    await chain.on_tool_end(ctx, None, None, None)
    assert log == ["a:on_tool_end", "b:on_tool_end", "c:on_tool_end"]

    log.clear()
    await chain.on_agent_done(ctx, None)
    assert log == ["a:on_agent_done", "b:on_agent_done", "c:on_agent_done"]


@pytest.mark.asyncio
async def test_one_hook_raising_is_isolated(caplog):
    """0.3: a child raising does not propagate, siblings still run, error logged."""
    log: list[str] = []
    boom = _BoomHooks()
    ok = _RecordingHooks("ok", log)
    chain = ChainedAgentRunHooks(boom, ok)
    ctx = _ctx()

    # Must not raise even though boom.on_llm_request raises.
    await chain.on_llm_request(ctx)

    # The sibling still ran.
    assert log == ["ok:on_llm_request"]
    # The exception was logged.
    assert any("boom" in record.getMessage() or record.exc_info for record in caplog.records)
