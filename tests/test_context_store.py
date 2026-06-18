"""6.5: persistence is caller-provided via the ContextStore seam (load before, save after)."""

from __future__ import annotations

import pytest

from agent_runtime.core import FunctionToolExecutor, SessionContext
from agent_runtime.core.context import ContextStore, InMemoryContextStore
from agent_runtime.core.hooks import BaseAgentRunHooks
from agent_runtime.core.run_context import ContextWrapper
from agent_runtime.core.runners.tool_loop_agent_runner import ToolLoopAgentRunner
from agent_runtime.provider.entities import ProviderRequest
from agent_runtime.tests.fakes import FakeProvider, llm_text


@pytest.mark.asyncio
async def test_in_memory_store_roundtrip():
    store: ContextStore = InMemoryContextStore()
    assert await store.load("s1") == []  # new session -> empty history


@pytest.mark.asyncio
async def test_load_before_save_after_run():
    """The caller loads history into the run, then persists the runner's messages back."""
    store = InMemoryContextStore()
    provider = FakeProvider([llm_text("ok")])

    history = await store.load("s1")
    run_context = ContextWrapper(
        context=SessionContext(session_id="s1"),
        messages=list(history),
    )
    runner = ToolLoopAgentRunner()
    await runner.reset(
        provider=provider,
        request=ProviderRequest(prompt="hi"),
        run_context=run_context,
        tool_executor=FunctionToolExecutor(provider),
        agent_hooks=BaseAgentRunHooks(),
    )
    async for _ in runner.step_until_done(10):
        pass

    await store.save("s1", runner.run_context.messages)

    loaded = await store.load("s1")
    # The runner appends an assistant message during the run; it must round-trip.
    assert loaded, "saved history must be non-empty"
    assert any(msg.role == "assistant" for msg in loaded)
