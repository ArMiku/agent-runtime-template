"""Phase 3 tests: CompositeAgentRunHooks aggregation, ordering, isolation."""

from __future__ import annotations

import pytest

from agent_runtime.core.message import Message
from agent_runtime.core.run_context import ContextWrapper
from agent_runtime.core.session_context import SessionContext
from agent_runtime.extensions.plugins.base import Plugin
from agent_runtime.extensions.plugins.context import PluginContext
from agent_runtime.extensions.plugins.contributions import collect_contribution
from agent_runtime.extensions.plugins.decorators import on_llm_request
from agent_runtime.extensions.plugins.hooks import CompositeAgentRunHooks
from agent_runtime.extensions.plugins.metadata import PluginMetadata


def _ctx(name: str) -> PluginContext:
    return PluginContext(metadata=PluginMetadata(name=name, author="a", desc="d", version="1"))


def _run_context() -> ContextWrapper:
    return ContextWrapper(context=SessionContext(session_id="t"), messages=[])


async def test_two_plugins_hooks_run_in_order_and_accumulate() -> None:
    class _First(Plugin):
        @on_llm_request
        async def a(self, run_context) -> None:
            run_context.messages.append(Message(role="system", content="first"))

    class _Second(Plugin):
        @on_llm_request
        async def b(self, run_context) -> None:
            run_context.messages.append(Message(role="system", content="second"))

    contributions = [
        collect_contribution(_First(_ctx("first"))),
        collect_contribution(_Second(_ctx("second"))),
    ]
    composite = CompositeAgentRunHooks(contributions)

    run_context = _run_context()
    await composite.on_llm_request(run_context)

    contents = [m.content for m in run_context.messages]
    assert contents == ["first", "second"]


async def test_single_plugin_hook_exception_isolated() -> None:
    class _Boom(Plugin):
        @on_llm_request
        async def boom(self, run_context) -> None:
            raise RuntimeError("boom")

    class _Ok(Plugin):
        @on_llm_request
        async def ok(self, run_context) -> None:
            run_context.messages.append(Message(role="system", content="ok"))

    contributions = [
        collect_contribution(_Boom(_ctx("boom"))),
        collect_contribution(_Ok(_ctx("ok"))),
    ]
    composite = CompositeAgentRunHooks(contributions)

    run_context = _run_context()
    # Must not raise; the second plugin still runs.
    await composite.on_llm_request(run_context)
    assert [m.content for m in run_context.messages] == ["ok"]


async def test_empty_composite_is_noop() -> None:
    composite = CompositeAgentRunHooks([])
    run_context = _run_context()
    await composite.on_llm_request(run_context)
    await composite.on_agent_begin(run_context)
    assert run_context.messages == []
