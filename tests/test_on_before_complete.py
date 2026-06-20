"""Kernel ``on_before_complete`` veto: default admit, chain/composite fan-out with
veto semantics, and the runner staying RUNNING so ``step_until_done`` continues."""

from __future__ import annotations

import pytest

from agent_runtime.core.hooks import BaseAgentRunHooks
from agent_runtime.core.hooks_chain import ChainedAgentRunHooks
from agent_runtime.core.message import Message
from agent_runtime.core.run_context import ContextWrapper
from agent_runtime.core.session_context import SessionContext

from .fakes import FakeProvider, llm_text, llm_tool_call


def _ctx() -> ContextWrapper[SessionContext]:
    return ContextWrapper(context=SessionContext(session_id="test"), messages=[])


class _VetoOnce(BaseAgentRunHooks):
    """Vetoes the first completion, admits afterward; appends a reminder on veto."""

    def __init__(self) -> None:
        self.calls = 0

    async def on_before_complete(self, run_context, llm_response) -> bool:
        self.calls += 1
        if self.calls == 1:
            run_context.messages.append(Message(role="user", content="keep going"))
            return False
        return True


async def test_default_on_before_complete_admits():
    """2.1: the base hook admits completion unconditionally (backward-compatible)."""
    assert await BaseAgentRunHooks().on_before_complete(_ctx(), None) is True


async def test_chain_admits_when_all_admit():
    """2.2: an all-admit chain (and the empty chain) admits."""
    assert await ChainedAgentRunHooks().on_before_complete(_ctx(), None) is True
    chain = ChainedAgentRunHooks(BaseAgentRunHooks(), BaseAgentRunHooks())
    assert await chain.on_before_complete(_ctx(), None) is True


async def test_chain_vetoes_when_any_child_vetoes():
    """2.2: a single child returning False vetoes the whole chain."""

    class _Veto(BaseAgentRunHooks):
        async def on_before_complete(self, run_context, llm_response) -> bool:
            return False

    chain = ChainedAgentRunHooks(BaseAgentRunHooks(), _Veto(), BaseAgentRunHooks())
    assert await chain.on_before_complete(_ctx(), None) is False


async def test_chain_veto_isolates_raising_child(caplog):
    """2.2: a raising child is logged and treated as an admit, never breaking the vote."""

    class _Boom(BaseAgentRunHooks):
        async def on_before_complete(self, run_context, llm_response) -> bool:
            raise RuntimeError("boom")

    chain = ChainedAgentRunHooks(_Boom(), BaseAgentRunHooks())
    # Boom is treated as admit; the other admits → overall admit, no propagation.
    assert await chain.on_before_complete(_ctx(), None) is True
    assert any("boom" in r.getMessage() or r.exc_info for r in caplog.records)


async def test_composite_vetoes_when_any_plugin_vetoes():
    """2.3: CompositeAgentRunHooks fan-out — any plugin returning False vetoes."""
    from agent_runtime.extensions.plugins.contributions import PluginContribution
    from agent_runtime.extensions.plugins.hooks import CompositeAgentRunHooks

    async def _veto(run_context, llm_response) -> bool:
        return False

    async def _admit(run_context, llm_response) -> bool:
        return True

    # Construct contributions directly with hook_methods keyed on the new event; the
    # composite reads that dict, so no plugin decorator is needed to exercise fan-out.
    admit_only = CompositeAgentRunHooks(
        [PluginContribution(plugin=None, hook_methods={"on_before_complete": [_admit]})]
    )
    assert await admit_only.on_before_complete(_ctx(), None) is True

    with_veto = CompositeAgentRunHooks(
        [PluginContribution(plugin=None, hook_methods={"on_before_complete": [_admit, _veto]})]
    )
    assert await with_veto.on_before_complete(_ctx(), None) is False


async def test_runner_continues_after_veto():
    """2.4/2.5: a veto keeps the agent RUNNING and step_until_done runs another round;
    the second (admitted) completion finalizes with the later response."""
    from agent_runtime.core import FunctionToolExecutor
    from agent_runtime.core.runners.tool_loop_agent_runner import ToolLoopAgentRunner
    from agent_runtime.provider.entities import ProviderRequest

    # The model tries to finish twice (two no-tool responses). The hook vetoes the first.
    provider = FakeProvider([llm_text("first attempt"), llm_text("second attempt")])
    hook = _VetoOnce()
    request = ProviderRequest(prompt="go", system_prompt="", func_tool=None)
    run_context = ContextWrapper(context=SessionContext(session_id="test"), messages=[])
    runner = ToolLoopAgentRunner()
    await runner.reset(
        provider=provider,
        request=request,
        run_context=run_context,
        tool_executor=FunctionToolExecutor(provider),
        agent_hooks=hook,
    )
    async for _ in runner.step_until_done(max_step=10):
        pass

    assert hook.calls == 2  # vetoed once, then admitted
    assert runner.done()
    final = runner.get_final_llm_resp()
    assert final is not None and final.completion_text == "second attempt"
    # The veto reminder made it into history between the two attempts.
    assert any(getattr(m, "content", None) == "keep going" for m in run_context.messages)


async def test_unrelated_tool_call_path_unaffected():
    """2.4: on_before_complete only gates the no-tool-call completion; a normal tool
    round still proceeds to a later completion."""
    from agent_runtime.core.tool import FunctionTool, ToolSet

    async def _noop(run_context, **kwargs) -> str:
        return "ok"

    tool = FunctionTool(
        name="noop",
        description="noop",
        parameters={"type": "object", "properties": {}},
        handler=_noop,
    )
    provider = FakeProvider([llm_tool_call("noop", {}), llm_text("done")])
    from .fakes import run

    final, _ = await run(provider, ToolSet([tool]), "go")
    assert final is not None and final.completion_text == "done"
