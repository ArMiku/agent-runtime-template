"""Phase 0 tests: the ``on_llm_request`` hook on ``BaseAgentRunHooks``.

These exercise the承重墙 added in ``add-plugin-system``: the hook fires before each
LLM step builds its provider payload, mutations to ``run_context.messages`` reach the
provider, it fires once per step, and an exception in the hook is isolated.
"""

from __future__ import annotations

import pytest

from agent_runtime.core.hooks import BaseAgentRunHooks
from agent_runtime.core.message import Message
from agent_runtime.core.run_context import ContextWrapper

from .fakes import FakeProvider, llm_text, llm_tool_call, run


class _InjectingHooks(BaseAgentRunHooks):
    """Inserts a system message at the front of the context on every LLM step."""

    def __init__(self) -> None:
        self.calls = 0

    async def on_llm_request(self, run_context: ContextWrapper) -> None:
        self.calls += 1
        run_context.messages.insert(0, Message(role="system", content=f"[injected #{self.calls}]"))


async def _run_with_hooks(provider, tools, prompt, hooks, *, max_step=10):
    from agent_runtime.core import FunctionToolExecutor, SessionContext
    from agent_runtime.core.runners.tool_loop_agent_runner import ToolLoopAgentRunner
    from agent_runtime.provider.entities import ProviderRequest

    executor = FunctionToolExecutor(provider)
    request = ProviderRequest(prompt=prompt, system_prompt="", func_tool=tools)
    run_context = ContextWrapper(context=SessionContext(session_id="test"), messages=[])
    runner = ToolLoopAgentRunner()
    await runner.reset(
        provider=provider,
        request=request,
        run_context=run_context,
        tool_executor=executor,
        agent_hooks=hooks,
        streaming=False,
    )
    responses = []
    async for resp in runner.step_until_done(max_step):
        responses.append(resp)
    return runner.get_final_llm_resp(), responses


async def test_hook_injected_message_reaches_provider() -> None:
    provider = FakeProvider([llm_text("done")])
    hooks = _InjectingHooks()

    await _run_with_hooks(provider, None, "hello", hooks)

    assert hooks.calls == 1
    # The provider received the injected system message in its contexts payload.
    contexts = provider.calls[0]["contexts"]
    roles_and_content = [(m.role, m.content) for m in contexts]
    assert ("system", "[injected #1]") in roles_and_content


async def test_hook_fires_before_payload_built() -> None:
    """The injected message must be present in the very payload of the same step."""

    class _AssertingProvider(FakeProvider):
        def __init__(self, script):
            super().__init__(script)
            self.saw_injection = False

        async def text_chat(self, **kwargs):
            for m in kwargs["contexts"]:
                if m.role == "system" and m.content == "[injected #1]":
                    self.saw_injection = True
            return await super().text_chat(**kwargs)

    provider = _AssertingProvider([llm_text("done")])
    await _run_with_hooks(provider, None, "hi", _InjectingHooks())
    assert provider.saw_injection


async def test_hook_fires_once_per_step() -> None:
    """ReAct multi-step: hook fires once per LLM step (here: tool call + final)."""
    from agent_runtime.core.tool import FunctionTool, ToolSet

    async def echo(run_context, text: str) -> str:
        return text

    tool = FunctionTool(
        name="echo",
        description="echo",
        parameters={"type": "object", "properties": {"text": {"type": "string"}}},
        handler=echo,
    )
    provider = FakeProvider(
        [
            llm_tool_call("echo", {"text": "x"}),
            llm_text("done"),
        ]
    )
    hooks = _InjectingHooks()
    await _run_with_hooks(provider, ToolSet([tool]), "go", hooks)
    # Two LLM steps -> two hook fires.
    assert hooks.calls == 2


async def test_hook_exception_does_not_break_run() -> None:
    class _BoomHooks(BaseAgentRunHooks):
        async def on_llm_request(self, run_context: ContextWrapper) -> None:
            raise RuntimeError("boom")

    provider = FakeProvider([llm_text("survived")])
    final, _ = await _run_with_hooks(provider, None, "hi", _BoomHooks())
    assert final is not None
    assert final.completion_text == "survived"


async def test_default_hook_is_noop() -> None:
    provider = FakeProvider([llm_text("ok")])
    final, _ = await run(provider, None, "hi")
    assert final is not None
    assert final.completion_text == "ok"
