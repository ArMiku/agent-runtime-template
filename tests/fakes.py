"""Test doubles for the agent-runtime test suite.

``FakeProvider`` is a scripted :class:`~agent_runtime.provider.provider.Provider` that
returns queued :class:`~agent_runtime.provider.entities.LLMResponse` objects, so the
ReAct loop can be exercised end-to-end with no network or API key.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, Sequence

from agent_runtime.core import FunctionToolExecutor, SessionContext
from agent_runtime.core.hooks import BaseAgentRunHooks
from agent_runtime.core.run_context import ContextWrapper
from agent_runtime.core.runners.tool_loop_agent_runner import ToolLoopAgentRunner
from agent_runtime.core.tool import ToolSet
from agent_runtime.provider.entities import LLMResponse, ProviderRequest
from agent_runtime.provider.provider import Provider

__all__ = ["FakeProvider", "llm_text", "llm_tool_call", "run"]


class FakeProvider(Provider):
    """Scripted provider returning LLMResponse objects from a queue."""

    def __init__(self, script: Sequence[LLMResponse], *, chunk_text: bool = False) -> None:
        super().__init__(
            {
                "id": "fake",
                "type": "fake_chat",
                "max_context_tokens": 0,  # disable token-based context guarding
                "modalities": [],
            },
            {},
        )
        self._script: list[LLMResponse] = list(script)
        self._idx = 0
        self.chunk_text = chunk_text
        self.calls: list[dict] = []

    def _next(self) -> LLMResponse:
        if self._idx >= len(self._script):
            return LLMResponse(role="assistant", completion_text="")
        resp = self._script[self._idx]
        self._idx += 1
        return resp

    # --- Provider ABC -------------------------------------------------
    def get_current_key(self) -> str:
        return "fake-key"

    def set_key(self, key: str) -> None:  # noqa: ARG002
        ...

    async def get_models(self) -> list[str]:
        return ["fake-model"]

    async def text_chat(self, **kwargs) -> LLMResponse:
        self.calls.append(kwargs)
        return self._next()

    async def text_chat_stream(self, **kwargs) -> AsyncGenerator[LLMResponse, None]:
        self.calls.append(kwargs)
        resp = self._next()
        # Provider contract: yield is_chunk=True increments, then one full response.
        if self.chunk_text and resp.completion_text and not resp.tools_call_name:
            text = resp.completion_text
            mid = max(1, len(text) // 2)
            yield LLMResponse(
                role="assistant",
                completion_text=text[:mid],
                is_chunk=True,
                reasoning_content=resp.reasoning_content,
            )
            yield LLMResponse(
                role="assistant",
                completion_text=text[mid:],
                is_chunk=True,
            )
        yield resp


def llm_text(text: str, *, reasoning: str | None = None) -> LLMResponse:
    """A non-tool assistant response."""
    return LLMResponse(
        role="assistant",
        completion_text=text,
        reasoning_content=reasoning,
    )


def llm_tool_call(name: str, args: dict, call_id: str = "call_1") -> LLMResponse:
    """An assistant response requesting one tool call."""
    return LLMResponse(
        role="assistant",
        tools_call_name=[name],
        tools_call_args=[args],
        tools_call_ids=[call_id],
    )


async def run(
    provider: Provider,
    tools: ToolSet | None,
    prompt: str,
    *,
    streaming: bool = False,
    max_step: int = 10,
    system_prompt: str = "",
    session: SessionContext | None = None,
):
    """Drive a fresh ToolLoopAgentRunner one full run; return (final_resp, responses)."""
    executor = FunctionToolExecutor(provider)
    request = ProviderRequest(
        prompt=prompt,
        system_prompt=system_prompt,
        func_tool=tools,
    )
    run_context = ContextWrapper(
        context=session or SessionContext(session_id="test"),
        messages=[],
    )
    runner = ToolLoopAgentRunner()
    await runner.reset(
        provider=provider,
        request=request,
        run_context=run_context,
        tool_executor=executor,
        agent_hooks=BaseAgentRunHooks(),
        streaming=streaming,
    )
    responses = []
    async for resp in runner.step_until_done(max_step):
        responses.append(resp)
    return runner.get_final_llm_resp(), responses
