"""~50-line driver: provider + tools + system prompt -> final LLM response.

Exercises exactly the four seams (Provider, BaseFunctionToolExecutor,
BaseAgentRunHooks, TContext via ContextWrapper) and nothing else — no
``event`` / ``platform`` / ``star`` objects appear anywhere (spec "四个显式对外接缝").

Run with a real OpenAI-compatible endpoint::

    OPENAI_API_KEY=sk-... OPENAI_MODEL=gpt-4o-mini \
    OPENAI_API_BASE=https://api.openai.com/v1 \
    python -m agent_runtime.examples.driver
"""

from __future__ import annotations

import asyncio
import os

from agent_runtime.core import FunctionToolExecutor, SessionContext
from agent_runtime.core.hooks import BaseAgentRunHooks
from agent_runtime.core.run_context import ContextWrapper
from agent_runtime.core.runners.tool_loop_agent_runner import ToolLoopAgentRunner
from agent_runtime.core.tool import FunctionTool, ToolSet
from agent_runtime.examples.example_provider import make_openai_compat_provider
from agent_runtime.provider.entities import ProviderRequest

__all__ = ["AddTool", "run"]


class AddTool(FunctionTool):
    """A custom tool defined the clean subclass way: override ``call(context, **kwargs)``.

    The first argument is the :class:`ContextWrapper` (carrying the caller's
    ``SessionContext``); the return is a plain ``str``. No event, no MessageEventResult.
    """

    def __init__(self) -> None:
        super().__init__(
            name="add",
            description="Add two integers and return their sum.",
            parameters={
                "type": "object",
                "properties": {
                    "a": {"type": "integer", "description": "first addend"},
                    "b": {"type": "integer", "description": "second addend"},
                },
                "required": ["a", "b"],
            },
        )

    async def call(self, context, **kwargs) -> str:  # noqa: ANN001
        return str(int(kwargs["a"]) + int(kwargs["b"]))


async def run(prompt: str = "What is 7 + 5? Use the add tool.") -> str:
    # Seam 1 — Provider: the OpenAI-compatible base covers OpenAI / DeepSeek / zhipu /
    # groq / xai / openrouter / ... (just change api_base + model).
    provider = make_openai_compat_provider(
        api_key=os.environ["OPENAI_API_KEY"],
        model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        api_base=os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1"),
    )

    # Seam 2 — BaseFunctionToolExecutor: handler/call bispatch (+ sub-agent handoff).
    executor = FunctionToolExecutor(provider)

    tools = ToolSet()
    tools.add_tool(AddTool())

    request = ProviderRequest(
        prompt=prompt,
        system_prompt="You are a helpful assistant. Use tools when appropriate.",
        func_tool=tools,
    )

    # Seam 4 — TContext via ContextWrapper: SessionContext is the default context.
    run_context = ContextWrapper(context=SessionContext(session_id="demo"))

    runner = ToolLoopAgentRunner()
    await runner.reset(
        provider=provider,
        request=request,
        run_context=run_context,
        tool_executor=executor,
        # Seam 3 — hooks are optional in spirit; BaseAgentRunHooks() is the no-op default.
        agent_hooks=BaseAgentRunHooks(),
    )
    async for _ in runner.step_until_done(max_step=10):
        pass

    final = runner.get_final_llm_resp()
    return (final.completion_text if final else "") or ""


if __name__ == "__main__":
    print(asyncio.run(run()))
