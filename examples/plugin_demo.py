"""End-to-end example: a neutral plugin driving the runtime.

This proves the plugin *mechanism* is self-consistent — it deliberately implements **no**
real capability (memory / retrieval belong to their own later changes). ``ExamplePlugin``
exercises every seam:

* lifecycle — ``initialize`` seeds a private counter;
* private persistence — ``put_kv_data`` via the ``PluginStore`` seam;
* tool contribution — ``@tool`` exposes ``echo``;
* loop hook — ``@on_llm_request`` mutates ``run_context.messages`` (scheme X).

Run it::

    python -m examples.plugin_demo
"""

from __future__ import annotations

import asyncio

from agent_runtime.core import FunctionToolExecutor, SessionContext
from agent_runtime.core.message import Message
from agent_runtime.core.run_context import ContextWrapper
from agent_runtime.core.runners.tool_loop_agent_runner import ToolLoopAgentRunner
from agent_runtime.core.tool import ToolSet
from agent_runtime.extensions.plugins import (
    CompositeAgentRunHooks,
    Plugin,
    PluginManager,
    on_llm_request,
    tool,
)
from agent_runtime.provider.entities import LLMResponse, ProviderRequest
from agent_runtime.provider.provider import Provider

INJECTED_SYSTEM_NOTE = "[example] injected by plugin"


class ExamplePlugin(Plugin):
    """A mechanism-only plugin: lifecycle + KV + tool + loop hook."""

    name, author, desc, version = "example", "rt", "mechanism check", "0.1.0"

    async def initialize(self) -> None:
        await self.put_kv_data("count", 0)

    @tool
    async def echo(self, run_context, text: str) -> str:
        """Echo the input text.

        Args:
            text(string): the text to echo back
        """
        return text

    @on_llm_request
    async def inject(self, run_context) -> None:
        run_context.messages.insert(0, Message(role="system", content=INJECTED_SYSTEM_NOTE))


class _DemoProvider(Provider):
    """Records the contexts it is asked to chat over, then returns a fixed reply."""

    def __init__(self) -> None:
        super().__init__({"id": "demo", "type": "demo", "max_context_tokens": 0, "modalities": []}, {})
        self.seen_contexts: list = []

    def get_current_key(self) -> str:
        return "demo-key"

    def set_key(self, key: str) -> None: ...

    async def get_models(self) -> list[str]:
        return ["demo-model"]

    async def text_chat(self, *args, **kwargs) -> LLMResponse:
        self.seen_contexts = kwargs.get("contexts") or []
        return LLMResponse(role="assistant", completion_text="done")

    async def text_chat_stream(self, *args, **kwargs):
        yield await self.text_chat(*args, **kwargs)


async def main() -> dict:
    manager = PluginManager()
    contribution = await manager.register(ExamplePlugin)

    # Tools the plugin contributes feed the request's func_tool.
    tools = ToolSet(list(contribution.tools))
    # Hooks the plugin registers aggregate into one BaseAgentRunHooks for the runner.
    hooks = CompositeAgentRunHooks(manager.contributions)

    provider = _DemoProvider()
    request = ProviderRequest(prompt="hello", system_prompt="", func_tool=tools)
    run_context = ContextWrapper(context=SessionContext(session_id="demo"), messages=[])

    runner = ToolLoopAgentRunner()
    await runner.reset(
        provider=provider,
        request=request,
        run_context=run_context,
        tool_executor=FunctionToolExecutor(provider),
        agent_hooks=hooks,
    )
    async for _ in runner.step_until_done(max_step=5):
        pass

    injected = any(
        getattr(m, "role", None) == "system" and getattr(m, "content", None) == INJECTED_SYSTEM_NOTE
        for m in provider.seen_contexts
    )
    final = runner.get_final_llm_resp()
    result = {
        "tools": tools.names(),
        "provider_saw_injected_message": injected,
        "final_text": final.completion_text if final else None,
    }
    print(result)
    return result


if __name__ == "__main__":
    asyncio.run(main())
