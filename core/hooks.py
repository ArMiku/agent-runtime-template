from typing import Generic

import mcp

from agent_runtime.core.tool import FunctionTool
from agent_runtime.provider.entities import LLMResponse

from .run_context import ContextWrapper, TContext


class BaseAgentRunHooks(Generic[TContext]):
    async def on_agent_begin(self, run_context: ContextWrapper[TContext]) -> None: ...
    async def on_llm_request(self, run_context: ContextWrapper[TContext]) -> None:
        """Fired before each LLM step assembles its provider payload.

        Hooks influence the next LLM call by mutating ``run_context.messages`` — the
        message list the runner actually sends to the provider and rebuilds the payload
        from every step. The mutation takes effect *before* that step's context
        compaction. Only ``run_context`` is passed: no ``event`` / ``req`` / platform
        objects. The default is a no-op, so existing hooks subclasses stay
        backward-compatible.
        """
        ...
    async def on_tool_start(
        self,
        run_context: ContextWrapper[TContext],
        tool: FunctionTool,
        tool_args: dict | None,
    ) -> None: ...
    async def on_tool_end(
        self,
        run_context: ContextWrapper[TContext],
        tool: FunctionTool,
        tool_args: dict | None,
        tool_result: mcp.types.CallToolResult | None,
    ) -> None: ...
    async def on_agent_done(
        self,
        run_context: ContextWrapper[TContext],
        llm_response: LLMResponse,
    ) -> None: ...
