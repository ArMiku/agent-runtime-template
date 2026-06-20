"""Chain N ``BaseAgentRunHooks`` into one.

The runner's ``reset(agent_hooks=...)`` accepts a *single* ``BaseAgentRunHooks``, but
local assembly routinely needs two independent hooks live at once — the skills-inventory
hook and the aggregated plugin hooks. ``ChainedAgentRunHooks`` wraps any number of hook
objects and fans each event out to all of them in construction order.

A single child hook raising must not break its siblings or the run, so each call is
wrapped in try/except and logged — the fault-tolerance semantics are byte-for-byte the
same as ``CompositeAgentRunHooks._dispatch`` (per-hook try/except + ``logger.error(...,
exc_info=True)``). The empty chain is a legal no-op, letting the assembly layer always
hand the runner a valid hook object without a ``None`` special case.

This primitive depends only on ``core`` (``BaseAgentRunHooks`` + the package logger); it
imports nothing from ``extensions/*``.
"""

from __future__ import annotations

import mcp

from agent_runtime import logger
from agent_runtime.core.hooks import BaseAgentRunHooks
from agent_runtime.core.run_context import ContextWrapper, TContext
from agent_runtime.core.tool import FunctionTool
from agent_runtime.provider.entities import LLMResponse

__all__ = ["ChainedAgentRunHooks"]


class ChainedAgentRunHooks(BaseAgentRunHooks[TContext]):
    """Chain several ``BaseAgentRunHooks`` and fan each event out to all of them."""

    def __init__(self, *hooks: BaseAgentRunHooks[TContext]) -> None:
        self._hooks = list(hooks)

    async def _dispatch(self, event: str, *args) -> None:
        for hook in self._hooks:
            try:
                await getattr(hook, event)(*args)
            except Exception as e:  # noqa: BLE001 - isolate one hook's failure
                logger.error(f"Error in chained {event} hook: {e}", exc_info=True)

    async def on_agent_begin(self, run_context: ContextWrapper[TContext]) -> None:
        await self._dispatch("on_agent_begin", run_context)

    async def on_llm_request(self, run_context: ContextWrapper[TContext]) -> None:
        await self._dispatch("on_llm_request", run_context)

    async def on_agent_done(
        self,
        run_context: ContextWrapper[TContext],
        llm_response: LLMResponse,
    ) -> None:
        await self._dispatch("on_agent_done", run_context, llm_response)

    async def on_before_complete(
        self,
        run_context: ContextWrapper[TContext],
        llm_response: LLMResponse,
    ) -> bool:
        """Fan completion veto out to every child; any ``False`` vetoes the whole.

        Each child is polled in construction order with the same per-hook try/except
        isolation as ``_dispatch`` — a raising child is logged and treated as an admit,
        never breaking the vote. The aggregate admits only when *every* child admits; a
        single ``False`` vetoes the completion. The empty chain admits (returns ``True``).
        """
        admit = True
        for hook in self._hooks:
            try:
                if await hook.on_before_complete(run_context, llm_response) is False:
                    admit = False
            except Exception as e:  # noqa: BLE001 - isolate one hook's failure
                logger.error(f"Error in chained on_before_complete hook: {e}", exc_info=True)
        return admit

    async def on_tool_start(
        self,
        run_context: ContextWrapper[TContext],
        tool: FunctionTool,
        tool_args: dict | None,
    ) -> None:
        await self._dispatch("on_tool_start", run_context, tool, tool_args)

    async def on_tool_end(
        self,
        run_context: ContextWrapper[TContext],
        tool: FunctionTool,
        tool_args: dict | None,
        tool_result: mcp.types.CallToolResult | None,
    ) -> None:
        await self._dispatch("on_tool_end", run_context, tool, tool_args, tool_result)
