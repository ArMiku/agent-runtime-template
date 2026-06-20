"""Composite hooks: aggregate many plugins' hooks into one ``BaseAgentRunHooks``.

The runner's ``reset(agent_hooks=...)`` still takes a single ``BaseAgentRunHooks``; this
composite wraps N plugins' hook methods and dispatches each event to them in plugin load
order. A single plugin hook raising must not break the other plugins' hooks or the whole
run, so each call is wrapped in try/except and logged — matching the runner's existing
per-hook fault tolerance.
"""

from __future__ import annotations

from collections.abc import Callable

import mcp

from agent_runtime import logger
from agent_runtime.core.hooks import BaseAgentRunHooks
from agent_runtime.core.run_context import ContextWrapper, TContext
from agent_runtime.core.tool import FunctionTool
from agent_runtime.provider.entities import LLMResponse

from .contributions import PluginContribution

__all__ = ["CompositeAgentRunHooks"]


class CompositeAgentRunHooks(BaseAgentRunHooks[TContext]):
    """Aggregate the hook methods of several plugin contributions."""

    def __init__(self, contributions: list[PluginContribution]) -> None:
        self._contributions = list(contributions)

    def _hooks_for(self, event: str) -> list[Callable]:
        hooks: list[Callable] = []
        for contribution in self._contributions:
            hooks.extend(contribution.hook_methods.get(event, []))
        return hooks

    async def _dispatch(self, event: str, *args) -> None:
        for hook in self._hooks_for(event):
            try:
                await hook(*args)
            except Exception as e:  # noqa: BLE001 - isolate one plugin's failure
                logger.error(f"Error in plugin {event} hook: {e}", exc_info=True)

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
        """Poll every plugin's completion veto; any ``False`` vetoes the whole.

        Each plugin hook is polled in load order with the same per-hook try/except
        isolation as ``_dispatch`` — a raising plugin is logged and treated as an admit,
        never breaking the vote. The aggregate admits only when *every* plugin admits.
        """
        admit = True
        for hook in self._hooks_for("on_before_complete"):
            try:
                if await hook(run_context, llm_response) is False:
                    admit = False
            except Exception as e:  # noqa: BLE001 - isolate one plugin's failure
                logger.error(f"Error in plugin on_before_complete hook: {e}", exc_info=True)
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
