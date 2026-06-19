"""Default :class:`BaseFunctionToolExecutor` — the clean tool-execution seam (§5.2).

With ``event`` washed out of the contract, tool execution collapses to a handler/call
bispatch:

* :class:`HandoffTool`  → run the embedded sub-agent via a fresh
  :class:`~agent_runtime.core.runners.tool_loop_agent_runner.ToolLoopAgentRunner`
  sharing this executor's provider/hooks; yield the sub-agent's final text.
* function-style tool (``handler`` set) → ``tool.handler(run_context, **params)``.
* subclass / :class:`~agent_runtime.core.mcp_client.MCPTool` (``call`` overridden)
  → ``tool.call(run_context, **params)``.

No ``event``, no ``MessageEventResult`` — the spec's "不含 event 的工具执行契约".
"""

from __future__ import annotations

import inspect
from collections.abc import AsyncGenerator, Awaitable
from typing import cast

import mcp

from agent_runtime.core.handoff import HandoffTool
from agent_runtime.core.hooks import BaseAgentRunHooks
from agent_runtime.core.message import Message
from agent_runtime.core.run_context import ContextWrapper, TContext
from agent_runtime.core.runners.tool_loop_agent_runner import ToolLoopAgentRunner
from agent_runtime.core.tool import FunctionTool, ToolExecResult, ToolSet
from agent_runtime.core.tool_executor import BaseFunctionToolExecutor
from agent_runtime.provider.entities import ProviderRequest
from agent_runtime.provider.provider import Provider

__all__ = ["FunctionToolExecutor"]


class FunctionToolExecutor(BaseFunctionToolExecutor[TContext]):
    """Default executor: handler/call bispatch + sub-agent handoff (design.md §5.2).

    Constructed with the chat :class:`Provider` (and optional hooks) so that
    :class:`HandoffTool` delegation can spawn a sub-agent runner sharing them.
    """

    def __init__(
        self,
        provider: Provider,
        agent_hooks: BaseAgentRunHooks[TContext] | None = None,
        *,
        max_sub_agent_steps: int = 30,
    ) -> None:
        self.provider = provider
        self.agent_hooks: BaseAgentRunHooks[TContext] = agent_hooks if agent_hooks is not None else BaseAgentRunHooks()
        self.max_sub_agent_steps = max_sub_agent_steps

    async def execute(
        self,
        tool: FunctionTool,
        run_context: ContextWrapper[TContext],
        **tool_args,
    ) -> AsyncGenerator[ToolExecResult, None]:  # type: ignore[override]
        if isinstance(tool, HandoffTool):
            async for result in self._run_handoff(tool, run_context, **tool_args):
                yield result
            return

        # function-style tool -> handler; subclass/MCPTool -> call()
        if tool.handler is not None:
            result = tool.handler(run_context, **tool_args)
        else:
            result = tool.call(run_context, **tool_args)

        if inspect.isasyncgen(result):
            async for item in result:
                yield self._normalize(item)
        else:
            yield self._normalize(await cast("Awaitable[ToolExecResult]", result))

    @staticmethod
    def _normalize(item):
        """Coerce a ToolExecResult (``str | CallToolResult``) into a CallToolResult.

        The runner consumes ``mcp.types.CallToolResult``; the ergonomic ``str`` return
        form allowed by the ``ToolExecResult`` contract is wrapped here so a bare-string
        tool result is not silently dropped by the runner.
        """
        if isinstance(item, mcp.types.CallToolResult):
            return item
        if isinstance(item, str):
            return mcp.types.CallToolResult(content=[mcp.types.TextContent(type="text", text=item)])
        return item

    async def _run_handoff(
        self,
        tool: HandoffTool,
        run_context: ContextWrapper[TContext],
        **tool_args,
    ):
        """Delegate to ``tool.agent``: compile it into a ProviderRequest + sub-runner.

        The sub-agent shares this executor's provider/hooks; per-subagent provider
        override (``tool.provider_id``) is a known limitation (see README).
        """
        sub_agent = tool.agent
        input_ = tool_args.get("input")
        image_urls = list(tool_args.get("image_urls") or [])

        toolset = ToolSet()
        for entry in sub_agent.tools or []:
            if isinstance(entry, FunctionTool):
                toolset.add_tool(entry)

        request = ProviderRequest(
            prompt=input_,
            image_urls=image_urls,
            system_prompt=sub_agent.instructions or "",
            func_tool=toolset if not toolset.empty() else None,
        )

        # Fresh message list for the sub-agent; carry the caller's TContext.
        sub_run_context = ContextWrapper(
            context=run_context.context,
            messages=[],
            tool_call_timeout=run_context.tool_call_timeout,
        )
        for dialog in sub_agent.begin_dialogs or []:
            try:
                sub_run_context.messages.append(
                    dialog if isinstance(dialog, Message) else Message.model_validate(dialog)
                )
            except Exception:  # noqa: BLE001 - skip undecodable seed dialogs
                continue

        sub_runner = ToolLoopAgentRunner()
        await sub_runner.reset(
            provider=self.provider,
            request=request,
            run_context=sub_run_context,
            tool_executor=self,
            agent_hooks=sub_agent.run_hooks or self.agent_hooks,
        )
        async for _ in sub_runner.step_until_done(self.max_sub_agent_steps):
            pass

        final = sub_runner.get_final_llm_resp()
        text = (final.completion_text if final else "") or ""
        yield mcp.types.CallToolResult(content=[mcp.types.TextContent(type="text", text=text)])
