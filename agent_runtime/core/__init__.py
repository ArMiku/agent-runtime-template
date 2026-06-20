"""Agent engine: the tool-loop runner, tools, hooks, and context management.

The seam *abstractions* and their *default implementations* live side by side here,
grouped by domain rather than by role:

* :class:`~agent_runtime.core.tool_executor.BaseFunctionToolExecutor` (abstract seam) and
  :class:`~agent_runtime.core.function_tool_executor.FunctionToolExecutor` (default impl
  with sub-agent handoff).
* :class:`~agent_runtime.core.run_context.ContextWrapper` / ``TContext`` (abstract) and
  :class:`~agent_runtime.core.session_context.SessionContext` (default ``TContext``).

The persistence seam (:class:`~agent_runtime.core.context.context_store.ContextStore`
protocol + ``InMemoryContextStore`` default) lives under :mod:`agent_runtime.core.context`.
"""

from __future__ import annotations

from agent_runtime.core.function_tool_executor import FunctionToolExecutor
from agent_runtime.core.session_context import SessionContext

__all__ = ["FunctionToolExecutor", "SessionContext"]
