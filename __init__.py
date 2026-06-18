"""agent_runtime — a framework-agnostic Agent runtime + LLM provider template.

Drivable in ~50 lines: a :class:`~agent_runtime.provider.provider.Provider`, a tool set,
a system prompt, and a
:class:`~agent_runtime.core.runners.tool_loop_agent_runner.ToolLoopAgentRunner`.

This top-level module intentionally stays import-light to avoid pulling the whole runtime
on ``import agent_runtime``. Import submodules explicitly, e.g.::

    from agent_runtime.core.runners.tool_loop_agent_runner import ToolLoopAgentRunner

The four explicit seams (Provider, BaseFunctionToolExecutor, BaseAgentRunHooks,
``TContext`` via :class:`~agent_runtime.core.run_context.ContextWrapper`) are documented
in the README and design.md.
"""

from __future__ import annotations

__version__ = "0.1.0"

# Re-export the logging shim so ``from agent_runtime import logger`` works package-wide.
from .foundation.log import logger

__all__ = ["__version__", "logger"]
