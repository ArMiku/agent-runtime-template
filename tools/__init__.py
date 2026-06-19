"""Tool management layer: registry + MCP-server lifecycle orchestration.

This is an outer layer that sits beside :mod:`agent_runtime.provider` and depends only
inward on :mod:`agent_runtime.core` (tool primitives ``FunctionTool`` / ``ToolSet`` /
``MCPTool`` / ``MCPClient``) and :mod:`agent_runtime.foundation`. It mirrors AstrBot's
split between ``core/agent`` (tool *primitives* + engine seams) and ``core/tools`` (the
tool *management* layer): the primitives stay in ``core`` because the runner and tool
executor import them, while registration/MCP orchestration lives out here.

``core`` MUST NOT import this package; nothing in the engine depends on the registry —
it is wired in by the host (or the composition root).
"""

from __future__ import annotations

from agent_runtime.tools.func_tool_manager import (
    FuncCall,
    FunctionToolManager,
    llm_tools,
)

__all__ = ["FuncCall", "FunctionToolManager", "llm_tools"]
