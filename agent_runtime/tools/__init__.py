"""Tool management layer: registry + MCP-server lifecycle orchestration.

This is an outer layer that sits beside :mod:`agent_runtime.provider` and depends only
inward on :mod:`agent_runtime.core` (tool primitives ``FunctionTool`` / ``ToolSet`` /
``MCPTool`` / ``MCPClient``) and :mod:`agent_runtime.foundation`.

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
