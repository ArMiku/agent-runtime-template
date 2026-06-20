"""End-to-end example: an MCP tool driving the runtime.

This is the MCP counterpart of ``local_runtime_demo``. It proves the full
**装配 → 执行 → 关闭** loop for MCP tools, which is otherwise not wired into
``build_local_agent``:

* a tiny **stdio MCP echo server** is generated into a temp data dir and launched
  as a subprocess (no external MCP server required);
* ``FunctionToolManager.init_mcp_clients()`` reads ``mcp_server.json`` from that
  data dir, connects, and turns the server's tools into ``MCPTool`` instances;
* those tools are merged into the runner's ``ToolSet``; a scripted provider asks
  the model to call ``echo``; the executor dispatches the ``MCPTool`` (``call()``
  → ``MCPClient.call_tool_with_reconnect`` → the subprocess);
* finally ``disable_mcp_server()`` tears every connection down.

Run it::

    python -m examples.mcp_demo
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import textwrap

from agent_runtime.core import FunctionToolExecutor, SessionContext
from agent_runtime.core.hooks import BaseAgentRunHooks
from agent_runtime.core.run_context import ContextWrapper
from agent_runtime.core.runners.tool_loop_agent_runner import ToolLoopAgentRunner
from agent_runtime.provider.entities import LLMResponse, ProviderRequest
from agent_runtime.provider.provider import Provider
from agent_runtime.tools.func_tool_manager import FunctionToolManager

# A minimal FastMCP stdio server exposing one tool: ``echo(text) -> "echo: {text}"``.
# Launched via ``sys.executable`` so the child process shares this venv (and the
# ``mcp`` package). ``python`` is on the stdio command allowlist, and the file
# argument passes ``validate_mcp_stdio_config`` (no shell metachars, no inline code).
_ECHO_SERVER_SRC = textwrap.dedent(
    """\
    from mcp.server.fastmcp import FastMCP

    app = FastMCP("echo-server")


    @app.tool()
    def echo(text: str) -> str:
        \"\"\"Echo the given text back with a prefix.\"\"\"
        return f"echo: {text}"


    if __name__ == "__main__":
        app.run()
    """,
)

_ECHO_PROMPT = "hello mcp"


class _ScriptedProvider(Provider):
    """Returns scripted replies: first a tool call to ``echo``, then a final line."""

    def __init__(self) -> None:
        super().__init__({"id": "demo", "type": "demo", "max_context_tokens": 0, "modalities": []}, {})

    def get_current_key(self) -> str:
        return "demo-key"

    def set_key(self, key: str) -> None: ...

    async def get_models(self) -> list[str]:
        return ["demo-model"]

    async def text_chat(self, *args, **kwargs) -> LLMResponse:
        return LLMResponse(
            role="assistant",
            tools_call_name=["echo"],
            tools_call_args=[{"text": _ECHO_PROMPT}],
            tools_call_ids=["c1"],
        )

    async def text_chat_stream(self, *args, **kwargs):
        yield await self.text_chat(*args, **kwargs)


def _seed_echo_server(data_dir: str) -> str:
    """Write the echo MCP server into the data dir; return its path."""
    server_path = os.path.join(data_dir, "echo_server.py")
    with open(server_path, "w", encoding="utf-8") as f:
        f.write(_ECHO_SERVER_SRC)
    return server_path


def _seed_mcp_config(data_dir: str, server_path: str) -> None:
    """Write ``mcp_server.json`` pointing the ``echo`` server at the subprocess."""
    config = {
        "mcpServers": {
            "echo": {
                "command": sys.executable,
                "args": [server_path],
            }
        }
    }
    import json

    with open(os.path.join(data_dir, "mcp_server.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=4)


async def main() -> dict:
    data_dir = tempfile.mkdtemp(prefix="mcp_demo_")
    os.environ["AGENT_RUNTIME_DATA_DIR"] = data_dir

    server_path = _seed_echo_server(data_dir)
    _seed_mcp_config(data_dir, server_path)

    # 1. 装配：读 mcp_server.json → 连接 stdio 服务 → 产出 MCPTool。
    manager = FunctionToolManager()
    summary = await manager.init_mcp_clients(raise_on_all_failed=True)
    assert summary.success == 1 and summary.failed == [], summary

    tools = manager.get_full_tool_set()
    assert "echo" in tools.names(), tools.names()

    # 2. 执行：scripted provider 触发 echo 工具调用，executor 分派 MCPTool。
    provider = _ScriptedProvider()
    request = ProviderRequest(prompt=_ECHO_PROMPT, system_prompt="", func_tool=tools)
    run_context = ContextWrapper(context=SessionContext(session_id="mcp-demo"), messages=[])

    runner = ToolLoopAgentRunner()
    await runner.reset(
        provider=provider,
        request=request,
        run_context=run_context,
        tool_executor=FunctionToolExecutor(provider),
        agent_hooks=BaseAgentRunHooks(),
    )
    async for _ in runner.step_until_done(max_step=5):
        pass

    tool_results = [str(getattr(m, "content", "")) for m in run_context.messages if getattr(m, "role", None) == "tool"]
    echo_ran = any(_ECHO_PROMPT in t for t in tool_results)

    # 3. 关闭：停掉全部 MCP 连接（公共入口，无需碰私有 _shutdown_runtimes）。
    await manager.disable_mcp_server()

    result = {
        "mcp_summary": {"total": summary.total, "success": summary.success, "failed": summary.failed},
        "tool_names": sorted(tools.names()),
        "echo_tool_ran": echo_ran,
    }
    print(result)
    assert echo_ran, f"echo MCP tool did not run; messages: {run_context.messages}"
    return result


if __name__ == "__main__":
    asyncio.run(main())
