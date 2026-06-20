"""aiohttp backend: assemble tools + Bilibili MCP, stream the agent's thinking via SSE.

Composition:

* a process-wide :class:`FunctionToolManager` connects the ``bilibili-search`` MCP
  server (``npx bilibili-mcp-js``) and yields its tools;
* those tools are merged with the local ``calculate`` tool into one ``ToolSet``;
* each user turn drives a :class:`ToolLoopAgentRunner` with ``streaming=True``,
  carrying prior messages forward so the conversation is multi-turn;
* every ``AgentResponse`` the runner yields (reasoning delta, tool_call,
  tool_call_result, answer delta) is serialized to a Server-Sent Event so the
  browser can render the thinking process live.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
from pathlib import Path
from typing import Any

from aiohttp import web

from agent_runtime.core import FunctionToolExecutor, SessionContext
from agent_runtime.core.hooks import BaseAgentRunHooks
from agent_runtime.core.message import Message
from agent_runtime.core.run_context import ContextWrapper
from agent_runtime.core.runners.tool_loop_agent_runner import ToolLoopAgentRunner
from agent_runtime.core.tool import ToolSet
from agent_runtime.provider.entities import ProviderRequest
from agent_runtime.tools.func_tool_manager import FunctionToolManager
from examples.chatbot.calculator import build_calculate_tool
from examples.example_provider import make_openai_compat_provider

# The MCP server the task asks for: Bilibili video search over stdio (npx launcher).
# Defined inline and registered via `enable_mcp_server` — this demo owns its config and
# never writes a `mcp_server.json` into the runtime data dir.
_MCP_SERVER_NAME = "bilibili-search"
_MCP_SERVER_CONFIG = {
    "command": "npx",
    "args": ["bilibili-mcp-js"],
    "description": "B站视频搜索 MCP 服务，可以在AI应用中搜索B站视频内容。",
}

_SYSTEM_PROMPT = (
    "You are a helpful Chinese-speaking assistant with two abilities:\n"
    "1. `calculate` — for ANY arithmetic, call this tool instead of computing in your head.\n"
    "2. Bilibili search tools — when the user wants videos, search Bilibili and present "
    "the title, author/UP主, and link for the top results.\n"
    "Think step by step, call tools when they help, and reply in 中文."
)

_INDEX_HTML = Path(__file__).with_name("index.html")
_MAX_STEPS = 12


def _sse(event: str, payload: dict[str, Any]) -> bytes:
    """Encode a payload as one Server-Sent Event frame.

    Args:
        event: The SSE event name (consumed by the frontend dispatcher).
        payload: JSON-serializable event data.

    Returns:
        The encoded ``event:``/``data:`` frame as bytes.
    """
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n".encode()


async def _build_tools(app: web.Application) -> ToolSet:
    """Connect the Bilibili MCP server and merge its tools with ``calculate``.

    Args:
        app: The aiohttp app, used to stash the MCP manager for later shutdown.

    Returns:
        The combined :class:`ToolSet` handed to every run.
    """
    # enable_mcp_server() takes the per-server config dict directly, so this demo keeps
    # its MCP config inline (above) instead of writing a mcp_server.json into the data dir.
    #
    # The `bilibili-mcp-js` package prints debug objects to stdout (it should use stderr),
    # which violates the stdio JSON-RPC protocol. The MCP client tolerates each bad line
    # (it `continue`s) but logs a full traceback per line, flooding the console. Drop only
    # those parse-failure records via a targeted filter, so genuine errors from this logger
    # are still surfaced. The noise is harmless — tool results still come through.
    mcp_stdio_logger = logging.getLogger("mcp.client.stdio")
    mcp_stdio_logger.addFilter(
        lambda record: "Failed to parse JSONRPC message" not in record.getMessage()
    )

    manager = FunctionToolManager()
    app["mcp_manager"] = manager
    try:
        await manager.enable_mcp_server(_MCP_SERVER_NAME, _MCP_SERVER_CONFIG)
        print(f"[chatbot] MCP connected: {_MCP_SERVER_NAME}.")
    except Exception as exc:  # noqa: BLE001 - degrade gracefully without video search
        # The arithmetic path still works without Bilibili; warn and continue.
        print(
            f"[chatbot] WARNING: Bilibili MCP failed to start ({type(exc).__name__}: {exc}). "
            "Video search will be unavailable; arithmetic still works."
        )

    tools = manager.get_full_tool_set()
    tools.add_tool(build_calculate_tool())
    print(f"[chatbot] Tools available: {sorted(tools.names())}")
    return tools


async def chat_handler(request: web.Request) -> web.StreamResponse:
    """Stream one user turn as SSE: reasoning, tool calls/results, and the answer.

    Args:
        request: A POST carrying ``{"message": str, "session_id": str}``.

    Returns:
        A streaming SSE response driven by the agent's tool loop.
    """
    body = await request.json()
    user_message = str(body.get("message", "")).strip()
    session_id = str(body.get("session_id") or "default")
    if not user_message:
        return web.json_response({"error": "empty message"}, status=400)

    provider = request.app["provider"]
    tools: ToolSet = request.app["tools"]
    # Per-session message history: keeps the conversation multi-turn across requests.
    histories: dict[str, list[Message]] = request.app["histories"]
    prior_messages = histories.get(session_id, [])

    response = web.StreamResponse(
        headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )
    await response.prepare(request)

    request_obj = ProviderRequest(
        prompt=user_message,
        system_prompt=_SYSTEM_PROMPT if not prior_messages else "",
        func_tool=tools,
        session_id=session_id,
        contexts=[m.model_dump() for m in prior_messages],
    )
    run_context: ContextWrapper[SessionContext] = ContextWrapper(
        context=SessionContext(session_id=session_id), messages=[]
    )
    runner = ToolLoopAgentRunner()
    await runner.reset(
        provider=provider,
        request=request_obj,
        run_context=run_context,
        tool_executor=FunctionToolExecutor(provider),
        agent_hooks=BaseAgentRunHooks(),
        streaming=True,
    )

    try:
        async for resp in runner.step_until_done(max_step=_MAX_STEPS):
            chain = resp.data["chain"]
            if resp.type == "streaming_delta":
                # Reasoning vs answer text are distinguished by the chain type.
                kind = "reasoning" if chain.type == "reasoning" else "answer"
                text = chain.get_plain_text()
                if text:
                    await response.write(_sse("delta", {"kind": kind, "text": text}))
            elif resp.type == "tool_call":
                # chain carries one Json component: {id, name, args, ts}.
                await response.write(_sse("tool_call", chain.chain[0].data))
            elif resp.type == "tool_call_result":
                await response.write(_sse("tool_result", chain.chain[0].data))
            elif resp.type == "err":
                await response.write(_sse("error", {"message": chain.get_plain_text()}))
    except Exception as exc:  # noqa: BLE001 - surface any run failure to the client
        await response.write(_sse("error", {"message": f"{type(exc).__name__}: {exc}"}))
    else:
        # Persist the full turn (prior history + this turn's new messages) for next time.
        histories[session_id] = run_context.messages
        final = runner.get_final_llm_resp()
        await response.write(_sse("done", {"text": (final.completion_text if final else "") or ""}))

    await response.write_eof()
    return response


async def index_handler(_: web.Request) -> web.Response:
    """Serve the single-page frontend."""
    return web.Response(text=_INDEX_HTML.read_text(encoding="utf-8"), content_type="text/html")


async def _on_startup(app: web.Application) -> None:
    """Build provider + tools once when the server starts."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Run with: uv run --env-file .env python -m examples.chatbot")
    app["provider"] = make_openai_compat_provider(
        api_key=api_key,
        model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        api_base=os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1"),
    )
    app["tools"] = await _build_tools(app)
    app["histories"] = {}


async def _on_cleanup(app: web.Application) -> None:
    """Tear down every MCP connection on shutdown."""
    manager = app.get("mcp_manager")
    if manager is not None:
        with contextlib.suppress(Exception):
            await manager.disable_mcp_server()


def main() -> None:
    """Build the app and run the server on 127.0.0.1:8000."""
    app = web.Application()
    app.add_routes(
        [
            web.get("/", index_handler),
            web.post("/api/chat", chat_handler),
        ]
    )
    app.on_startup.append(_on_startup)
    app.on_cleanup.append(_on_cleanup)
    print("[chatbot] Open http://127.0.0.1:8000")
    web.run_app(app, host="127.0.0.1", port=8000, print=None)


if __name__ == "__main__":
    with contextlib.suppress(KeyboardInterrupt):
        main()
