"""aiohttp backend: assemble tools + Bilibili MCP, stream the agent's thinking via SSE.

Composition:

* a process-wide :class:`FunctionToolManager` connects the ``bilibili-search`` MCP
  server (``npx bilibili-mcp-js``) and yields its tools;
* those tools are merged with the local ``calculate`` tool into one ``ToolSet``;
* a :class:`SkillManager` discovers ``SKILL.md`` bundles from ``data/skills/`` and
  registers the ``Skill`` tool so the agent can load skill instructions on demand;
* when planning is enabled (``CHATBOT_PLANNING`` env, default on), the ``write_todos``
  tool joins that set and each turn runs under a :class:`PlanningHook` — the agent can
  keep a todo plan for multi-step asks, the plan is injected into the system message each
  step, and a premature finish (unfinished todos) is vetoed;
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
from agent_runtime.core.runners.plan_execute_runner import PlanExecuteRunner
from agent_runtime.core.runners.tool_loop_agent_runner import ToolLoopAgentRunner
from agent_runtime.core.tool import ToolSet
from agent_runtime.extensions.planning import PlanningHook, build_write_todos_tool
from agent_runtime.extensions.planning.store import PLANNING_PLUGIN_ID
from agent_runtime.extensions.plugins.store import InMemoryPluginStore
from agent_runtime.extensions.skills import SkillManager, SkillsPromptHook, build_skill_tool
from agent_runtime.provider.entities import ProviderRequest
from agent_runtime.tools.func_tool_manager import FunctionToolManager
from examples.chatbot.calculator import build_calculate_tool
from examples.chatbot.web_search import build_web_search_tool
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
    "You are a helpful Chinese-speaking assistant with several abilities:\n"
    "1. `calculate` — for ANY arithmetic, call this tool instead of computing in your head.\n"
    "2. `web_search` — for anything current or factual you don't already know (news, recent "
    "events, docs, prices, people), search the live web instead of guessing.\n"
    "3. Bilibili search tools — when the user wants videos, search Bilibili and present "
    "the title, author/UP主, and link for the top results.\n"
    "For an open-ended, multi-step request, first lay out a todo plan with `write_todos` "
    "(if that tool is available), then work through it, marking items completed as you go.\n"
    "Think step by step, call tools when they help, and reply in 中文."
)

# Planning is on by default; set CHATBOT_PLANNING=0 to run the plain ReAct chatbot.
_PLANNING_ENABLED = os.environ.get("CHATBOT_PLANNING", "1") not in ("0", "false", "False", "")

_INDEX_HTML = Path(__file__).with_name("index.html")
# ReAct path step budget (route 2). Kept modest — that loop self-terminates normally.
_MAX_STEPS = 12

# Route-1 (/plan) liveness ceilings. DISASTER-LOOSE by design: high enough that a normal task
# never touches them, so they only cut a *runaway* short and report it honestly — convergence
# is the planner's job, not these guards'. A tight value here would re-create the very "guess
# the task's shape" misfire they exist to avoid. All three are env-overridable.
_PLAN_MAX_STEPS = int(os.environ.get("CHATBOT_PLAN_MAX_STEPS", "200"))
_PLAN_CALL_TIMEOUT_S = float(os.environ.get("CHATBOT_PLAN_CALL_TIMEOUT", "150"))
_PLAN_TURN_DEADLINE_S = float(os.environ.get("CHATBOT_PLAN_TURN_DEADLINE", "1500"))

# Typing "/plan <task>" routes that turn through the explicit PlanExecuteRunner (route 1,
# PLAN → EXEC → REPLAN) instead of the default ReAct + PlanningHook loop (route 2).
_PLAN_PREFIX = "/plan"


def _parse_plan_command(message: str) -> tuple[bool, str]:
    """Detect the ``/plan`` trigger.

    Returns ``(use_plan_execute, effective_task)``: the prefix is stripped so the planner only
    sees the actual task. A bare ``/plan`` with no task returns an empty string so the caller
    can reject it.
    """
    if message.startswith(_PLAN_PREFIX):
        return True, message[len(_PLAN_PREFIX):].lstrip()
    return False, message


def _looks_like_plan(text: str) -> bool:
    """Heuristic: a PLAN/REPLAN response renders the plan as a checklist (or the empty marker)."""
    stripped = text.lstrip()
    return stripped.startswith(("- [ ]", "- [x]", "- [~]")) or stripped == "_No plan._"


def _env_json(name: str, default: Any) -> Any:
    """Parse a JSON env var, returning ``default`` when unset or malformed."""
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        print(f"[chatbot] WARNING: {name} is not valid JSON, ignored.")
        return default


def _plan_execute_sub_hook_factory() -> Any:
    """Child-hook factory for the chatbot's PlanExecuteRunner.

    The chatbot has no SkillManager (it uses raw tools, not skills), and route-1 planning is
    driven by the state machine — so each executor child gets a plain no-op hook chain. The
    important property is isolation: the child never sees the main hook chain.
    """

    def factory(sub_session_id: str) -> BaseAgentRunHooks:  # noqa: ARG001
        return BaseAgentRunHooks()

    return factory


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
    mcp_stdio_logger.addFilter(lambda record: "Failed to parse JSONRPC message" not in record.getMessage())

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
    # web_search degrades to "absent" when tavily-python is missing or TAVILY_API_KEY
    # is unset — the rest of the chatbot (arithmetic, Bilibili, planning) still works.
    web_search_tool = build_web_search_tool()
    if web_search_tool is not None:
        tools.add_tool(web_search_tool)
    # Planning: the write_todos tool joins the shared set (it reads session_id from
    # run_context at call time, so one instance serves every session). The plan itself
    # lives in an app-level store so it persists across this session's turns; the
    # per-turn PlanningHook (in chat_handler) reads/injects it and vetoes premature finish.
    if _PLANNING_ENABLED:
        store = InMemoryPluginStore()
        app["plan_store"] = store
        tools.add_tool(build_write_todos_tool(store))
    # Skills: discover SKILL.md bundles from data/skills/ and register the Skill tool.
    skill_manager = SkillManager()
    app["skill_manager"] = skill_manager
    tools.add_tool(build_skill_tool(skill_manager))
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

    # "/plan <task>" routes this turn through the explicit PlanExecuteRunner (route 1).
    use_plan_execute, effective_message = _parse_plan_command(user_message)
    if use_plan_execute and not effective_message:
        return web.json_response({"error": "empty task after /plan"}, status=400)

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

    if use_plan_execute:
        # Route 1: explicit PLAN → EXEC → REPLAN. Each /plan turn plans fresh (it clears any
        # recovered progress so a second /plan doesn't resume the first). The turn is
        # self-contained, so it does not feed back into the regular chat history.
        await _stream_plan_execute(response, request, provider, tools, effective_message, session_id)
        await response.write_eof()
        return response

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
    # Planning hook is per-turn (fresh reminder counter each turn) but points at the
    # app-level store, so the plan persists across this session's turns. Without planning,
    # fall back to the default no-op hooks — identical to the plain ReAct chatbot.
    # SkillsPromptHook wraps whichever inner hook is active so the skill inventory is
    # injected into the system message every step.
    plan_store = request.app.get("plan_store")
    inner_hooks = PlanningHook(plan_store) if plan_store is not None else BaseAgentRunHooks()
    skill_manager: SkillManager | None = request.app.get("skill_manager")
    # When both skills and planning are active, chain them: SkillsPromptHook handles
    # on_llm_request (inventory), PlanningHook handles on_llm_request (plan) +
    # on_before_complete (veto). Use a lightweight composite to dispatch to both.
    if skill_manager and plan_store is not None:

        class _CompositeHooks(BaseAgentRunHooks):
            """Dispatch on_llm_request to both SkillsPromptHook and PlanningHook."""

            def __init__(self, skills_hook, planning_hook):
                self._skills = skills_hook
                self._planning = planning_hook

            async def on_llm_request(self, run_context):
                await self._skills.on_llm_request(run_context)
                await self._planning.on_llm_request(run_context)

            async def on_before_complete(self, run_context, llm_response):
                return await self._planning.on_before_complete(run_context, llm_response)

        agent_hooks = _CompositeHooks(SkillsPromptHook(skill_manager), inner_hooks)
    elif skill_manager:
        agent_hooks = SkillsPromptHook(skill_manager)
    else:
        agent_hooks = inner_hooks
    runner = ToolLoopAgentRunner()
    await runner.reset(
        provider=provider,
        request=request_obj,
        run_context=run_context,
        tool_executor=FunctionToolExecutor(provider),
        agent_hooks=agent_hooks,
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


async def _stream_plan_execute(
    response: web.StreamResponse,
    request: web.Request,
    provider: Any,
    tools: ToolSet,
    task: str,
    session_id: str,
) -> None:
    """Drive a ``PlanExecuteRunner`` for one ``/plan`` turn and stream it as SSE.

    Emits the same ``delta``/``tool_call``/``tool_result``/``error`` events as the ReAct path
    (EXEC forwards the child runner's responses), plus a ``plan`` event whenever PLAN/REPLAN
    produces a revised plan checklist. The final assembled summary is sent as ``done``.
    """
    # Dedicated store for route-1 state (phase snapshot + plan mirror), isolated from route 2's
    # write_todos store so the two never collide on the same session key.
    store = request.app.get("plan_execute_store")
    if store is None:
        store = InMemoryPluginStore()
        request.app["plan_execute_store"] = store
    # Plan fresh each turn: clear any recovered progress from a prior /plan on this session.
    # Key mirrors PlanExecuteRunner's phase key (PLANNING_PLUGIN_ID, "<session>__phase").
    await store.delete(PLANNING_PLUGIN_ID, f"{session_id}__phase")

    # Executor tools: route 1 plans via submit_plan, so drop the emergent write_todos tool to
    # keep the executor focused on calculate / Bilibili search.
    executor_tools = ToolSet([t for t in tools.tools if t.name != "write_todos"])

    pe_request = ProviderRequest(
        prompt=task,
        system_prompt=_SYSTEM_PROMPT,
        func_tool=executor_tools,
        session_id=session_id,
    )
    pe_run_context: ContextWrapper[SessionContext] = ContextWrapper(
        context=SessionContext(session_id=session_id), messages=[]
    )
    runner = PlanExecuteRunner()
    # The runner ships no token budget by default (model context windows differ widely, so the
    # budget is the host's call). This demo exposes the two planner levers as env vars — only
    # set them if the model truncates the planner before it emits submit_plan.
    _planner_max_tokens_env = os.environ.get("CHATBOT_PLANNER_MAX_TOKENS")
    await runner.reset(
        provider=provider,
        request=pe_request,
        run_context=pe_run_context,
        tool_executor=FunctionToolExecutor(provider),
        agent_hooks=BaseAgentRunHooks(),
        plugin_store=store,
        tool_set=executor_tools,
        sub_hook_factory=_plan_execute_sub_hook_factory(),
        streaming=True,
        enforce_max_turns=_PLAN_MAX_STEPS,
        planner_max_tokens=int(_planner_max_tokens_env) if _planner_max_tokens_env else None,
        planner_extra_body=_env_json("CHATBOT_PLANNER_EXTRA_BODY", {}) or None,
        # Liveness ceilings (all disaster-loose, env-overridable). They live inside step() so a
        # runaway is cut short and reported honestly instead of hanging or sending an empty done.
        max_step=_PLAN_MAX_STEPS,
        per_call_timeout_s=_PLAN_CALL_TIMEOUT_S,
        per_turn_deadline_s=_PLAN_TURN_DEADLINE_S,
    )

    try:
        async for resp in runner.step_until_done(max_step=_PLAN_MAX_STEPS):
            chain = resp.data["chain"]
            if resp.type == "streaming_delta":
                kind = "reasoning" if chain.type == "reasoning" else "answer"
                text = chain.get_plain_text()
                if text:
                    await response.write(_sse("delta", {"kind": kind, "text": text}))
            elif resp.type == "tool_call":
                await response.write(_sse("tool_call", chain.chain[0].data))
            elif resp.type == "tool_call_result":
                await response.write(_sse("tool_result", chain.chain[0].data))
            elif resp.type == "llm_result":
                # PLAN/REPLAN render the plan as a checklist; EXEC's per-todo answer is plain text.
                text = chain.get_plain_text()
                if _looks_like_plan(text):
                    await response.write(_sse("plan", {"text": text}))
                elif text:
                    await response.write(_sse("delta", {"kind": "answer", "text": text}))
            elif resp.type == "err":
                await response.write(_sse("error", {"message": chain.get_plain_text()}))
    except Exception as exc:  # noqa: BLE001 - surface any run failure to the client
        await response.write(_sse("error", {"message": f"{type(exc).__name__}: {exc}"}))
    else:
        final = runner.get_final_llm_resp()
        await response.write(_sse("done", {"text": (final.completion_text if final else "") or ""}))


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
