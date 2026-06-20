"""The ``write_todos`` tool persists the plan to the session's independent state and
returns a rendered confirmation (tasks 4.2, 4.3)."""

from __future__ import annotations

from agent_runtime.core.run_context import ContextWrapper
from agent_runtime.core.session_context import SessionContext
from agent_runtime.extensions.planning.store import load_plan
from agent_runtime.extensions.planning.todo_tool import build_write_todos_tool
from agent_runtime.extensions.plugins.store import InMemoryPluginStore


def _ctx(session_id: str = "s1") -> ContextWrapper[SessionContext]:
    return ContextWrapper(context=SessionContext(session_id=session_id), messages=[])


async def test_tool_writes_plan_to_store():
    store = InMemoryPluginStore()
    tool = build_write_todos_tool(store)

    result = await tool.handler(
        _ctx(),
        todos=[
            {"content": "research", "status": "in_progress"},
            {"content": "write up", "status": "pending"},
        ],
    )

    plan = await load_plan(store, "s1")
    assert [t.content for t in plan] == ["research", "write up"]
    # The tool returns a rendered confirmation of the current plan.
    assert "research" in result and "write up" in result


async def test_tool_full_replacement():
    store = InMemoryPluginStore()
    tool = build_write_todos_tool(store)

    await tool.handler(_ctx(), todos=[{"content": "first", "status": "pending"}])
    await tool.handler(_ctx(), todos=[{"content": "second", "status": "pending"}])

    plan = await load_plan(store, "s1")
    assert [t.content for t in plan] == ["second"]


async def test_tool_schema_shape():
    store = InMemoryPluginStore()
    tool = build_write_todos_tool(store)
    assert tool.name == "write_todos"
    props = tool.parameters["properties"]
    assert "todos" in props
    assert props["todos"]["type"] == "array"
    item_props = props["todos"]["items"]["properties"]
    assert set(item_props) == {"content", "status"}
    assert set(item_props["status"]["enum"]) == {"pending", "in_progress", "completed"}
