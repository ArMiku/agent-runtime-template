"""Plan persistence over the ``PluginStore`` seam: round-trip, full-replacement, and
session isolation (tasks 3.3, 7.x)."""

from __future__ import annotations

from agent_runtime.extensions.planning.entities import Todo, TodoStatus
from agent_runtime.extensions.planning.store import load_plan, save_plan
from agent_runtime.extensions.plugins.store import InMemoryPluginStore


async def test_round_trip():
    store = InMemoryPluginStore()
    plan = [
        Todo(content="a", status=TodoStatus.COMPLETED),
        Todo(content="b", status=TodoStatus.IN_PROGRESS),
    ]
    await save_plan(store, "s1", plan)

    loaded = await load_plan(store, "s1")
    assert loaded == plan


async def test_empty_when_absent():
    store = InMemoryPluginStore()
    assert await load_plan(store, "never-written") == []


async def test_full_replacement_keeps_only_latest():
    store = InMemoryPluginStore()
    await save_plan(store, "s1", [Todo(content="old")])
    await save_plan(store, "s1", [Todo(content="new1"), Todo(content="new2")])

    loaded = await load_plan(store, "s1")
    assert [t.content for t in loaded] == ["new1", "new2"]


async def test_sessions_are_isolated():
    store = InMemoryPluginStore()
    await save_plan(store, "s1", [Todo(content="for-s1")])
    await save_plan(store, "s2", [Todo(content="for-s2")])

    assert [t.content for t in await load_plan(store, "s1")] == ["for-s1"]
    assert [t.content for t in await load_plan(store, "s2")] == ["for-s2"]


async def test_cross_process_recovery_with_shared_store():
    """7.1: a persistent store shared across 'processes' recovers the plan by session_id.

    InMemoryPluginStore stands in for a host-injected persistent store: the second reader
    holds the same instance (as a real persistent backend would), so the plan survives."""
    store = InMemoryPluginStore()
    await save_plan(store, "sess", [Todo(content="step", status=TodoStatus.IN_PROGRESS)])

    # New "process" with the same backing store and session id.
    recovered = await load_plan(store, "sess")
    assert recovered == [Todo(content="step", status=TodoStatus.IN_PROGRESS)]


async def test_malformed_record_falls_back_gracefully():
    """A non-list or dict-with-bad-status never breaks loading."""
    store = InMemoryPluginStore()
    await store.put("__planning__", "s1", "not-a-list")
    assert await load_plan(store, "s1") == []

    await store.put("__planning__", "s2", [{"content": "x", "status": "bogus"}])
    loaded = await load_plan(store, "s2")
    assert loaded == [Todo(content="x", status=TodoStatus.PENDING)]
