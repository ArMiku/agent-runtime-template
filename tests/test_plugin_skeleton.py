"""Phase 1 tests: plugin skeleton (metadata / store / context / registry / base)."""

from __future__ import annotations

import pytest

from agent_runtime.extensions.plugins.base import Plugin
from agent_runtime.extensions.plugins.context import PluginContext
from agent_runtime.extensions.plugins.metadata import PluginMetadata
from agent_runtime.extensions.plugins.registry import PluginRegistry
from agent_runtime.extensions.plugins.store import InMemoryPluginStore


def _ctx(name: str = "p", store: InMemoryPluginStore | None = None) -> PluginContext:
    return PluginContext(
        metadata=PluginMetadata(name=name, author="a", desc="d", version="1"),
        plugin_store=store,
    )


def test_subclass_auto_registers_class() -> None:
    registry = PluginRegistry.default()

    class _AutoReg(Plugin):
        name = "auto-reg"

    assert _AutoReg in registry.classes()
    assert registry.get_class(_AutoReg.__module__) is _AutoReg


async def test_lifecycle_callable() -> None:
    events: list[str] = []

    class _LC(Plugin):
        async def initialize(self) -> None:
            events.append("init")

        async def terminate(self) -> None:
            events.append("term")

    plugin = _LC(_ctx())
    await plugin.initialize()
    await plugin.terminate()
    assert events == ["init", "term"]


async def test_kv_roundtrip_via_store() -> None:
    store = InMemoryPluginStore()

    class _KV(Plugin): ...

    plugin = _KV(_ctx(name="kv-plugin", store=store))
    await plugin.put_kv_data("count", 7)
    assert await plugin.get_kv_data("count") == 7
    assert await plugin.get_kv_data("missing", default=0) == 0
    await plugin.delete_kv_data("count")
    assert await plugin.get_kv_data("count", default=-1) == -1


async def test_kv_isolated_by_plugin_id() -> None:
    store = InMemoryPluginStore()

    class _A(Plugin): ...

    class _B(Plugin): ...

    a = _A(_ctx(name="plugin-a", store=store))
    b = _B(_ctx(name="plugin-b", store=store))
    await a.put_kv_data("k", "from-a")
    await b.put_kv_data("k", "from-b")
    assert await a.get_kv_data("k") == "from-a"
    assert await b.get_kv_data("k") == "from-b"


async def test_kv_without_store_raises() -> None:
    class _NoStore(Plugin): ...

    plugin = _NoStore(_ctx(name="no-store", store=None))
    with pytest.raises(RuntimeError, match="plugin_store is not set"):
        await plugin.put_kv_data("k", 1)


def test_metadata_missing_required_fields() -> None:
    md = PluginMetadata(name="", author="a", desc="", version="1")
    missing = md.missing_required_fields()
    assert "name" in missing and "desc" in missing
    assert "author" not in missing and "version" not in missing


async def test_conversation_history_without_store_returns_empty() -> None:
    ctx = _ctx()
    assert await ctx.conversation_history("session-1") == []
