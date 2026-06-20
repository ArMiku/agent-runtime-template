"""Phase 4 + 4b tests: PluginManager registration/lifecycle + session enable/disable."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from agent_runtime.extensions.plugins.base import Plugin
from agent_runtime.extensions.plugins.decorators import on_llm_request, tool
from agent_runtime.extensions.plugins.manager import (
    MetadataValidationError,
    PluginManager,
)
from agent_runtime.extensions.plugins.metadata import PluginMetadata
from agent_runtime.extensions.plugins.registry import PluginRegistry
from agent_runtime.extensions.plugins.session import is_plugin_enabled_for_session
from agent_runtime.extensions.plugins.store import InMemoryPluginStore


def _manager() -> PluginManager:
    # Fresh registry so cross-test class auto-registration does not leak.
    return PluginManager(registry=PluginRegistry(), plugin_store=InMemoryPluginStore())


async def test_register_instantiates_initializes_and_collects() -> None:
    events: list[str] = []

    class _P(Plugin):
        name, author, desc, version = "p1", "rt", "demo", "0.1.0"

        async def initialize(self) -> None:
            events.append("init")

        @tool
        async def echo(self, run_context, text: str) -> str:
            """Echo.

            Args:
                text(string): t
            """
            return text

    manager = _manager()
    contribution = await manager.register(_P)

    assert events == ["init"]
    assert [t.name for t in contribution.tools] == ["echo"]
    # Instance recorded in the registry.
    assert manager.registry.get_instance("p1") is contribution.plugin


async def test_class_attribute_metadata_or_explicit_metadata() -> None:
    class _NoAttrs(Plugin): ...

    manager = _manager()
    md = PluginMetadata(name="explicit", author="a", desc="d", version="1")
    contribution = await manager.register(_NoAttrs, metadata=md)
    assert contribution.plugin.metadata.name == "explicit"


async def test_incomplete_metadata_rejected() -> None:
    class _Bad(Plugin):
        name = "only-name"  # missing author/desc/version

    manager = _manager()
    with pytest.raises(MetadataValidationError, match="author"):
        await manager.register(_Bad)


async def test_unload_calls_terminate_and_drops_contribution() -> None:
    events: list[str] = []

    class _P(Plugin):
        name, author, desc, version = "p2", "rt", "d", "1"

        async def terminate(self) -> None:
            events.append("term")

    manager = _manager()
    await manager.register(_P)
    assert len(manager.contributions) == 1
    await manager.unload("p2")
    assert events == ["term"]
    assert manager.contributions == []
    assert manager.registry.get_instance("p2") is None


async def test_reload_terminates_then_reinitializes() -> None:
    events: list[str] = []

    class _P(Plugin):
        name, author, desc, version = "p3", "rt", "d", "1"

        async def initialize(self) -> None:
            events.append("init")

        async def terminate(self) -> None:
            events.append("term")

    manager = _manager()
    await manager.register(_P)
    await manager.reload("p3")
    assert events == ["init", "term", "init"]
    assert len(manager.contributions) == 1


async def test_subclass_auto_registers_to_registry() -> None:
    class _Auto(Plugin):
        name, author, desc, version = "auto", "a", "d", "1"

    # Auto-registration goes to the *default* registry; verify the mechanism there.
    assert _Auto in PluginRegistry.default().classes()


async def test_optional_directory_load(tmp_path: Path) -> None:
    plugin_dir = tmp_path / "myplugin"
    plugin_dir.mkdir()
    (plugin_dir / "main.py").write_text(
        textwrap.dedent(
            '''
            from agent_runtime.extensions.plugins.base import Plugin
            from agent_runtime.extensions.plugins.decorators import tool


            class MyDirPlugin(Plugin):
                name, author, desc, version = "dir-plugin", "rt", "d", "1"

                @tool
                async def hello(self, run_context, who: str) -> str:
                    """Say hi.

                    Args:
                        who(string): name
                    """
                    return f"hi {who}"
            '''
        )
    )

    manager = _manager()
    contributions = await manager.load_from_directory(tmp_path)
    names = {c.plugin.metadata.name for c in contributions}
    assert "dir-plugin" in names


# —— Phase 4b: session enable/disable ——
async def test_session_disable_returns_false() -> None:
    store = InMemoryPluginStore()
    from agent_runtime.extensions.plugins.session import (
        SESSION_CONFIG_KEY,
        SESSION_CONFIG_PLUGIN_ID,
    )

    await store.put(
        SESSION_CONFIG_PLUGIN_ID,
        SESSION_CONFIG_KEY,
        {"sess-1": {"disabled_plugins": ["p"]}},
    )
    assert await is_plugin_enabled_for_session(store, "sess-1", "p") is False


async def test_session_default_enabled() -> None:
    store = InMemoryPluginStore()
    assert await is_plugin_enabled_for_session(store, "sess-x", "anything") is True


async def test_session_isolated() -> None:
    store = InMemoryPluginStore()
    from agent_runtime.extensions.plugins.session import (
        SESSION_CONFIG_KEY,
        SESSION_CONFIG_PLUGIN_ID,
    )

    await store.put(
        SESSION_CONFIG_PLUGIN_ID,
        SESSION_CONFIG_KEY,
        {"sess-1": {"disabled_plugins": ["p"]}},
    )
    # Disabled in sess-1, but default-enabled in sess-2.
    assert await is_plugin_enabled_for_session(store, "sess-1", "p") is False
    assert await is_plugin_enabled_for_session(store, "sess-2", "p") is True
