"""The ``Plugin`` base class.

Subclasses auto-register on definition (via ``__init_subclass__``) and receive a neutral
``PluginContext`` at construction time — no god-object, no ``event``, no ``html_render``
helpers.

Identity is declared either as class attributes (``name`` / ``author`` / ``desc`` /
``version``) — collected into ``PluginMetadata`` at registration — or supplied as a
``PluginMetadata`` at registration time. The KV-convenience methods
(``put_kv_data`` / ``get_kv_data`` / ``delete_kv_data``) delegate to the injected
``PluginStore``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .context import PluginContext
from .metadata import PluginMetadata
from .registry import PluginRegistry

__all__ = ["Plugin"]


class Plugin:
    """Base class for runtime plugins. Subclass it and register with ``PluginManager``."""

    metadata: PluginMetadata
    context: PluginContext

    # Optional identity declared as class attributes (alternative to passing metadata
    # at registration). Left as ``None`` here so the manager can detect what to use.
    name: str | None = None
    author: str | None = None
    desc: str | None = None
    version: str | None = None
    repo: str | None = None
    display_name: str | None = None
    short_desc: str | None = None

    skills_dirs: list[Path] | None = None
    """Optional bundled skill directories this plugin contributes. Declare as paths
    (e.g. ``Path(__file__).parent / "skills"``); the host aggregates them and injects
    them into ``SkillManager`` as read-only extra scan roots. The skills extension never
    imports this layer — the two are wired together only via this path list at the host.
    """

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # Auto-register the class, keyed by its defining module.
        PluginRegistry.default().register_class(cls)

    def __init__(self, context: PluginContext) -> None:
        self.context = context
        self.metadata = context.metadata

    # —— lifecycle ——
    async def initialize(self) -> None:
        """Called when the plugin is activated."""

    async def terminate(self) -> None:
        """Called when the plugin is unloaded or reloaded."""

    # —— private KV persistence (delegates to context.plugin_store) ——
    @property
    def plugin_id(self) -> str:
        return self.metadata.name

    async def put_kv_data(self, key: str, value: Any) -> None:
        """Store a private key/value pair for this plugin."""
        store = self._require_store()
        await store.put(self.plugin_id, key, value)

    async def get_kv_data(self, key: str, default: Any = None) -> Any:
        """Read a private value previously stored by this plugin."""
        store = self._require_store()
        return await store.get(self.plugin_id, key, default)

    async def delete_kv_data(self, key: str) -> None:
        """Delete a private key for this plugin."""
        store = self._require_store()
        await store.delete(self.plugin_id, key)

    def _require_store(self):
        store = self.context.plugin_store
        if store is None:
            raise RuntimeError(
                "PluginContext.plugin_store is not set; cannot use KV persistence. "
                "Inject a PluginStore (e.g. InMemoryPluginStore) when registering."
            )
        return store
