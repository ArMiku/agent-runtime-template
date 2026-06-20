"""Lightweight context injected into plugins.

A neutral, minimal injection object. It deliberately holds **no** ``event`` /
``platform`` / concrete DB handle / conversation-manager god object. Instead it exposes
just enough injectable seams for a plugin to reach the conversation context and its own
persistence:

* (A) live multi-turn messages of the current run — already delivered via each hook's
  ``run_context.messages`` argument, so not duplicated here;
* (B) cross-run persisted session history — via the existing ``ContextStore`` seam;
* (C) the plugin's private KV persistence — via the new ``PluginStore`` seam.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from agent_runtime.core.context.context_store import ContextStore
from agent_runtime.core.message import Message

from .metadata import PluginMetadata
from .store import PluginStore

if TYPE_CHECKING:
    from .registry import PluginRegistry

__all__ = ["PluginContext"]


@dataclass
class PluginContext:
    """Injected into each plugin instance at construction time."""

    metadata: PluginMetadata
    """This plugin's own metadata."""
    config: dict | None = None
    """This plugin's configuration (from metadata or supplied at registration)."""
    registry: "PluginRegistry | None" = None
    """Optional handle for looking up other plugins."""

    # —— conversation context (injectable seams; default in-memory, no DB) ——
    context_store: ContextStore | None = None
    """Cross-run session-history read/write (existing runtime seam)."""
    plugin_store: PluginStore | None = None
    """Plugin-private KV persistence (seam added by this change)."""

    async def conversation_history(self, session_id: str) -> list[Message]:
        """Read a session's persisted history (delegates to ``context_store``).

        Returns an empty list when no ``context_store`` is injected.
        """
        if self.context_store is None:
            return []
        return await self.context_store.load(session_id)
