"""Injectable plugin KV-persistence seam (design.md §3.2.1).

The runtime never assumes a storage backend. A plugin's private key/value data is read
and written through the injectable :class:`PluginStore` Protocol, isolated by
``plugin_id``. The runtime ships :class:`InMemoryPluginStore` (a process-local dict, no
database); a host application injects its own persistent adapter. This mirrors the
existing :class:`~agent_runtime.core.context.context_store.ContextStore` seam.
"""

from __future__ import annotations

from typing import Protocol, TypeVar, runtime_checkable

__all__ = ["PluginStore", "InMemoryPluginStore", "StoreValue"]

StoreValue = int | float | str | bytes | bool | dict | list | None
_VT = TypeVar("_VT")


@runtime_checkable
class PluginStore(Protocol):
    """Per-plugin key/value persistence, isolated by ``plugin_id``."""

    async def get(self, plugin_id: str, key: str, default: _VT) -> _VT:
        """Return the stored value for ``(plugin_id, key)`` or ``default`` if absent."""
        ...

    async def put(self, plugin_id: str, key: str, value: StoreValue) -> None:
        """Store ``value`` under ``(plugin_id, key)``."""
        ...

    async def delete(self, plugin_id: str, key: str) -> None:
        """Delete ``(plugin_id, key)`` if present."""
        ...


class InMemoryPluginStore:
    """Default ``PluginStore``: an in-process nested dict. No database, no persistence."""

    def __init__(self) -> None:
        self._store: dict[str, dict[str, StoreValue]] = {}

    async def get(self, plugin_id: str, key: str, default: _VT) -> _VT:
        bucket = self._store.get(plugin_id)
        if bucket is None or key not in bucket:
            return default
        return bucket[key]  # type: ignore[return-value]

    async def put(self, plugin_id: str, key: str, value: StoreValue) -> None:
        self._store.setdefault(plugin_id, {})[key] = value

    async def delete(self, plugin_id: str, key: str) -> None:
        bucket = self._store.get(plugin_id)
        if bucket is not None:
            bucket.pop(key, None)
