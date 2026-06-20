"""Injectable context-persistence seam.

The runner manages only an in-memory ``list[Message]`` per run; it never assumes a
storage backend. Persistence is the caller's concern, exposed through
:class:`ContextStore`. The default :class:`InMemoryContextStore` keeps history in a
process-local dict and pulls in no database — enough for a single-run demo and as a
reference for real implementations (SQLite, Redis, …).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from agent_runtime.core.message import Message

__all__ = ["ContextStore", "InMemoryContextStore"]


@runtime_checkable
class ContextStore(Protocol):
    """Load/save a session's message history."""

    async def load(self, session_id: str) -> list[Message]:
        """Return the persisted message history for ``session_id`` (empty if new)."""
        ...

    async def save(self, session_id: str, messages: list[Message]) -> None:
        """Persist the current message history for ``session_id``."""
        ...


class InMemoryContextStore:
    """Default ``ContextStore``: an in-process dict. No database, no persistence."""

    def __init__(self) -> None:
        self._store: dict[str, list[Message]] = {}

    async def load(self, session_id: str) -> list[Message]:
        return list(self._store.get(session_id, []))

    async def save(self, session_id: str, messages: list[Message]) -> None:
        self._store[session_id] = list(messages)
