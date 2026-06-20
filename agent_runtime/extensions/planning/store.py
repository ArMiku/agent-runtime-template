"""Plan persistence over the ``PluginStore`` KV seam.

The plan lives as independent state — not in the message stream — so context compaction
never drops it and a host-injected persistent ``PluginStore`` makes it survive across
processes. Storage is isolated by ``session_id`` under one reserved ``plugin_id``
(:data:`PLANNING_PLUGIN_ID`), mirroring how plugins isolate their own private KV data.

Both ``write_todos`` (machine edits) and host-side manual edits write through the same
:func:`save_plan`, so plan recovery has exactly one code path.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .entities import Todo

if TYPE_CHECKING:
    from agent_runtime.extensions.plugins.store import PluginStore

__all__ = ["PLANNING_PLUGIN_ID", "load_plan", "save_plan"]

# Reserved plugin_id namespace for plan state. Distinct from any real plugin's id so plan
# storage never collides with a plugin's private KV data.
PLANNING_PLUGIN_ID = "__planning__"


async def load_plan(store: PluginStore, session_id: str) -> list[Todo]:
    """Load the current plan for a session.

    Args:
        store: The KV seam to read from.
        session_id: The session whose plan to load.

    Returns:
        The session's todo list (most recent ``write_todos`` content), or an empty list
        when the session has no plan yet.
    """
    raw = await store.get(PLANNING_PLUGIN_ID, session_id, [])
    if not isinstance(raw, list):
        return []
    return [Todo.from_dict(item) for item in raw if isinstance(item, dict)]


async def save_plan(store: PluginStore, session_id: str, todos: list[Todo]) -> None:
    """Persist a session's plan, replacing any prior plan wholesale.

    Args:
        store: The KV seam to write to.
        session_id: The session whose plan to overwrite.
        todos: The complete new plan (full-replacement semantics).
    """
    await store.put(PLANNING_PLUGIN_ID, session_id, [todo.to_dict() for todo in todos])
