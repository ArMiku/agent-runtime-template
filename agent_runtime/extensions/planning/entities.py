"""Plan data structures: the todo item and its lifecycle status.

A plan is an ordered list of :class:`Todo` items. The runtime stores it as the plan
independent state (keyed by ``session_id``), not in the message stream — see
``store.py``. ``write_todos`` overwrites the whole list each call (full-replacement
semantics), so a :class:`Todo` carries no identity beyond its position and content.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

__all__ = ["TodoStatus", "Todo"]


class TodoStatus(str, Enum):
    """The lifecycle status of a single todo item.

    Subclassing ``str`` makes the value JSON-serializable as a plain string, so a plan
    round-trips through the ``PluginStore`` KV seam (which accepts ``StoreValue``) without
    a custom encoder.
    """

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"


@dataclass
class Todo:
    """A single plan item: a description plus its current status."""

    content: str
    status: TodoStatus = TodoStatus.PENDING

    def to_dict(self) -> dict[str, str]:
        """Serialize to a plain ``dict`` for ``PluginStore`` persistence.

        Returns:
            A ``{"content", "status"}`` dict with the status as its string value.
        """
        return {"content": self.content, "status": self.status.value}

    @classmethod
    def from_dict(cls, data: dict[str, str]) -> "Todo":
        """Rebuild a :class:`Todo` from its persisted ``dict`` form.

        Unknown or missing ``status`` values fall back to ``PENDING`` so a malformed or
        forward-incompatible record never breaks plan loading.

        Args:
            data: A mapping with ``content`` and (optionally) ``status`` keys.

        Returns:
            The reconstructed :class:`Todo`.
        """
        raw_status = data.get("status", TodoStatus.PENDING.value)
        try:
            status = TodoStatus(raw_status)
        except ValueError:
            status = TodoStatus.PENDING
        return cls(content=str(data.get("content", "")), status=status)
