"""Default ``TContext`` implementation.

``TContext`` is the opaque, caller-owned session object threaded through
:class:`~agent_runtime.core.run_context.ContextWrapper`. The runtime makes no
assumptions about it. This module ships a minimal default — just enough identity
for session-keyed persistence and logging — that consumers replace with their own
richer context (request objects, auth principals, tenant ids, …).
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["SessionContext"]


@dataclass
class SessionContext:
    """Minimal default context: a session (and optional user) identity."""

    session_id: str
    user_id: str | None = None
