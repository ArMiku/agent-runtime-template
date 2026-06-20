"""Extensions layer — pluggable add-on subsystems built on top of the core seams.

Nothing in ``core`` / ``provider`` / ``message`` / ``media`` depends on this layer; it
only depends *inward*. Seam abstractions and their open-box defaults live beside each
other inside their domain packages (e.g. ``core`` holds both ``BaseFunctionToolExecutor``
and the default ``FunctionToolExecutor``) — not here. This layer is for optional
subsystems:

* :mod:`~agent_runtime.extensions.skills` — read-only skills subsystem: discovers
  ``SKILL.md`` bundles, injects the active-skill inventory into the system message, and
  loads instructions by name via the ``Skill`` tool (see its README).
* :mod:`~agent_runtime.extensions.plugins` — framework-agnostic, event-free plugin
  mechanism: plugins contribute tools (``@tool``) and agent-loop hooks (``@on_*``) via a
  neutral ``Plugin`` base + ``PluginManager`` (see its README).
"""

from __future__ import annotations
