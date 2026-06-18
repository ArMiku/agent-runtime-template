"""Extensions layer — pluggable add-on subsystems built on top of the core seams.

Nothing in ``core`` / ``provider`` / ``message`` / ``media`` depends on this layer; it
only depends *inward*. Seam abstractions and their open-box defaults live beside each
other inside their domain packages (e.g. ``core`` holds both ``BaseFunctionToolExecutor``
and the default ``FunctionToolExecutor``) — not here. This layer is for optional
subsystems:

* :mod:`~agent_runtime.extensions.skills` — skills loading (roadmap; see its README).
* :mod:`~agent_runtime.extensions.plugins` — plugin / Star system (roadmap; see its README).
"""

from __future__ import annotations
