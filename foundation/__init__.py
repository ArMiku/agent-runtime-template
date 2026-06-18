"""Foundation layer — host-neutral infrastructure with no domain knowledge.

These modules know nothing about agents, providers, or messages; everything else in
the package is allowed to depend on them, never the other way around. Keeping them in
one layer makes the dependency direction obvious and the import contract auditable.

* :mod:`~agent_runtime.foundation.log` — logging shim (``logger`` is re-exported at
  package root as ``from agent_runtime import logger``).
* :mod:`~agent_runtime.foundation.paths` — injectable data/temp/project dir helpers.
* :mod:`~agent_runtime.foundation.network` — shared HTTP/SSL helpers.
* :mod:`~agent_runtime.foundation.io_utils` — download / file helpers.
* :mod:`~agent_runtime.foundation.string_utils` — pure string helpers.
* :mod:`~agent_runtime.foundation.exceptions` — runtime exception hierarchy.
* :mod:`~agent_runtime.foundation.config` — ``FileCallbackService`` application seam.
"""

from __future__ import annotations
