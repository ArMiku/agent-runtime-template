"""Neutral, injectable path helpers.

The runtime needs only three directory roots and must not assume any host
application's source tree or packaging layout. Layout:

* ``get_data_dir()``  — persistent runtime data (default ``$AGENT_RUNTIME_DATA_DIR`` or
  ``<cwd>/data``; the MCP tool manager stores its config here).
* ``get_temp_dir()``  — scratch space for downloads / image caches (a per-package
  directory under the system temp dir, created on demand).
* ``get_project_dir()`` — the package install root; used only to resolve bundled sample
  assets (e.g. the STT health-check clip) and is best-effort.
"""

from __future__ import annotations

import os
import tempfile

__all__ = [
    "get_data_dir",
    "get_temp_dir",
    "get_project_dir",
]

_DATA_DIR_ENV = "AGENT_RUNTIME_DATA_DIR"
_TEMP_SUBDIR = "agent_runtime"


def get_data_dir() -> str:
    """Return the runtime's persistent data directory (injectable via env)."""
    base = os.environ.get(_DATA_DIR_ENV)
    if not base:
        base = os.path.join(os.getcwd(), "data")
    path = os.path.realpath(base)
    os.makedirs(path, exist_ok=True)
    return path


def get_temp_dir() -> str:
    """Return a scratch temp directory (created on demand)."""
    path = os.path.realpath(os.path.join(tempfile.gettempdir(), _TEMP_SUBDIR))
    os.makedirs(path, exist_ok=True)
    return path


def get_project_dir() -> str:
    """Return this package's install root (best-effort, for bundled assets)."""
    return os.path.realpath(os.path.join(os.path.dirname(os.path.abspath(__file__))))
