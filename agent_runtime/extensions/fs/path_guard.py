"""Path containment for the read-only fs tools (the only security boundary).

``resolve_within_roots`` normalizes a requested path with ``os.path.realpath``
(resolving symlinks, ``..`` and relative segments) and admits it only if it falls
inside one of the allowed roots — checked with a strict prefix (``root + sep``) so a
root like ``/data/skills`` does not also admit ``/data/skills-secret``. Any path that
escapes after resolution (including a static symlink escape) raises
``PermissionError``.

When no roots are given, the skills directory (``get_skills_dir``) is the default
fence — this closes the skills ancillary-file read dependency. The fence is a
check-then-open guard: it does not defend against a TOCTOU race between the
containment check and the subsequent open. For a local, read-only workload that race
is low-risk (winning it requires swapping a path already known to be inside the
fence), so it is recorded here as a known trade-off rather than left implicit.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path

from agent_runtime.foundation.paths import get_skills_dir

__all__ = ["resolve_within_roots"]


def resolve_within_roots(
    path: str | os.PathLike[str],
    allowed_roots: Iterable[str | os.PathLike[str]] | None,
) -> Path:
    """Resolve ``path`` and confirm it sits under one of ``allowed_roots``.

    Returns the resolved ``Path`` (symlinks / ``..`` expanded). Raises
    ``PermissionError`` if the resolved path is not contained by any root — including
    a symlink that escapes the fence after resolution. When ``allowed_roots`` is
    ``None`` or empty, the skills directory is used as the default fence.
    """
    roots = [os.fspath(root) for root in (allowed_roots or [get_skills_dir()])]
    real = os.path.realpath(os.fspath(path))
    for root in roots:
        root_real = os.path.realpath(root)
        if real == root_real or real.startswith(root_real + os.sep):
            return Path(real)
    raise PermissionError(f"path '{real}' is not under any allowed root")
