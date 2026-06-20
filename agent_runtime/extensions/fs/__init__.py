"""Read-only filesystem tool set for local agent execution (safe-by-default).

These narrow tools give the agent a **path-addressed** file-access capability that
complements the skills subsystem's **name-addressed** instruction loading
(``Skill(name)`` loads ``SKILL.md``; the tools here read the ancillary files a
SKILL.md references — ``scripts/`` / ``references/`` / ``assets/`` — and anything else
under an allowed root).

Safe-by-default design: every tool is read-only, takes JSON-Schema parameters (no
shell), and confines every ``path`` to an allowed-root fence via ``realpath``
normalization. No execution isolation, container, or remote execution is used here.

**Transition contract.** These tools are the safety default *for environments that
lack an isolated executor*, not a forever-shape: when the host provides an isolated
execution environment (e.g. a container-isolated shell), the host SHOULD drop this
whole tool set from the ``ToolSet`` and substitute a single Bash tool, trading the
fence for the composition power of a real shell. See ``README.md`` and each tool's
docstring for the substitution wording.

The package depends only inward (``core`` / ``foundation``); it does not import the
skills or plugins extensions — those are wired to it only via plain path lists the
host hands in at construction.
"""

from __future__ import annotations

from .fs_tools import build_fs_tools
from .path_guard import resolve_within_roots

__all__ = [
    "build_fs_tools",
    "resolve_within_roots",
]
