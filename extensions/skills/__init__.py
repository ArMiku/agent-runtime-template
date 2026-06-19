"""Read-only skills subsystem for local agent execution.

Discover ``SKILL.md`` instruction bundles, push the active-skill inventory into the
system message, and load a named skill's instructions on demand via the ``Skill`` tool.
Local execution only: the runtime reads ``skills_root`` and ``skills.json`` as inputs —
it never installs, deletes, toggles, or persists. See ``README.md`` for the addressing-
space boundary (``Skill`` = name addressing vs a general READ tool = path addressing),
the two-stage data flow, and the host operations contract.
"""

from __future__ import annotations

from .hooks import SkillsPromptHook
from .skill_manager import SkillInfo, SkillManager, build_skills_prompt
from .skill_tool import build_skill_tool

__all__ = [
    "SkillManager",
    "SkillInfo",
    "build_skills_prompt",
    "SkillsPromptHook",
    "build_skill_tool",
]
