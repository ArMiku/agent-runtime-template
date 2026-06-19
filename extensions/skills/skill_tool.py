"""The ``Skill(name)`` tool — load a named skill's instructions on demand.

This is the symmetric counterpart of name-based discovery: the inventory discloses skills
by ``name`` (system message), and this tool loads one by ``name`` (tool result). It is the
*only* way the LLM fetches ``SKILL.md`` instructions in this runtime.

Addressing is strictly by skill name (the registry), never by file path. The handler
accepts a single ``name`` argument and delegates to ``SkillManager.load_skill``, which
resolves the directory internally. Because no path argument is exposed, the LLM cannot
use this tool to read arbitrary files in a skill directory (ancillary files like
``scripts/`` / ``references/`` / ``assets/`` are read by a separate general-purpose READ
capability, delivered by an independent change) and there is no path-traversal surface.
"""

from __future__ import annotations

from typing import Any

from agent_runtime.core.run_context import ContextWrapper
from agent_runtime.core.tool import FunctionTool

from .skill_manager import SkillManager

__all__ = ["build_skill_tool"]


def build_skill_tool(skill_manager: SkillManager) -> FunctionTool:
    """Build the ``Skill`` function tool bound to a ``SkillManager``.

    The tool loads a named skill's ``SKILL.md`` instructions. It takes exactly one
    argument — the skill ``name`` — and accepts no path argument (see module docstring).
    """

    async def handler(run_context: ContextWrapper[Any], *, name: str) -> str:
        """Load the full SKILL.md instructions for a named skill.

        Call this with the skill's ``name`` (as shown in the available-skills inventory)
        before executing that skill, so you act on its exact instructions rather than
        assumptions.
        """
        return skill_manager.load_skill(name)

    return FunctionTool(
        name="Skill",
        description=(
            "Load the full SKILL.md instructions for a named skill. Pass the skill's "
            "name exactly as it appears in the available-skills inventory; call this "
            "before executing a skill to ground yourself in its instructions. This tool "
            "loads instructions only — it does not read arbitrary files."
        ),
        parameters={
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "The skill name to load, from the available-skills inventory.",
                }
            },
            "required": ["name"],
        },
        handler=handler,
    )
