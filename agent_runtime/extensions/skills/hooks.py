"""Inject the active-skill inventory into the system message before each LLM step.

``SkillsPromptHook`` rides the ``on_llm_request``承重墙 from the plugin system: it
fires before each step's provider payload is assembled and before that step's context
compaction, so mutations to ``run_context.messages`` reach the provider. Placing the
inventory in the **leading system message** keeps it stable across the conversation,
which is the sweet spot for provider automatic prefix caching (the compactor preserves
leading system messages verbatim, so the inventory is never summarized away).

Idempotency is mandatory: ``on_llm_request`` fires every step, and the system message
is preserved verbatim by compaction, so appending per step would grow it without bound.
The inventory is therefore wrapped in sentinel comments and **replaced wholesale** each
step — never appended.
"""

from __future__ import annotations

import re
from typing import Any

from agent_runtime import logger
from agent_runtime.core.hooks import BaseAgentRunHooks
from agent_runtime.core.message import Message
from agent_runtime.core.run_context import ContextWrapper, TContext

from .skill_manager import SkillManager, build_skills_prompt

__all__ = ["SkillsPromptHook"]


class SkillsPromptHook(BaseAgentRunHooks[TContext]):
    """Write the active-skill inventory into the leading system message each LLM step."""

    _OPEN = "<!-- skills-inventory -->"
    _CLOSE = "<!-- /skills-inventory -->"
    _SEGMENT_RE = re.compile(re.escape(_OPEN) + r".*?" + re.escape(_CLOSE), re.DOTALL)

    def __init__(self, skill_manager: SkillManager) -> None:
        self.skill_manager = skill_manager

    async def on_llm_request(self, run_context: ContextWrapper[TContext]) -> None:
        try:
            skills = self.skill_manager.list_skills(active_only=True)
            block = build_skills_prompt(skills) if skills else ""
            self._apply_inventory(run_context, block)
        except Exception as e:  # noqa: BLE001 - a skills failure must never break the run
            logger.error(f"Error injecting skills inventory: {e}", exc_info=True)

    # —— internals ——

    def _apply_inventory(self, run_context: ContextWrapper[Any], block: str) -> None:
        system_msg = self._ensure_system_message(run_context)
        # System messages are string-typed in this runtime; guard anyway so structured
        # content is never corrupted.
        if not isinstance(system_msg.content, str):
            return
        system_msg.content = self._replace_inventory(system_msg.content, block)

    @staticmethod
    def _ensure_system_message(
        run_context: ContextWrapper[Any],
    ) -> Message:
        messages = run_context.messages
        if messages and getattr(messages[0], "role", None) == "system":
            return messages[0]
        msg = Message(role="system", content="")
        messages.insert(0, msg)
        return msg

    @classmethod
    def _replace_inventory(cls, text: str, block: str) -> str:
        """Return ``text`` with the inventory segment replaced (or added/removed).

        * block non-empty + segment present → replace the segment in place.
        * block non-empty + no segment      → append once, blank-line separated.
        * block empty     + segment present → remove the segment.
        * block empty     + no segment      → unchanged.
        """
        segment = f"{cls._OPEN}\n{block}\n{cls._CLOSE}" if block else ""
        if cls._SEGMENT_RE.search(text):
            return cls._SEGMENT_RE.sub(segment, text, count=1)
        if not segment:
            return text
        body = text.rstrip()
        return f"{body}\n\n{segment}" if body else segment
