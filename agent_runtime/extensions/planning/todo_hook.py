"""Inject the live plan each step and veto premature completion.

``PlanningHook`` rides the same ``on_llm_request`` 承重墙 as ``SkillsPromptHook``: it
reads the current plan from independent state (the ``PluginStore``, keyed by
``session_id``) and writes it, sentinel-wrapped, into the leading system message every
step. Because the plan's source of truth is independent state — not the message stream —
the injection survives context compaction: even after the original ``write_todos`` tool
message is summarized away, the plan is re-read from the store and re-injected intact.

The hook also implements the kernel's new ``on_before_complete`` veto: when the LLM tries
to finish with unfinished todos still on the plan, the hook appends a reminder and refuses
the completion, so ``step_until_done`` runs another round. A per-session reminder cap
(``_MAX_REMINDERS``) prevents an infinite veto loop — once hit, the completion is admitted.

Idempotency mirrors the skills hook: the plan block is wrapped in its own sentinels
(``<!-- todo-state -->`` … ``<!-- /todo-state -->``) and replaced wholesale each step,
never appended, and is distinct from the skills sentinels so the two coexist in one
system message without interfering.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from agent_runtime import logger
from agent_runtime.core.hooks import BaseAgentRunHooks
from agent_runtime.core.message import Message
from agent_runtime.core.run_context import ContextWrapper, TContext
from agent_runtime.provider.entities import LLMResponse

from .entities import Todo, TodoStatus
from .prompts import build_planning_prompt
from .store import load_plan, save_plan

if TYPE_CHECKING:
    from agent_runtime.extensions.plugins.store import PluginStore

__all__ = ["PlanningHook"]

# Reminder injected when the LLM tries to finish with unfinished todos still on the plan.
_REMINDER = (
    "You have not finished the plan. There are still unfinished todos. Continue working "
    "on the next pending item, or call `write_todos` to update the plan. Only stop once "
    "every item is completed."
)


class PlanningHook(BaseAgentRunHooks[TContext]):
    """Inject the live plan each step and veto completion while todos remain unfinished."""

    _OPEN = "<!-- todo-state -->"
    _CLOSE = "<!-- /todo-state -->"
    _SEGMENT_RE = re.compile(re.escape(_OPEN) + r".*?" + re.escape(_CLOSE), re.DOTALL)

    def __init__(self, store: "PluginStore", max_reminders: int = 2) -> None:
        """Bind the hook to a plan store.

        Args:
            store: The KV seam the plan is read from and persisted through.
            max_reminders: Per-session cap on completion-veto reminders before the
                completion is admitted regardless of unfinished todos (default 2,
                matching DeerFlow's ``TodoMiddleware``).
        """
        self.store = store
        self._max_reminders = max_reminders
        # Per-session veto count, so the cap is independent across concurrent sessions.
        self._reminder_counts: dict[str, int] = {}

    # —— plan read/write surface (shared by tool, injection, and host manual edits) ——

    async def read_plan(self, session_id: str) -> list[Todo]:
        """Return the current plan for ``session_id`` (empty list if none)."""
        return await load_plan(self.store, session_id)

    async def write_plan(self, session_id: str, todos: list[Todo]) -> None:
        """Overwrite the plan for ``session_id`` through the shared write channel.

        Host-side manual edits call this, landing in the same independent state the LLM's
        ``write_todos`` writes — so recovery and injection have a single source of truth.
        """
        await save_plan(self.store, session_id, todos)

    # —— hook events ——

    async def on_llm_request(self, run_context: ContextWrapper[TContext]) -> None:
        try:
            session_id = getattr(run_context.context, "session_id", "") or ""
            todos = await load_plan(self.store, session_id)
            block = build_planning_prompt(todos) if todos else ""
            self._apply_plan(run_context, block)
        except Exception as e:  # noqa: BLE001 - a planning failure must never break the run
            logger.error(f"Error injecting plan state: {e}", exc_info=True)

    async def on_before_complete(
        self,
        run_context: ContextWrapper[TContext],
        llm_response: LLMResponse,
    ) -> bool:
        """Veto completion while unfinished todos remain, up to the per-session cap.

        Returns ``False`` (refusing the completion) and appends a reminder when the plan
        has any non-``completed`` item and the session's reminder count is still under the
        cap. Otherwise returns ``True`` to admit — including when the plan is empty/all
        done, or the cap has been reached.
        """
        session_id = getattr(run_context.context, "session_id", "") or ""
        todos = await load_plan(self.store, session_id)
        unfinished = any(todo.status is not TodoStatus.COMPLETED for todo in todos)
        if not unfinished:
            return True
        if self._reminder_counts.get(session_id, 0) >= self._max_reminders:
            return True  # cap reached — admit to avoid an infinite veto loop
        self._reminder_counts[session_id] = self._reminder_counts.get(session_id, 0) + 1
        run_context.messages.append(Message(role="user", content=_REMINDER))
        return False

    # —— internals (sentinel-delimited wholesale replacement, mirrors SkillsPromptHook) ——

    def _apply_plan(self, run_context: ContextWrapper[Any], block: str) -> None:
        system_msg = self._ensure_system_message(run_context)
        # System messages are string-typed in this runtime; guard anyway so structured
        # content is never corrupted.
        if not isinstance(system_msg.content, str):
            return
        system_msg.content = self._replace_segment(system_msg.content, block)

    @staticmethod
    def _ensure_system_message(run_context: ContextWrapper[Any]) -> Message:
        messages = run_context.messages
        if messages and getattr(messages[0], "role", None) == "system":
            return messages[0]
        msg = Message(role="system", content="")
        messages.insert(0, msg)
        return msg

    @classmethod
    def _replace_segment(cls, text: str, block: str) -> str:
        """Return ``text`` with the plan segment replaced (or added/removed).

        * block non-empty + segment present → replace in place.
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
