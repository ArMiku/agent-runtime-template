"""The ``write_todos`` tool — the sole entry point for writing the plan.

Full-replacement semantics: each call carries the complete todo list and overwrites the
prior plan wholesale (decision 3 in the change design). This makes LLM-driven replanning
free — resending the list *is* the replan — and lets "read the current plan" degrade to
"read the stored state" with no history merge.

The tool binds a ``PluginStore`` at build time (mirroring ``build_skill_tool`` binding a
``SkillManager``) and resolves the ``session_id`` from ``run_context.context`` at call
time, so the plan it writes is the same session state the hook injects and persists.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agent_runtime.core.run_context import ContextWrapper
from agent_runtime.core.tool import FunctionTool

from .entities import Todo, TodoStatus
from .prompts import render_plan
from .store import save_plan

if TYPE_CHECKING:
    from agent_runtime.extensions.plugins.store import PluginStore

__all__ = ["build_write_todos_tool"]


def build_write_todos_tool(store: "PluginStore") -> FunctionTool:
    """Build the ``write_todos`` function tool bound to a ``PluginStore``.

    Args:
        store: The KV seam the plan is persisted through.

    Returns:
        The configured ``write_todos`` :class:`FunctionTool`.
    """

    async def handler(run_context: ContextWrapper[Any], *, todos: list[dict]) -> str:
        """Create or replace the current todo plan (full list, not a diff).

        Pass the **complete** list every call: the new list replaces the old plan
        wholesale. Each item needs ``content`` (the step) and ``status`` (one of
        ``pending`` / ``in_progress`` / ``completed``). Keep exactly one item
        ``in_progress``.
        """
        session_id = getattr(run_context.context, "session_id", "") or ""
        # Normalize raw tool args into typed Todos; unknown statuses fall back to pending
        # (Todo.from_dict), so a malformed item never breaks the write.
        plan = [Todo.from_dict(item) for item in todos if isinstance(item, dict)]
        await save_plan(store, session_id, plan)
        return f"Plan updated:\n\n{render_plan(plan)}"

    status_values = [s.value for s in TodoStatus]
    return FunctionTool(
        name="write_todos",
        description=(
            "Create or replace the current todo plan for an open-ended, multi-step task. "
            "Full-replacement: always pass the complete list of items, not a diff. Each "
            "item has 'content' (the step) and 'status' (pending/in_progress/completed). "
            "Keep exactly one item in_progress at a time. Update the plan as you make "
            "progress."
        ),
        parameters={
            "type": "object",
            "properties": {
                "todos": {
                    "type": "array",
                    "description": "The complete todo list, replacing any prior plan.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "content": {
                                "type": "string",
                                "description": "The step description.",
                            },
                            "status": {
                                "type": "string",
                                "enum": status_values,
                                "description": "The item's lifecycle status.",
                            },
                        },
                        "required": ["content", "status"],
                    },
                }
            },
            "required": ["todos"],
        },
        handler=handler,
    )
