"""Planning prompts: the system guidance block and the plan renderer.

Two surfaces share one renderer:

* :func:`render_plan` turns the current plan into a compact checklist. It feeds both the
  ``on_llm_request`` injection (so the LLM always sees the live plan) and the
  ``write_todos`` tool result (so the LLM confirms what it just wrote).
* :func:`build_planning_prompt` is the static guidance that teaches the LLM how and when
  to use ``write_todos``. It is injected once, sentinel-wrapped, into the leading system
  message — mirroring the skills inventory.
"""

from __future__ import annotations

from .entities import Todo, TodoStatus

__all__ = ["render_plan", "build_planning_prompt"]

# Checkbox glyphs per status — a glanceable plan for both the LLM and a human reading logs.
_STATUS_MARK = {
    TodoStatus.PENDING: "[ ]",
    TodoStatus.IN_PROGRESS: "[~]",
    TodoStatus.COMPLETED: "[x]",
}


def render_plan(todos: list[Todo]) -> str:
    """Render the current plan as a markdown checklist.

    Args:
        todos: The current plan items, in order.

    Returns:
        A markdown checklist, one line per item, or a placeholder line when the plan is
        empty.
    """
    if not todos:
        return "_No plan yet. Call `write_todos` to create one._"
    lines = [f"- {_STATUS_MARK[todo.status]} {todo.content}" for todo in todos]
    return "\n".join(lines)


def build_planning_prompt(todos: list[Todo]) -> str:
    """Build the planning guidance block, embedding the current plan.

    The block teaches the full-replacement contract and the single-``in_progress``
    convention, then shows the live plan so the LLM never has to reconstruct it from the
    message history (which context compaction may have dropped).

    Args:
        todos: The current plan items, rendered inline under "Current plan".

    Returns:
        The markdown guidance block (without sentinels — the hook wraps it).
    """
    return (
        "## Plan\n\n"
        "For any open-ended, multi-step task, keep a todo plan with the `write_todos` "
        "tool. The plan is your single source of truth for progress.\n\n"
        "Rules:\n"
        "- `write_todos` is **full-replacement**: always pass the complete list, not a "
        "diff. To change one item, resend every item.\n"
        "- Each item has `content` and a `status` of `pending`, `in_progress`, or "
        "`completed`.\n"
        "- Keep **exactly one** item `in_progress` at a time; mark it `completed` before "
        "starting the next.\n"
        "- Update the plan as you go, and do not declare the task finished while items "
        "remain unfinished.\n\n"
        "### Current plan\n\n"
        f"{render_plan(todos)}"
    )
