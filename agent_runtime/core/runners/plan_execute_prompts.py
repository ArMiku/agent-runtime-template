"""Prompts and the ``submit_plan`` tool for the plan-and-execute runner.

The plan-and-execute runner (``plan_execute_runner.py``) drives an explicit
``PLAN → EXEC → REPLAN`` state machine. ``PLAN`` and ``REPLAN`` are each a single
structured reasoning step — they produce or revise a todo plan — and they must NOT
route through the ReAct tool loop. To get a reliable structured result from a single
non-agent LLM call, both phases call ``provider.text_chat(func_tool=<submit_plan only>,
tool_choice="required")``: the model is forced to emit a ``submit_plan`` tool call whose
arguments carry the plan. The runner parses those arguments into ``list[Todo]`` directly
— the tool's handler is never executed (there is nothing to execute; the call exists only
to constrain the output shape).

This module is intentionally free of any hook/store coupling: it holds only static prompt
text, the tool schema, and a pure plan renderer. The planning extension's ``Todo`` schema
is reused (``extensions.planning.entities``) so plan-and-execute shares one plan data
model with the emergent-todo route.
"""

from __future__ import annotations

from agent_runtime.core.run_context import ContextWrapper
from agent_runtime.core.tool import FunctionTool
from agent_runtime.extensions.planning.entities import Todo, TodoStatus

__all__ = [
    "PLANNER_SYSTEM_PROMPT",
    "REPLANNER_SYSTEM_PROMPT",
    "build_submit_plan_tool",
    "render_plan_checklist",
]

# The planner/replanner must hand back a plan through one tool call. The schema mirrors the
# planning extension's ``write_todos`` (content + status) so the parsed list drops straight
# into the shared ``Todo`` model and the ``save_plan`` mirror.
_SUBMIT_PLAN_PARAMETERS = {
    "type": "object",
    "properties": {
        "todos": {
            "type": "array",
            "description": "The complete todo plan, in execution order.",
            "items": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The step description.",
                    },
                    "status": {
                        "type": "string",
                        "enum": [s.value for s in TodoStatus],
                        "description": "The item's lifecycle status.",
                    },
                },
                "required": ["content", "status"],
            },
        }
    },
    "required": ["todos"],
}

PLANNER_SYSTEM_PROMPT = (
    "You are the planner in a plan-and-execute pipeline. Given the user's task, produce a "
    "complete, ordered todo plan that an executor agent will carry out step by step.\n\n"
    "Rules:\n"
    "- Call the `submit_plan` tool exactly once with the full plan. Do not answer in prose.\n"
    "- Each item has `content` (a concrete, self-contained step) and `status`. Emit every "
    "item as `pending` except at most one `in_progress`; never mark an item `completed` "
    "before it has run.\n"
    "- Keep the plan minimal and executable: as few steps as needed, each one independently "
    "verifiable. If the task needs no steps, submit an empty list.\n"
    "- Every todo must have a clear, recognizable completion criterion — a condition the "
    "executor can hit and stop at (a test passing, a specific list produced, a fixed sample "
    "gathered). Avoid open-ended or fuzzily bounded scope such as 'all', 'every', 'cover "
    "everything': concretely bound it (a specific list, a fixed sample, a passing check) or "
    "decompose it. A todo may legitimately take many steps — what matters is that it "
    "converges to a recognizable done, not that it is small.\n"
    "- Match planning depth to the task: exploratory or research tasks do best with a small "
    "coarse skeleton that the replanner refines progressively as findings arrive (do not "
    "pre-enumerate exhaustively); well-specified implementation tasks (e.g. coding from a "
    "clear spec) warrant a fuller, detailed upfront plan.\n"
    "- You are planning only — do not attempt the work yourself."
)

REPLANNER_SYSTEM_PROMPT = (
    "You are the replanner in a plan-and-execute pipeline. One step has just been executed "
    "and its result summary is in the conversation. Revise the remaining plan in light of it.\n\n"
    "Rules:\n"
    "- Call the `submit_plan` tool exactly once with the complete revised plan (full list, "
    "not a diff). Do not answer in prose.\n"
    "- Carry forward finished steps marked `completed`; set the next step to run to "
    "`in_progress`; leave the rest `pending`.\n"
    "- Drop steps that are now unnecessary, add steps the result revealed as needed, and "
    "reorder for efficiency.\n"
    "- Converge rather than chase exhaustive coverage: once the task's goal is reasonably "
    "met, stop spawning new steps and steer the remainder toward synthesis/completion — a "
    "bounded, sufficient result beats an unbounded one. When nothing significant remains, "
    "submit the plan with all items `completed` (or an empty list).\n"
    "- Any new step you add must itself have a clear, recognizable completion criterion; "
    "never introduce an open-ended or fuzzily bounded step (e.g. 'search all', 'cover "
    "everything') — concretely bound it or decompose it first.\n"
    "- You are replanning only — do not attempt the work yourself."
)


def render_plan_checklist(todos: list[Todo]) -> str:
    """Render a plan as a markdown checklist (pure; for the PLAN/REPLAN ``llm_result``).

    Independent of the planning extension's renderer so this module stays decoupled from
    ``extensions.planning`` hook/tool surfaces — only the shared ``Todo`` data model is reused.
    """
    if not todos:
        return "_No plan._"
    mark = {
        TodoStatus.PENDING: "[ ]",
        TodoStatus.IN_PROGRESS: "[~]",
        TodoStatus.COMPLETED: "[x]",
    }
    return "\n".join(f"- {mark.get(t.status, '[ ]')} {t.content}" for t in todos)


def build_submit_plan_tool() -> FunctionTool:
    """Build the ``submit_plan`` tool used to force a structured plan from one LLM call.

    The handler is a stub: the plan-and-execute runner parses the emitted tool-call arguments
    directly and never invokes the handler. It exists only so the provider sends a tool schema
    that ``tool_choice="required"`` can bind to.
    """

    async def _handler(run_context: ContextWrapper, *, todos: list[dict]) -> str:  # noqa: ARG001
        # Not executed by the runner; provided so the tool is a well-formed FunctionTool.
        plan = [Todo.from_dict(item) for item in todos if isinstance(item, dict)]
        return f"submit_plan received {len(plan)} item(s)."

    return FunctionTool(
        name="submit_plan",
        description=(
            "Submit the complete todo plan for the task. Always pass the full ordered list "
            "(not a diff). Each item has 'content' (the step) and 'status' "
            "(pending/in_progress/completed). Emit this once — do not answer in prose."
        ),
        parameters=_SUBMIT_PLAN_PARAMETERS,
        handler=_handler,
    )
