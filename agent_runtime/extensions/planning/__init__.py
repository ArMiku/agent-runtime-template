"""Plan-and-execute capability as an emergent-todo extension.

Sibling to ``skills/`` and ``fs/``: this extension adds plan-and-execute behavior to the
existing ReAct loop without touching the control-flow kernel. It contributes one tool and
one hook:

* ``write_todos`` — the sole entry point for writing the plan (full-replacement).
* :class:`PlanningHook` — injects the live plan into the system message each step
  (``on_llm_request``) and vetoes premature completion while todos remain unfinished
  (``on_before_complete``).

The plan is independent state keyed by ``session_id``, persisted through the ``PluginStore``
KV seam, so it survives context compaction and (with a host-injected persistent store)
cross-process restarts. The extension depends only on ``core`` abstractions plus the
``PluginStore`` protocol — never on a concrete runner — so a future runner can reuse it
unchanged.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from agent_runtime.core.tool import FunctionTool

from .entities import Todo, TodoStatus
from .prompts import build_planning_prompt, render_plan
from .store import PLANNING_PLUGIN_ID, load_plan, save_plan
from .todo_hook import PlanningHook
from .todo_tool import build_write_todos_tool

if TYPE_CHECKING:
    from agent_runtime.extensions.plugins.store import PluginStore

__all__ = [
    "Todo",
    "TodoStatus",
    "PlanningHook",
    "build_write_todos_tool",
    "build_planning_prompt",
    "render_plan",
    "load_plan",
    "save_plan",
    "PLANNING_PLUGIN_ID",
    "build_planning_extension",
]


def build_planning_extension(store: "PluginStore", max_reminders: int = 2) -> tuple[FunctionTool, PlanningHook]:
    """Construct the planning tool and hook bound to one shared plan store.

    Centralizes assembly so the composition root never reaches into the extension's
    internals: it hands over a ``PluginStore`` and gets back the ``write_todos`` tool and
    the :class:`PlanningHook`, both reading/writing the same session-keyed plan state.

    Args:
        store: The KV seam the plan is persisted through (the same instance is shared by
            the tool and the hook so they see one plan).
        max_reminders: Per-session completion-veto cap passed to :class:`PlanningHook`.

    Returns:
        A ``(write_todos_tool, planning_hook)`` pair ready to add to a ``ToolSet`` and a
        hook chain respectively.
    """
    tool = build_write_todos_tool(store)
    hook = PlanningHook(store, max_reminders=max_reminders)
    return tool, hook
