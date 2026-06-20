"""Planning assembly switch (default off, on wires tool+hook coexisting with skills) and
the dependency-direction guard that keeps planning runner-agnostic (tasks 6.4, 6.5)."""

from __future__ import annotations

import ast
from pathlib import Path

from agent_runtime.core.hooks_chain import ChainedAgentRunHooks
from agent_runtime.extensions.planning.todo_hook import PlanningHook
from agent_runtime.local_runtime import build_local_agent_basics

PKG_ROOT = Path(__file__).resolve().parent.parent
PLANNING_DIR = PKG_ROOT / "agent_runtime" / "extensions" / "planning"


def test_default_off_excludes_planning():
    basics = build_local_agent_basics(include_fs=False)
    assert basics.planning_hook is None
    assert "write_todos" not in basics.tools.names()


def test_on_wires_tool_and_hook():
    basics = build_local_agent_basics(include_fs=False, include_planning=True)
    assert isinstance(basics.planning_hook, PlanningHook)
    assert "write_todos" in basics.tools.names()


def test_planning_coexists_with_skills_hook():
    """6.4: with planning on, the hook chain holds both the skills hook and the planning
    hook (chained), so they coexist."""
    basics = build_local_agent_basics(include_fs=False, include_planning=True)
    assert isinstance(basics.hooks, ChainedAgentRunHooks)
    hook_types = {type(h).__name__ for h in basics.hooks._hooks}
    assert "SkillsPromptHook" in hook_types
    assert "PlanningHook" in hook_types


def test_injected_store_is_shared_by_tool_and_hook():
    """The plan written via the tool is visible to the hook — same backing store."""
    from agent_runtime.extensions.plugins.store import InMemoryPluginStore

    store = InMemoryPluginStore()
    basics = build_local_agent_basics(include_fs=False, include_planning=True, plugin_store=store)
    assert basics.planning_hook is not None
    assert basics.planning_hook.store is store


def _is_type_checking_guard(node: ast.If) -> bool:
    """True if ``node`` is an ``if TYPE_CHECKING:`` block (bare name or typing.TYPE_CHECKING)."""
    test = node.test
    if isinstance(test, ast.Name) and test.id == "TYPE_CHECKING":
        return True
    return isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"


def _planning_runtime_imports() -> list[str]:
    """Absolute imports in planning source, excluding ``if TYPE_CHECKING:`` bodies."""
    modules: list[str] = []
    for path in sorted(PLANNING_DIR.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        # Collect import nodes nested inside any TYPE_CHECKING guard, to exclude them.
        guarded: set[int] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.If) and _is_type_checking_guard(node):
                for inner in ast.walk(node):
                    guarded.add(id(inner))
        for node in ast.walk(tree):
            if id(node) in guarded:
                continue
            if isinstance(node, ast.Import):
                modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.level == 0 and node.module:
                    modules.append(node.module)
    return modules


def test_planning_does_not_import_concrete_runner():
    """6.5: the planning extension must not depend on ToolLoopAgentRunner or core.runners
    internals — it interacts only via BaseAgentRunHooks / ToolSet / PluginStore abstractions,
    so a future runner can reuse it unchanged."""
    offenders = [
        m
        for m in _planning_runtime_imports()
        if m.startswith("agent_runtime.core.runners") or "ToolLoopAgentRunner" in m
    ]
    assert not offenders, f"planning must not import a concrete runner: {offenders}"


def test_planning_runtime_imports_only_abstractions():
    """6.5 (cont.): the only cross-extension coupling is to the PluginStore seam, and even
    that is TYPE_CHECKING-only — no runtime import of the plugins package from planning
    source (the concrete store is injected at the composition root)."""
    plugins_runtime = [m for m in _planning_runtime_imports() if m.startswith("agent_runtime.extensions.plugins")]
    assert not plugins_runtime, (
        f"planning should reference PluginStore only under TYPE_CHECKING, not at runtime: {plugins_runtime}"
    )
