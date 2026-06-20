"""Assembly + architectural-boundary tests for plan-execute (tasks 8.4, 9.1–9.3).

* 8.4 — ``runner_type`` selects the runner and is orthogonal to ``include_planning``.
* 9.1 — ``core/runners/base.py`` is unmodified; ``PlanExecuteRunner`` plugs in via the
  existing abstract methods only.
* 9.2 — the default ``react`` path still assembles ``ToolLoopAgentRunner`` unchanged.
* 9.3 — ``plan_execute_runner`` depends one-way on ``tool_loop_agent_runner`` and reuses only
  the planning extension's pure storage/data functions, never its hook/tool implementations.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from agent_runtime.core.runners.base import BaseAgentRunner
from agent_runtime.core.runners.plan_execute_runner import PlanExecuteRunner
from agent_runtime.core.runners.tool_loop_agent_runner import ToolLoopAgentRunner
from agent_runtime.local_runtime import LocalAgent, build_local_agent

from .fakes import FakeProvider, llm_text

_RUNNERS_DIR = Path(__file__).resolve().parents[1] / "agent_runtime" / "core" / "runners"
_BASE_PY = _RUNNERS_DIR / "base.py"
_PLAN_EXECUTE_PY = _RUNNERS_DIR / "plan_execute_runner.py"
_TOOL_LOOP_PY = _RUNNERS_DIR / "tool_loop_agent_runner.py"


@pytest.fixture()
def skills_root(tmp_path, monkeypatch) -> Path:
    """Isolated skills root + data dir so assembly never touches the ambient data dir."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setenv("AGENT_RUNTIME_DATA_DIR", str(data_dir))
    root = tmp_path / "skills"
    root.mkdir()
    return root


# --- 8.4 runner_type selects the runner; orthogonal to include_planning --------


async def test_default_react_assembles_tool_loop_runner(skills_root):
    agent = await build_local_agent(
        FakeProvider([llm_text("hi")]), prompt="hi", skills_root=str(skills_root), include_fs=False
    )
    assert isinstance(agent.runner, ToolLoopAgentRunner)
    # The widened annotation accepts either runner.
    assert isinstance(agent.runner, BaseAgentRunner)


async def test_plan_execute_assembles_plan_execute_runner(skills_root):
    agent = await build_local_agent(
        FakeProvider([llm_text("hi")]),
        prompt="hi",
        skills_root=str(skills_root),
        include_fs=False,
        runner_type="plan_execute",
    )
    assert isinstance(agent.runner, PlanExecuteRunner)
    assert isinstance(agent.runner, BaseAgentRunner)
    # LocalAgent.runner is annotated as the widened BaseAgentRunner (not ToolLoopAgentRunner).
    assert LocalAgent.__dataclass_fields__["runner"].type in (BaseAgentRunner, "BaseAgentRunner")


async def test_runner_type_and_include_planning_are_orthogonal(skills_root):
    """All four combinations assemble without error."""
    combos = [
        ("react", False),
        ("react", True),
        ("plan_execute", False),
        ("plan_execute", True),
    ]
    for runner_type, include_planning in combos:
        agent = await build_local_agent(
            FakeProvider([llm_text("hi")]),
            prompt="hi",
            skills_root=str(skills_root),
            include_fs=False,
            runner_type=runner_type,
            include_planning=include_planning,
        )
        expected = PlanExecuteRunner if runner_type == "plan_execute" else ToolLoopAgentRunner
        assert isinstance(agent.runner, expected), f"{runner_type}/{include_planning} → {type(agent.runner)}"


async def test_unknown_runner_type_rejected(skills_root):
    with pytest.raises(ValueError, match="Unknown runner_type"):
        await build_local_agent(
            FakeProvider([llm_text("hi")]),
            prompt="hi",
            skills_root=str(skills_root),
            include_fs=False,
            runner_type="bogus",
        )


# --- 9.1 base.py unmodified; PlanExecuteRunner plugs in via abstract methods ---


def test_base_py_is_unmodified_vs_git():
    """9.1: ``core/runners/base.py`` is byte-identical to the committed version."""
    try:
        diff = subprocess.run(
            ["git", "diff", "--no-color", "HEAD", "--", str(_BASE_PY)],
            capture_output=True,
            text=True,
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        pytest.skip("git not available or repo state unreadable")
    assert diff.stdout == "", "core/runners/base.py must not be modified by this change"


def test_plan_execute_runner_is_concrete_via_existing_abstract_methods():
    """9.1: PlanExecuteRunner instantiates (so it implements every BaseAgentRunner abstract
    method) without any change to the contract."""
    runner = PlanExecuteRunner()
    for method in ("reset", "step", "step_until_done", "done", "get_final_llm_resp"):
        assert callable(getattr(runner, method)), f"missing {method}"


# --- 9.2 default react path still behaves as before --------------------------


async def test_react_path_runs_unchanged(skills_root):
    """9.2: the default react assembly still drives a scripted run to DONE as before."""
    agent = await build_local_agent(
        FakeProvider([llm_text("done")]),
        prompt="go",
        skills_root=str(skills_root),
        include_fs=False,
    )
    async for _ in agent.runner.step_until_done(10):
        pass
    assert agent.runner.done()
    assert agent.runner.get_final_llm_resp().completion_text == "done"


# --- 9.3 one-way dependency + pure-function reuse (no hook/tool coupling) -----


def test_tool_loop_does_not_import_plan_execute():
    """9.3: the dependency is one-way — tool_loop never reaches back into plan_execute."""
    source = _TOOL_LOOP_PY.read_text(encoding="utf-8")
    assert "plan_execute" not in source


def test_plan_execute_reuses_only_pure_planning_storage():
    """9.3: plan_execute_runner imports the planning extension's store/entities (pure storage
    + data) but never its hook (``todo_hook``) or tool (``todo_tool``) implementations."""
    source = _PLAN_EXECUTE_PY.read_text(encoding="utf-8")
    # Allowed pure-function reuse.
    assert "extensions.planning.store" in source
    assert "extensions.planning.entities" in source
    # Forbidden coupling to hook/tool implementations.
    assert "extensions.planning.todo_hook" not in source
    assert "extensions.planning.todo_tool" not in source
    assert "build_write_todos_tool" not in source
    # The sub-hook chain is built by the assembly layer, not imported here.
    assert "SkillsPromptHook" not in source
    assert "PlanningHook" not in source


# --- 8.2 liveness params pass through build_local_agent to the runner ----------


async def test_liveness_params_reach_plan_execute_runner(skills_root):
    """8.2: max_step / per_call_timeout_s / per_turn_deadline_s flow through the assembly's
    ``**runner_kwargs`` to the PlanExecuteRunner instance fields."""
    agent = await build_local_agent(
        FakeProvider([llm_text("hi")]),
        prompt="hi",
        skills_root=str(skills_root),
        include_fs=False,
        runner_type="plan_execute",
        max_step=321,
        per_call_timeout_s=12.5,
        per_turn_deadline_s=99.0,
    )
    runner = agent.runner
    assert isinstance(runner, PlanExecuteRunner)
    assert runner._max_step == 321
    assert runner._per_call_timeout_s == 12.5
    assert runner._per_turn_deadline_s == 99.0


async def test_plan_execute_runner_has_disaster_loose_defaults(skills_root):
    """8.2: with no liveness kwargs, the runner carries its disaster-loose defaults."""
    from agent_runtime.core.runners import plan_execute_runner as m

    agent = await build_local_agent(
        FakeProvider([llm_text("hi")]),
        prompt="hi",
        skills_root=str(skills_root),
        include_fs=False,
        runner_type="plan_execute",
    )
    runner = agent.runner
    assert runner._max_step == m._DEFAULT_MAX_STEP
    assert runner._per_call_timeout_s == m._DEFAULT_PER_CALL_TIMEOUT_S
    assert runner._per_turn_deadline_s == m._DEFAULT_PER_TURN_DEADLINE_S
