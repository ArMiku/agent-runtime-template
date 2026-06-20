"""Tier 3 tests: ``build_local_agent`` → ``LocalAgent``.

The async factory wraps Tier 2 plus the request/context/executor/runner and
``runner.reset`` into a ready-to-run object. These assert the wrapper is complete (the
caller never hand-writes ``runner.reset``), a drop-in skill runs through end to end,
``provider`` stays a caller responsibility, ``**runner_kwargs`` pass through to
``runner.reset``, ``run()`` returns the final response, and ``aclose()`` owns the
self-built executor.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from agent_runtime.local_runtime import LocalAgent, build_local_agent
from agent_runtime.provider.entities import LLMResponse

from .fakes import FakeProvider, llm_text, llm_tool_call


def _seed_skill(root: Path, name: str, desc: str) -> None:
    skill_dir = root / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {desc}\n---\n# {name}\n\nGreeting instructions.\n",
        encoding="utf-8",
    )


@pytest.fixture()
def skills_root(tmp_path, monkeypatch) -> Path:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setenv("AGENT_RUNTIME_DATA_DIR", str(data_dir))
    root = tmp_path / "skills"
    root.mkdir()
    return root


# --- 2.1 wrapper holds the full machinery, caller never wrote reset ----------


async def test_build_local_agent_holds_machinery(skills_root):
    provider = FakeProvider([llm_text("done")])
    agent = await build_local_agent(provider, prompt="hi", skills_root=str(skills_root))

    assert isinstance(agent, LocalAgent)
    assert agent.provider is provider
    assert agent.basics is not None
    assert agent.runner is not None
    assert agent.request is not None
    assert agent.run_context is not None
    # The runner was already reset during construction: it has the bundle's tools wired.
    assert agent.request.func_tool is agent.basics.tools


# --- 2.2 one-shot run-through --------------------------------------------------


async def test_run_through(skills_root):
    _seed_skill(skills_root, "greet", "Greet the user.")
    provider = FakeProvider([llm_tool_call("Skill", {"name": "greet"}), llm_text("hello")])

    agent = await build_local_agent(provider, prompt="greet me", skills_root=str(skills_root))
    final = await agent.run(max_step=10)

    assert final is not None
    assert isinstance(final, LLMResponse)

    system_msg = agent.run_context.messages[0]
    assert system_msg.role == "system"
    assert "greet" in system_msg.content
    tool_results = [
        str(getattr(m, "content", "")) for m in agent.run_context.messages if getattr(m, "role", None) == "tool"
    ]
    assert any("Greeting instructions." in t for t in tool_results)


# --- 2.3 provider supplied by caller -----------------------------------------


def test_provider_is_required_first_param():
    sig = inspect.signature(build_local_agent)
    params = list(sig.parameters.values())
    assert params[0].name == "provider"
    # Required (no default) and positional.
    assert params[0].default is inspect.Parameter.empty
    # The factory must not construct a Provider itself.
    src = inspect.getsource(build_local_agent)
    assert "Provider(" not in src


# --- 2.4 **runner_kwargs passthrough -----------------------------------------


async def test_runner_kwargs_passthrough(skills_root, monkeypatch):
    captured: dict = {}

    from agent_runtime.core.runners.tool_loop_agent_runner import ToolLoopAgentRunner

    orig_reset = ToolLoopAgentRunner.reset

    async def _spy_reset(self, *args, **kwargs):
        captured.update(kwargs)
        return await orig_reset(self, *args, **kwargs)

    monkeypatch.setattr(ToolLoopAgentRunner, "reset", _spy_reset)

    fb = FakeProvider([llm_text("x")])
    provider = FakeProvider([llm_text("done")])
    await build_local_agent(
        provider,
        prompt="hi",
        skills_root=str(skills_root),
        fallback_providers=[fb],
    )

    assert captured.get("fallback_providers") == [fb]


# --- 2.5 run() returns get_final_llm_resp() ----------------------------------


async def test_run_returns_final_resp(skills_root):
    provider = FakeProvider([llm_text("the-answer")])
    agent = await build_local_agent(provider, prompt="hi", skills_root=str(skills_root))

    final = await agent.run(max_step=5)

    assert agent.done() is True
    assert final is agent.get_final_llm_resp()
    assert final is not None
    assert final.completion_text == "the-answer"


# --- 2.5b aclose() owns the self-built executor ------------------------------


async def test_aclose_closes_self_built_executor(skills_root, monkeypatch):
    closed: list[int] = []

    class _ClosingExecutor:
        def __init__(self, provider, *args, **kwargs) -> None:
            self.provider = provider

        async def execute(self, *args, **kwargs):  # pragma: no cover - not driven here
            if False:
                yield None

        async def aclose(self) -> None:
            closed.append(1)

    monkeypatch.setattr("agent_runtime.local_runtime.FunctionToolExecutor", _ClosingExecutor)

    provider = FakeProvider([llm_text("done")])
    agent = await build_local_agent(provider, prompt="hi", skills_root=str(skills_root))

    await agent.aclose()
    assert closed == [1]


async def test_aclose_noop_when_executor_lacks_aclose(skills_root, monkeypatch):
    class _NoCloseExecutor:
        def __init__(self, provider, *args, **kwargs) -> None:
            self.provider = provider

        async def execute(self, *args, **kwargs):  # pragma: no cover
            if False:
                yield None

    monkeypatch.setattr("agent_runtime.local_runtime.FunctionToolExecutor", _NoCloseExecutor)

    provider = FakeProvider([llm_text("done")])
    agent = await build_local_agent(provider, prompt="hi", skills_root=str(skills_root))

    # Must not raise.
    await agent.aclose()
