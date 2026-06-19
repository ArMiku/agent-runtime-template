"""Tier 2 tests: ``build_local_agent_basics`` — the composition root that wires
skills + fs + plugins into a ``{skill_manager, tools, hooks}`` bundle (design.md §4).

These assert the bundle's *shape* (no provider/runner/request), that a drop-in skill is
auto-usable through the real runner, that plugin contributions merge, that ``include_fs``
gates the fs tools (and their import), that the bundle stays tweakable, and that the
one-shot path is observably equivalent to hand-wiring the Tier-1 builders.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from agent_runtime.core.hooks import BaseAgentRunHooks
from agent_runtime.core.run_context import ContextWrapper
from agent_runtime.core.runners.tool_loop_agent_runner import ToolLoopAgentRunner
from agent_runtime.core.session_context import SessionContext
from agent_runtime.core.tool import FunctionTool, ToolSet
from agent_runtime.local_runtime import LocalAgentBasics, build_local_agent_basics
from agent_runtime.tests.fakes import FakeProvider, llm_text, llm_tool_call


def _seed_skill(root: Path, name: str, desc: str) -> None:
    skill_dir = root / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {desc}\n---\n# {name}\n\nGreeting instructions.\n",
        encoding="utf-8",
    )


@pytest.fixture()
def skills_root(tmp_path, monkeypatch) -> Path:
    """An isolated skills root + data dir so tests never touch the ambient data dir."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setenv("AGENT_RUNTIME_DATA_DIR", str(data_dir))
    root = tmp_path / "skills"
    root.mkdir()
    return root


async def _drive(provider, tools, hooks, prompt="hi", max_step=8):
    """Drive a fresh runner one full run with the bundle's tools + hooks."""
    from agent_runtime.core import FunctionToolExecutor
    from agent_runtime.provider.entities import ProviderRequest

    request = ProviderRequest(prompt=prompt, system_prompt="", func_tool=tools)
    run_context = ContextWrapper(context=SessionContext(session_id="t"), messages=[])
    runner = ToolLoopAgentRunner()
    await runner.reset(
        provider=provider,
        request=request,
        run_context=run_context,
        tool_executor=FunctionToolExecutor(provider),
        agent_hooks=hooks,
    )
    async for _ in runner.step_until_done(max_step=max_step):
        pass
    return runner, run_context


# --- 1.1 bundle shape --------------------------------------------------------


def test_basics_returns_bundle_without_provider_runner_request(skills_root):
    basics = build_local_agent_basics(skills_root=str(skills_root))

    assert isinstance(basics, LocalAgentBasics)
    from agent_runtime.extensions.skills import SkillManager

    assert isinstance(basics.skill_manager, SkillManager)
    assert isinstance(basics.tools, ToolSet)
    assert isinstance(basics.hooks, BaseAgentRunHooks)

    # Tier-2 must NOT carry provider/runner/request — that's Tier 3's job.
    fields = set(vars(basics))
    assert "provider" not in fields
    assert "runner" not in fields
    assert "request" not in fields


# --- 1.2 drop-in skill auto-usable through the real runner -------------------


async def test_dropin_skill_auto_usable(skills_root):
    _seed_skill(skills_root, "greet", "Greet the user.")
    basics = build_local_agent_basics(skills_root=str(skills_root))

    assert "Skill" in basics.tools.names()

    # Script: call Skill(name="greet") -> finish.
    provider = FakeProvider(
        [llm_tool_call("Skill", {"name": "greet"}), llm_text("done")]
    )
    runner, run_context = await _drive(provider, basics.tools, basics.hooks, prompt="greet me")

    # Leading system message carries the inventory (the greet skill is listed).
    system_msg = run_context.messages[0]
    assert system_msg.role == "system"
    assert "greet" in system_msg.content
    assert "## Skills" in system_msg.content

    # Skill("greet") ran through the real tool path and returned the SKILL.md body.
    tool_results = [
        str(getattr(m, "content", "")) for m in run_context.messages if getattr(m, "role", None) == "tool"
    ]
    assert any("Greeting instructions." in t for t in tool_results)


# --- 1.3 plugin contributions merge ------------------------------------------


async def test_plugin_contributions_merge(skills_root, tmp_path):
    plugin_skill_root = tmp_path / "plugin_skills"
    _seed_skill(plugin_skill_root, "pskill", "A plugin-bundled skill.")

    fired: list[str] = []

    async def _hk(run_context) -> None:
        fired.append("hk")

    async def _t_handler(run_context, **kwargs) -> str:
        return "tool-T-result"

    custom_tool = FunctionTool(
        name="T",
        description="a plugin tool",
        parameters={"type": "object", "properties": {}},
        handler=_t_handler,
    )

    from agent_runtime.extensions.plugins.base import Plugin
    from agent_runtime.extensions.plugins.contributions import PluginContribution

    class _P(Plugin):
        name, author, desc, version = "p", "a", "d", "1"

    contribution = PluginContribution(
        plugin=_P.__new__(_P),
        tools=[custom_tool],
        hook_methods={"on_llm_request": [_hk]},
        skill_dirs=[plugin_skill_root],
    )

    basics = build_local_agent_basics(skills_root=str(skills_root), contributions=[contribution])

    # Plugin tool merged alongside Skill.
    assert "T" in basics.tools.names()
    assert "Skill" in basics.tools.names()
    # Plugin-bundled skill discoverable.
    assert "pskill" in [s.name for s in basics.skill_manager.list_skills()]
    # Plugin hook fires (coexisting with SkillsPromptHook, neither blocks the other).
    ctx = ContextWrapper(context=SessionContext(session_id="t"), messages=[])
    await basics.hooks.on_llm_request(ctx)
    assert fired == ["hk"]


# --- 1.4 include_fs toggle (and no fs import) --------------------------------


def test_include_fs_false_omits_fs_tools_and_import(skills_root):
    sys.modules.pop("agent_runtime.extensions.fs", None)
    sys.modules.pop("agent_runtime.extensions.fs.fs_tools", None)

    basics = build_local_agent_basics(skills_root=str(skills_root), include_fs=False)

    names = basics.tools.names()
    assert "list_dir" not in names
    assert "read_file" not in names
    # The fs extension was never imported by the factory on this path.
    assert "agent_runtime.extensions.fs" not in sys.modules


def test_include_fs_true_adds_fs_tools(skills_root):
    basics = build_local_agent_basics(skills_root=str(skills_root), include_fs=True)
    names = basics.tools.names()
    assert "list_dir" in names
    assert "read_file" in names


# --- 1.5 bundle is tweakable after assembly ----------------------------------


def test_bundle_is_tweakable(skills_root):
    basics = build_local_agent_basics(skills_root=str(skills_root), include_fs=False)

    async def _h(run_context, **kwargs) -> str:
        return "custom"

    custom = FunctionTool(
        name="custom", description="c", parameters={"type": "object", "properties": {}}, handler=_h
    )
    basics.tools.add_tool(custom)
    assert "custom" in basics.tools.names()

    other = BaseAgentRunHooks()
    basics.hooks = other
    assert basics.hooks is other


# --- 1.6 Tier 2 (one-shot) vs Tier 1 (hand-wired) equivalence ----------------


async def test_oneshot_equivalent_to_manual_wiring(skills_root):
    _seed_skill(skills_root, "greet", "Greet the user.")

    # --- Tier 2: one-shot ---
    basics = build_local_agent_basics(skills_root=str(skills_root), include_fs=True)

    # --- Tier 1: hand-wired with the same builders ---
    from agent_runtime.extensions.fs import build_fs_tools
    from agent_runtime.extensions.skills import (
        SkillManager,
        SkillsPromptHook,
        build_skill_tool,
    )

    mgr = SkillManager(skills_root=str(skills_root))
    manual_tools = ToolSet([build_skill_tool(mgr)])
    manual_tools.merge(ToolSet(build_fs_tools(allowed_roots=[mgr.skills_root])))
    manual_hooks = SkillsPromptHook(mgr)

    # Tool-name sets equal.
    assert set(basics.tools.names()) == set(manual_tools.names())

    # Injected inventory content equal.
    ctx_a = ContextWrapper(context=SessionContext(session_id="a"), messages=[])
    ctx_b = ContextWrapper(context=SessionContext(session_id="b"), messages=[])
    await basics.hooks.on_llm_request(ctx_a)
    await manual_hooks.on_llm_request(ctx_b)
    assert ctx_a.messages[0].content == ctx_b.messages[0].content
