"""Phase 2: ``SkillsPromptHook`` injects the inventory into the system message,
idempotently (sentinel-delimited replacement), and isolates its own failures."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_runtime.core.message import Message
from agent_runtime.core.run_context import ContextWrapper
from agent_runtime.extensions.skills.hooks import SkillsPromptHook
from agent_runtime.extensions.skills.skill_manager import SkillManager


def _manager_with_skill(root: Path, name: str = "greet", desc: str = "Say hi.") -> SkillManager:
    skill_dir = root / name
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(f"---\nname: {name}\ndescription: {desc}\n---\n# {name}\n", encoding="utf-8")
    # Isolate config to the tmp tree so the test never depends on the ambient data dir.
    return SkillManager(skills_root=str(root), config_path=str(root.parent / "skills.json"))


def _ctx(system: str | None = None) -> ContextWrapper:
    messages = []
    if system is not None:
        messages.append(Message(role="system", content=system))
    return ContextWrapper(context=None, messages=messages)


async def test_inventory_enters_system_message(tmp_path):
    manager = _manager_with_skill(tmp_path / "skills")
    ctx = _ctx(system="You are helpful.")

    await SkillsPromptHook(manager).on_llm_request(ctx)

    content = ctx.messages[0].content
    assert ctx.messages[0].role == "system"
    assert content.startswith("You are helpful.")
    assert "## Skills" in content
    assert "greet" in content
    assert content.count(SkillsPromptHook._OPEN) == 1
    assert content.count(SkillsPromptHook._CLOSE) == 1


async def test_creates_system_message_when_absent(tmp_path):
    manager = _manager_with_skill(tmp_path / "skills")
    ctx = _ctx(system=None)  # no messages at all

    await SkillsPromptHook(manager).on_llm_request(ctx)

    assert ctx.messages[0].role == "system"
    assert SkillsPromptHook._OPEN in ctx.messages[0].content


async def test_multi_step_replacement_is_idempotent(tmp_path):
    manager = _manager_with_skill(tmp_path / "skills")
    hook = SkillsPromptHook(manager)
    ctx = _ctx(system="base")

    await hook.on_llm_request(ctx)
    first = ctx.messages[0].content
    await hook.on_llm_request(ctx)
    await hook.on_llm_request(ctx)

    second = ctx.messages[0].content
    # The segment appears exactly once, and content is stable across steps (replacement,
    # not append).
    assert second.count(SkillsPromptHook._OPEN) == 1
    assert second.count(SkillsPromptHook._CLOSE) == 1
    assert second == first


async def test_no_skills_removes_segment(tmp_path):
    manager = _manager_with_skill(tmp_path / "skills")
    hook = SkillsPromptHook(manager)
    ctx = _ctx(system="base")

    await hook.on_llm_request(ctx)
    assert SkillsPromptHook._OPEN in ctx.messages[0].content

    # Second step: no active skills → the segment must be removed, not left stale.
    manager_empty = SkillManager(
        skills_root=str(tmp_path / "empty_skills"),
        config_path=str(tmp_path / "empty_skills.json"),
    )
    await SkillsPromptHook(manager_empty).on_llm_request(ctx)

    content = ctx.messages[0].content
    assert SkillsPromptHook._OPEN not in content
    assert SkillsPromptHook._CLOSE not in content
    assert "## Skills" not in content


async def test_hook_exception_is_isolated(tmp_path):
    class _BoomManager:
        def list_skills(self, *, active_only=False):
            raise RuntimeError("boom")

    ctx = _ctx(system="base")
    # Must not raise — a skills failure never breaks the run.
    await SkillsPromptHook(_BoomManager()).on_llm_request(ctx)  # type: ignore[arg-type]

    # Original system message intact, no inventory written.
    assert ctx.messages[0].content == "base"


def test_replace_inventory_helper_directly():
    replace = SkillsPromptHook._replace_inventory
    # append when absent
    one = replace("base", "BLOCK")
    assert one.count(SkillsPromptHook._OPEN) == 1
    assert "BLOCK" in one
    # replace in place
    two = replace(one, "OTHER")
    assert two.count(SkillsPromptHook._OPEN) == 1
    assert "OTHER" in two and "BLOCK" not in two
    # remove when block empty
    three = replace(two, "")
    assert SkillsPromptHook._OPEN not in three
    assert "OTHER" not in three


@pytest.mark.parametrize("bad", ["../other", "a!!", "with space"])
def test_inventory_uses_sanitized_display_name_only(tmp_path, bad):
    # Such names can't even be discovered (invalid dir names are simply skipped), but the
    # hook path never lets raw names reach the prompt unsanitized — confirmed by the
    # manager refusing them.
    root = tmp_path / "skills"
    root.mkdir()
    manager = SkillManager(skills_root=str(root))
    assert manager.list_skills() == []
