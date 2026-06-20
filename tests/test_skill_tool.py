"""Phase 3: the ``Skill(name)`` tool loads instructions by name, rejects bad input,
and exposes no path argument."""

from __future__ import annotations

import pytest

from agent_runtime.core.function_tool_executor import FunctionToolExecutor
from agent_runtime.core.run_context import ContextWrapper
from agent_runtime.extensions.skills.skill_manager import SkillManager
from agent_runtime.extensions.skills.skill_tool import build_skill_tool

GREET_BODY = "---\nname: greet\ndescription: Say hi.\n---\n# Greet\nWave politely.\n"


def _manager(tmp_path, *, name="greet", body=GREET_BODY):
    skill_dir = tmp_path / "skills" / name
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(body, encoding="utf-8")
    return SkillManager(skills_root=str(tmp_path / "skills"))


def _ctx() -> ContextWrapper:
    return ContextWrapper(context=None, messages=[])


async def test_skill_tool_loads_instructions(tmp_path):
    tool = build_skill_tool(_manager(tmp_path))

    result = await tool.handler(_ctx(), name="greet")

    assert result == GREET_BODY


async def test_skill_tool_executor_normalizes_string_result(tmp_path):
    """The executor wraps the bare-string return into a CallToolResult (runner contract)."""
    tool = build_skill_tool(_manager(tmp_path))
    executor = FunctionToolExecutor(provider=None)  # type: ignore[arg-type]

    outputs = [item async for item in executor.execute(tool, _ctx(), name="greet")]

    assert len(outputs) == 1
    text = outputs[0].content[0].text
    assert text == GREET_BODY


@pytest.mark.parametrize("bad", ["../other", "a!!", "with space", ""])
async def test_skill_tool_rejects_invalid_name(tmp_path, bad):
    tool = build_skill_tool(_manager(tmp_path))

    with pytest.raises(ValueError):
        await tool.handler(_ctx(), name=bad)


async def test_skill_tool_rejects_unknown(tmp_path):
    tool = build_skill_tool(_manager(tmp_path))

    with pytest.raises(FileNotFoundError):
        await tool.handler(_ctx(), name="nope")


async def test_skill_tool_rejects_inactive(tmp_path):
    import json

    manager = _manager(tmp_path)  # creates an active greet skill
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    config_path = data_dir / "skills.json"
    config_path.write_text(json.dumps({"skills": {"greet": {"active": False}}}), encoding="utf-8")
    manager.config_path = str(config_path)
    tool = build_skill_tool(manager)

    with pytest.raises(PermissionError):
        await tool.handler(_ctx(), name="greet")


def test_skill_tool_params_have_only_name():
    tool = build_skill_tool(_manager_via_stub())

    properties = tool.parameters["properties"]
    assert set(properties) == {"name"}
    assert "path" not in properties
    assert "rel_path" not in properties
    assert tool.parameters.get("required") == ["name"]
    assert tool.name == "Skill"


def test_skill_tool_public_surface_is_neutral():
    tool = build_skill_tool(_manager_via_stub())
    blob = (tool.name + " " + tool.description + " " + repr(tool.parameters)).lower()
    for banned in ("sandbox", "neo"):
        assert banned not in blob


def _manager_via_stub() -> SkillManager:
    """A manager that never touches the filesystem — only its type is needed to build the tool."""
    import os
    import tempfile

    return SkillManager(skills_root=os.path.join(tempfile.mkdtemp(), "skills"))
