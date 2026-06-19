"""build_fs_tools: assembly — two named tools + default/custom allowed roots."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from agent_runtime.core.run_context import ContextWrapper
from agent_runtime.extensions.fs import build_fs_tools


def _ctx() -> ContextWrapper:
    return ContextWrapper(context=None, messages=[])


def test_build_fs_tools_returns_two_named_tools():
    tools = build_fs_tools()

    assert [t.name for t in tools] == ["list_dir", "read_file"]


async def test_default_roots_accepts_skills_dir_file(tmp_path, monkeypatch):
    """Omitted allowed_roots -> the skills directory is the default fence."""
    monkeypatch.setenv("AGENT_RUNTIME_DATA_DIR", str(tmp_path))
    skills = Path(os.path.realpath(str(tmp_path / "skills")))
    f = skills / "greet" / "SKILL.md"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text("# hi\n", encoding="utf-8")

    tools = build_fs_tools()  # default roots -> skills dir
    read_tool = next(t for t in tools if t.name == "read_file")

    result = await read_tool.handler(_ctx(), path=str(f))

    assert "1\t# hi" in result


async def test_custom_roots_respected(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    inside = root / "a.txt"
    inside.write_text("x\n", encoding="utf-8")
    outside = tmp_path / "o.txt"
    outside.write_text("y\n", encoding="utf-8")

    tools = build_fs_tools([root])
    read_tool = next(t for t in tools if t.name == "read_file")

    ok = await read_tool.handler(_ctx(), path=str(inside))
    assert "1\tx" in ok

    with pytest.raises(PermissionError):
        await read_tool.handler(_ctx(), path=str(outside))


def test_build_fs_tools_surface_is_neutral():
    tools = build_fs_tools([Path("/tmp/never")])
    blob = " ".join(t.name + " " + t.description + " " + repr(t.parameters) for t in tools).lower()
    for banned in ("astrbot", "sandbox", "neo"):
        assert banned not in blob
