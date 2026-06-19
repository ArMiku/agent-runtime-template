"""list_dir: discovery — entry listing with type/size + an entry-count cap."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_runtime.core.function_tool_executor import FunctionToolExecutor
from agent_runtime.core.run_context import ContextWrapper
from agent_runtime.extensions.fs.fs_tools import (
    MAX_ENTRIES,
    build_list_dir_tool,
    list_dir_entries,
)


def _ctx() -> ContextWrapper:
    return ContextWrapper(context=None, messages=[])


def _make_file(tmp_path, name: str, content: str | bytes) -> tuple[Path, Path]:
    root = tmp_path / "root"
    root.mkdir(exist_ok=True)
    f = root / name
    f.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, bytes):
        f.write_bytes(content)
    else:
        f.write_text(content, encoding="utf-8")
    return f, root


def test_lists_files_and_dirs_with_types(tmp_path):
    _make_file(tmp_path, "a.txt", "x")
    (tmp_path / "root" / "sub").mkdir()
    root = tmp_path / "root"

    text = list_dir_entries(str(root), allowed_roots=[root])

    assert "a.txt\t[file" in text
    assert "sub\t[dir]" in text


def test_includes_file_size(tmp_path):
    _make_file(tmp_path, "a.txt", "hello")  # 5 bytes
    root = tmp_path / "root"

    text = list_dir_entries(str(root), allowed_roots=[root])

    assert "[file, 5B]" in text


def test_entry_cap_truncates_and_marks(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    for i in range(MAX_ENTRIES + 50):
        (root / f"f{i:05d}.txt").write_text("x", encoding="utf-8")

    text = list_dir_entries(str(root), allowed_roots=[root])

    assert "已截断" in text
    assert "还有 50 条" in text
    body = text.split("---", 1)[0]
    assert body.count("\t[file") <= MAX_ENTRIES


def test_empty_dir_reports_zero(tmp_path):
    root = tmp_path / "root"
    root.mkdir()

    text = list_dir_entries(str(root), allowed_roots=[root])

    assert "共 0 条" in text


def test_path_outside_fence_rejected(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    root = tmp_path / "root"
    root.mkdir()

    with pytest.raises(PermissionError):
        list_dir_entries(str(outside), allowed_roots=[root])


def test_missing_dir_raises_filenotfound(tmp_path):
    root = tmp_path / "root"
    root.mkdir()

    with pytest.raises(FileNotFoundError):
        list_dir_entries(str(root / "nope"), allowed_roots=[root])


def test_file_path_rejected(tmp_path):
    f, root = _make_file(tmp_path, "a.txt", "x")

    with pytest.raises(ValueError, match="文件"):
        list_dir_entries(str(f), allowed_roots=[root])


async def test_handler_returns_string(tmp_path):
    _make_file(tmp_path, "a.txt", "x")
    root = tmp_path / "root"
    tool = build_list_dir_tool([root])

    result = await tool.handler(_ctx(), path=str(root))

    assert isinstance(result, str)
    assert "a.txt\t[file" in result


async def test_executor_normalizes_string_result(tmp_path):
    _make_file(tmp_path, "a.txt", "x")
    root = tmp_path / "root"
    tool = build_list_dir_tool([root])
    executor = FunctionToolExecutor(provider=None)  # type: ignore[arg-type]

    outputs = [item async for item in executor.execute(tool, _ctx(), path=str(root))]

    assert len(outputs) == 1
    assert "a.txt\t[file" in outputs[0].content[0].text


def test_list_dir_params_and_surface():
    tool = build_list_dir_tool([Path("/tmp/never")])

    props = tool.parameters["properties"]
    assert set(props) == {"path"}
    assert tool.parameters["required"] == ["path"]
    assert tool.name == "list_dir"

    blob = (tool.name + " " + tool.description + " " + repr(tool.parameters)).lower()
    for banned in ("astrbot", "sandbox", "neo"):
        assert banned not in blob
