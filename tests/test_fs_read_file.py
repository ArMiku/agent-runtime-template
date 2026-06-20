"""read_file: line-numbered, dual-cap chunked reads with metadata + guardrails."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_runtime.core.function_tool_executor import FunctionToolExecutor
from agent_runtime.core.run_context import ContextWrapper
from agent_runtime.extensions.fs.fs_tools import (
    MAX_BYTES,
    build_read_file_tool,
    read_file_text,
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


# --- line numbers + offset --------------------------------------------------


def test_emits_forced_line_numbers(tmp_path):
    f, root = _make_file(tmp_path, "a.txt", "alpha\nbeta\ngamma\n")

    text = read_file_text(str(f), start_line=1, line_count=10, allowed_roots=[root])

    assert "1\talpha" in text
    assert "2\tbeta" in text
    assert "3\tgamma" in text


def test_start_line_offset_skips_earlier_lines(tmp_path):
    f, root = _make_file(tmp_path, "a.txt", "alpha\nbeta\ngamma\n")

    text = read_file_text(str(f), start_line=2, line_count=10, allowed_roots=[root])

    assert "2\tbeta" in text
    assert "alpha" not in text  # line 1 skipped


# --- chunking + remaining + fully read --------------------------------------


def test_line_count_chunk_reports_remaining(tmp_path):
    f, root = _make_file(tmp_path, "a.txt", "l1\nl2\nl3\nl4\nl5\n")

    text = read_file_text(str(f), start_line=1, line_count=2, allowed_roots=[root])

    assert "1\tl1" in text
    assert "2\tl2" in text
    assert "3\tl3" not in text  # not in this chunk
    assert "剩余 3 行未读" in text
    assert "start_line=3" in text  # resume hint


def test_full_read_reports_done(tmp_path):
    f, root = _make_file(tmp_path, "a.txt", "l1\nl2\nl3\n")

    text = read_file_text(str(f), start_line=1, line_count=50, allowed_roots=[root])

    assert "已读完" in text
    assert "剩余" not in text


def test_mid_chunk_resume_picks_up_where_left_off(tmp_path):
    f, root = _make_file(tmp_path, "a.txt", "l1\nl2\nl3\nl4\nl5\n")

    first = read_file_text(str(f), start_line=1, line_count=2, allowed_roots=[root])
    second = read_file_text(str(f), start_line=3, line_count=2, allowed_roots=[root])

    assert "2\tl2" in first and "3\tl3" not in first
    assert "3\tl3" in second and "4\tl4" in second
    assert "5\tl5" not in second  # only 2 lines requested


# --- byte cap (single huge line) -------------------------------------------


def test_huge_single_line_is_byte_truncated(tmp_path):
    f, root = _make_file(tmp_path, "min.txt", "x" * (MAX_BYTES + 5000))

    text = read_file_text(str(f), start_line=1, line_count=10, allowed_roots=[root])

    assert "截断" in text
    # Bounded well under the raw line size; the marker adds a few bytes.
    assert len(text) < MAX_BYTES + 4096


# --- metadata ---------------------------------------------------------------


def test_metadata_present(tmp_path):
    f, root = _make_file(tmp_path, "a.txt", "l1\nl2\n")

    text = read_file_text(str(f), start_line=1, line_count=10, allowed_roots=[root])

    assert "共 2 行" in text
    assert "encoding: utf-8" in text
    assert "mtime:" in text


# --- guardrails -------------------------------------------------------------


def test_rejects_non_utf8(tmp_path):
    f, root = _make_file(tmp_path, "bin.txt", b"\x80\x81\xff\xfe not utf8")

    with pytest.raises(ValueError, match="UTF-8"):
        read_file_text(str(f), start_line=1, line_count=10, allowed_roots=[root])


def test_rejects_binary_nul_byte(tmp_path):
    f, root = _make_file(tmp_path, "bin.dat", b"hello\x00world")

    with pytest.raises(ValueError, match="二进制"):
        read_file_text(str(f), start_line=1, line_count=10, allowed_roots=[root])


def test_out_of_range_start_line_is_friendly(tmp_path):
    f, root = _make_file(tmp_path, "a.txt", "l1\nl2\nl3\n")

    with pytest.raises(ValueError, match="1..3"):
        read_file_text(str(f), start_line=5000, line_count=10, allowed_roots=[root])


def test_start_line_below_one_is_rejected(tmp_path):
    f, root = _make_file(tmp_path, "a.txt", "l1\n")

    with pytest.raises(ValueError):
        read_file_text(str(f), start_line=0, line_count=10, allowed_roots=[root])


def test_path_outside_fence_rejected(tmp_path):
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    root = tmp_path / "root"
    root.mkdir()

    with pytest.raises(PermissionError):
        read_file_text(str(outside), start_line=1, line_count=10, allowed_roots=[root])


def test_missing_file_raises_filenotfound(tmp_path):
    root = tmp_path / "root"
    root.mkdir()

    with pytest.raises(FileNotFoundError):
        read_file_text(str(root / "nope.txt"), start_line=1, line_count=10, allowed_roots=[root])


def test_directory_path_is_rejected(tmp_path):
    root = tmp_path / "root"
    sub = root / "sub"
    sub.mkdir(parents=True)

    with pytest.raises(ValueError, match="目录"):
        read_file_text(str(sub), start_line=1, line_count=10, allowed_roots=[root])


def test_empty_file_does_not_crash(tmp_path):
    f, root = _make_file(tmp_path, "empty.txt", "")

    text = read_file_text(str(f), start_line=1, line_count=10, allowed_roots=[root])

    assert "0 行" in text
    assert "1\t" not in text


# --- tool contract ----------------------------------------------------------


async def test_handler_returns_string(tmp_path):
    f, root = _make_file(tmp_path, "a.txt", "alpha\n")
    tool = build_read_file_tool([root])

    result = await tool.handler(_ctx(), path=str(f))

    assert isinstance(result, str)
    assert "1\talpha" in result


async def test_executor_normalizes_string_result(tmp_path):
    f, root = _make_file(tmp_path, "a.txt", "alpha\n")
    tool = build_read_file_tool([root])
    executor = FunctionToolExecutor(provider=None)  # type: ignore[arg-type]

    outputs = [item async for item in executor.execute(tool, _ctx(), path=str(f))]

    assert len(outputs) == 1
    assert "1\talpha" in outputs[0].content[0].text


def test_read_file_params_and_surface():
    tool = build_read_file_tool([Path("/tmp/never")])

    props = tool.parameters["properties"]
    assert set(props) == {"path", "start_line", "line_count"}
    assert tool.parameters["required"] == ["path"]
    assert tool.name == "read_file"

    blob = (tool.name + " " + tool.description + " " + repr(tool.parameters)).lower()
    for banned in ("sandbox", "neo"):
        assert banned not in blob
