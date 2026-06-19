"""Read-only filesystem tools for the agent (``read_file`` + ``list_dir``).

Safe-by-default narrow tools: read-only, JSON-Schema parameters (no shell), and every
``path`` is confined to an allowed-root fence by ``realpath`` normalization (see
``path_guard``). This module depends only inward (``core`` / ``foundation``); it never
imports the skills or plugins extensions — the host wires those in via plain path lists.

Transition contract — these tools are the safety default *for environments that lack an
isolated executor*. When the host provides isolated execution (e.g. a container-isolated
shell), the host SHOULD drop this set and substitute a single Bash tool. Each tool's
docstring restates this.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any

from agent_runtime.core.run_context import ContextWrapper
from agent_runtime.core.tool import FunctionTool

from .path_guard import resolve_within_roots

__all__ = [
    "MAX_BYTES",
    "MAX_ENTRIES",
    "DEFAULT_LINE_COUNT",
    "read_file_text",
    "build_read_file_tool",
    "list_dir_entries",
    "build_list_dir_tool",
    "build_fs_tools",
]

# Per-chunk byte ceiling: the token-safety backstop. Line-count chunking alone cannot
# stop a single huge line (a minified / packed file) from blowing the context, so every
# chunk is also capped at roughly this many bytes, cutting mid-line if it must.
MAX_BYTES = 32 * 1024

# Default (and typical) lines-per-call. The byte cap is the real context guard; this
# just bounds a single logical page.
DEFAULT_LINE_COUNT = 2000

# Per-listdir entry ceiling: stops a huge directory (e.g. ``node_modules``) from
# flooding the context in one call.
MAX_ENTRIES = 1000

_TRANSITION_NOTE = (
    "safe-by-default narrow tool; superseded by an isolated-execution Bash tool when "
    "the host provides one"
)


def read_file_text(
    path: str | os.PathLike[str],
    *,
    start_line: int = 1,
    line_count: int = DEFAULT_LINE_COUNT,
    allowed_roots: list[str | os.PathLike[str]] | None = None,
) -> str:
    """Read a UTF-8 text file as line-numbered, dual-cap chunked text.

    safe-by-default narrow tool; superseded by an isolated-execution Bash tool when the
    host provides one.

    Returns the requested ``[start_line, start_line + line_count)`` slice as
    ``"{n}\\t{line}"`` lines, capped at ~``MAX_BYTES`` bytes per call (mid-line if
    needed), with a footer carrying metadata (total lines, mtime, encoding) and a
    "remaining lines" / "byte-truncated" indicator.

    Raises:
        PermissionError: ``path`` resolves outside every allowed root (incl. a static
            symlink escape).
        FileNotFoundError: the path does not exist.
        ValueError: the file is not UTF-8 text / looks binary / ``start_line`` is out
            of range / the path is a directory.
    """
    resolved = resolve_within_roots(path, allowed_roots)
    try:
        data = resolved.read_bytes()
    except IsADirectoryError as exc:
        raise ValueError(f"path '{resolved}' 是目录，不是文件") from exc

    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(
            f"文件非 UTF-8 文本（在第 {exc.start} 字节处解码失败）；请改用专用解析工具"
        ) from exc
    if "\x00" in text:
        raise ValueError("文件疑似二进制（检测到 NUL 字节）；请改用专用解析工具")

    mtime_iso = datetime.fromtimestamp(resolved.stat().st_mtime).isoformat(timespec="seconds")
    lines = text.splitlines()
    total_lines = len(lines)

    if total_lines == 0:
        return f"--- 共 0 行 | mtime: {mtime_iso} | encoding: utf-8\n(空文件)"

    if start_line < 1:
        raise ValueError(f"start_line 必须 >= 1（当前 {start_line}）")
    if start_line > total_lines:
        raise ValueError(
            f"start_line {start_line} 超出文件长度 {total_lines}（合法范围 1..{total_lines}）"
        )

    end = min(start_line - 1 + max(1, line_count), total_lines)
    out: list[str] = []
    emitted_full = 0
    emitted_bytes = 0
    byte_truncated = False
    truncated_at_line = 0

    i = start_line - 1
    while i < end:
        rendered = f"{i + 1}\t{lines[i]}"
        encoded = rendered.encode("utf-8")
        if emitted_bytes + len(encoded) <= MAX_BYTES:
            out.append(rendered)
            emitted_bytes += len(encoded)
            emitted_full += 1
            i += 1
            continue
        # This line does not fit in the remaining byte budget — cut it mid-line.
        budget = MAX_BYTES - emitted_bytes
        if budget >= 16:
            partial = encoded[:budget].decode("utf-8", errors="ignore")
            out.append(f"{partial} […字节上限截断]")
        byte_truncated = True
        truncated_at_line = i + 1
        break

    consumed_through = start_line - 1 + emitted_full
    reached_end = consumed_through >= total_lines

    body = "\n".join(out)
    footer = (
        f"\n--- 共 {total_lines} 行 | 已读 {emitted_full} 行（{emitted_bytes} 字节）"
        f" | mtime: {mtime_iso} | encoding: utf-8"
    )
    if byte_truncated:
        footer += (
            f"\n已因每块字节上限在约 {MAX_BYTES // 1024}KB 处截断"
            f"（第 {truncated_at_line} 行未完）；剩余内容含本行后续及之后的行。"
        )
    elif reached_end:
        footer += f"\n已读完全部 {total_lines} 行。"
    else:
        remaining = total_lines - consumed_through
        nxt = consumed_through + 1
        footer += f"\n剩余 {remaining} 行未读；续读：read_file(path, start_line={nxt})。"

    return body + footer


def build_read_file_tool(
    allowed_roots: list[str | os.PathLike[str]] | None = None,
) -> FunctionTool:
    """Build the ``read_file`` function tool bound to an allowed-root fence.

    ``allowed_roots`` defaults to the skills directory when omitted/empty (closing the
    skills ancillary-file read dependency).
    """
    roots = list(allowed_roots) if allowed_roots else []

    async def handler(
        run_context: ContextWrapper[Any],  # noqa: ARG001 - required by the tool contract
        *,
        path: str,
        start_line: int = 1,
        line_count: int = DEFAULT_LINE_COUNT,
    ) -> str:
        """safe-by-default narrow tool; superseded by an isolated-execution Bash tool
        when the host provides one.

        Read a UTF-8 text file as line-numbered, chunked text. Use this to read a
        SKILL.md's ancillary files (``scripts/`` / ``references/`` / ``assets/``) or
        any file under an allowed root. Returns one ``{line-number}\\t{line}`` block
        per call with a footer showing total lines, mtime, encoding, and where to
        resume. ``path`` must be absolute and inside an allowed root.
        """
        return read_file_text(
            path,
            start_line=start_line,
            line_count=line_count,
            allowed_roots=roots,
        )

    return FunctionTool(
        name="read_file",
        description=(
            "Read a UTF-8 text file as line-numbered, chunked text (safe-by-default "
            "narrow tool; superseded by an isolated-execution Bash tool when the host "
            "provides one). Pass an absolute path inside an allowed root, an optional "
            "1-based start_line, and an optional line_count. The result lists each "
            "line as '{n}\\t{line}' and ends with a footer (total lines, mtime, "
            "encoding, remaining lines / byte-truncation note)."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute path of the file to read; must be inside an allowed root.",
                },
                "start_line": {
                    "type": "integer",
                    "description": "1-based line to start reading at (default 1).",
                    "default": 1,
                    "minimum": 1,
                },
                "line_count": {
                    "type": "integer",
                    "description": "Maximum lines to return in this call (default 2000).",
                    "default": DEFAULT_LINE_COUNT,
                    "minimum": 1,
                },
            },
            "required": ["path"],
        },
        handler=handler,
    )


def list_dir_entries(
    path: str | os.PathLike[str],
    *,
    allowed_roots: list[str | os.PathLike[str]] | None = None,
) -> str:
    """List a directory's entries (name + type, file size) with an entry-count cap.

    safe-by-default narrow tool; superseded by an isolated-execution Bash tool when the
    host provides one.

    Returns ``"{name}\\t[type]"`` lines (``[dir]`` / ``[file, {n}B]``), sorted by name,
    capped at ``MAX_ENTRIES`` entries with a footer marking truncation and total count.

    Raises:
        PermissionError: ``path`` resolves outside every allowed root.
        FileNotFoundError: the path does not exist.
        ValueError: the path is a file, not a directory.
    """
    resolved = resolve_within_roots(path, allowed_roots)
    try:
        entries = sorted(os.scandir(resolved), key=lambda e: e.name)
    except NotADirectoryError as exc:
        raise ValueError(f"path '{resolved}' 是文件，不是目录") from exc

    total = len(entries)
    shown = entries[:MAX_ENTRIES]
    out: list[str] = []
    for entry in shown:
        try:
            if entry.is_dir():
                out.append(f"{entry.name}\t[dir]")
                continue
            size = entry.stat().st_size
        except OSError:
            size = 0
        out.append(f"{entry.name}\t[file, {size}B]")

    body = "\n".join(out)
    mtime_iso = datetime.fromtimestamp(resolved.stat().st_mtime).isoformat(timespec="seconds")
    meta = f"--- 共 {total} 条 | 已列 {len(shown)} 条 | mtime: {mtime_iso}"
    notice = f"\n已截断，还有 {total - MAX_ENTRIES} 条未列。" if total > MAX_ENTRIES else ""
    sep = "\n" if body else ""
    return f"{body}{sep}{meta}{notice}"


def build_list_dir_tool(
    allowed_roots: list[str | os.PathLike[str]] | None = None,
) -> FunctionTool:
    """Build the ``list_dir`` function tool bound to an allowed-root fence."""
    roots = list(allowed_roots) if allowed_roots else []

    async def handler(
        run_context: ContextWrapper[Any],  # noqa: ARG001 - required by the tool contract
        *,
        path: str,
    ) -> str:
        """safe-by-default narrow tool; superseded by an isolated-execution Bash tool
        when the host provides one.

        List the entries of a directory: each line is "{name}\\t[type]" where type is
        "[dir]" or "[file, {size}B]". Use this to discover what to read before calling
        read_file. ``path`` must be absolute and inside an allowed root. Very large
        directories are truncated with a notice.
        """
        return list_dir_entries(path, allowed_roots=roots)

    return FunctionTool(
        name="list_dir",
        description=(
            "List a directory's entries (safe-by-default narrow tool; superseded by an "
            "isolated-execution Bash tool when the host provides one). Pass an absolute "
            "directory path inside an allowed root. Each line is "
            "'{name}\\t[type]' (type is '[dir]' or '[file, {size}B]'); the result is "
            "sorted by name and truncated with a notice for very large directories."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute path of the directory to list; must be inside an allowed root.",
                },
            },
            "required": ["path"],
        },
        handler=handler,
    )


def build_fs_tools(
    allowed_roots: list[str | os.PathLike[str]] | None = None,
) -> list[FunctionTool]:
    """Build the read-only fs tool set bound to one allowed-root fence.

    Returns ``[list_dir, read_file]`` (discovery before reading). ``allowed_roots``
    defaults to the skills directory when omitted/empty, closing the skills
    ancillary-file read dependency.
    """
    return [build_list_dir_tool(allowed_roots), build_read_file_tool(allowed_roots)]
