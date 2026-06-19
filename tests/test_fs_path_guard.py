"""Path-fence primitives: containment via realpath normalization + strict-prefix match."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from agent_runtime.extensions.fs.path_guard import resolve_within_roots


def test_accepts_path_inside_allowed_root(tmp_path):
    root = tmp_path / "root"
    (root / "sub").mkdir(parents=True)
    f = root / "sub" / "f.txt"
    f.write_text("x", encoding="utf-8")

    resolved = resolve_within_roots(str(f), [root])

    assert resolved == Path(os.path.realpath(str(f)))


def test_accepts_root_itself(tmp_path):
    root = tmp_path / "root"
    root.mkdir()

    assert resolve_within_roots(str(root), [root]) == Path(os.path.realpath(str(root)))


def test_rejects_path_outside_roots(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "secret.txt"
    outside.write_text("x", encoding="utf-8")

    with pytest.raises(PermissionError):
        resolve_within_roots(str(outside), [root])


def test_strict_prefix_does_not_accept_sibling(tmp_path):
    """`/root` must not match `/root-secret` (the bare-prefix trap)."""
    root = tmp_path / "root"
    secret = tmp_path / "root-secret"
    root.mkdir()
    secret.mkdir()
    (secret / "f").write_text("x", encoding="utf-8")

    with pytest.raises(PermissionError):
        resolve_within_roots(str(secret / "f"), [root])


def test_symlink_escape_is_rejected(tmp_path):
    """A symlink inside the root whose target is outside is a static escape -> rejected."""
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("x", encoding="utf-8")
    link = root / "link.txt"
    os.symlink(outside, link)

    with pytest.raises(PermissionError):
        resolve_within_roots(str(link), [root])


def test_symlink_into_root_is_accepted(tmp_path):
    """A symlink whose resolved target is inside the root is allowed."""
    root = tmp_path / "root"
    (root / "real").mkdir(parents=True)
    (root / "real" / "t.txt").write_text("x", encoding="utf-8")
    link = root / "link.txt"
    os.symlink(root / "real" / "t.txt", link)

    assert resolve_within_roots(str(link), [root]) == Path(os.path.realpath(str(link)))


def test_default_roots_fall_back_to_skills_dir(tmp_path, monkeypatch):
    """No/empty allowed_roots -> the skills directory is the default fence."""
    monkeypatch.setenv("AGENT_RUNTIME_DATA_DIR", str(tmp_path))
    skills_dir = Path(os.path.realpath(str(tmp_path / "skills")))
    f = skills_dir / "greet" / "SKILL.md"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text("x", encoding="utf-8")

    assert resolve_within_roots(str(f), None) == Path(os.path.realpath(str(f)))
    assert resolve_within_roots(str(f), []) == Path(os.path.realpath(str(f)))


def test_accepts_path_inside_one_of_several_roots(tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    fb = b / "in_b.txt"
    fb.write_text("x", encoding="utf-8")

    assert resolve_within_roots(str(fb), [a, b]) == Path(os.path.realpath(str(fb)))
