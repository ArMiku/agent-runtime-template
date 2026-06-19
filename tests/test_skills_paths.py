"""Phase 0: ``get_skills_dir()`` resolves under the injectable data directory."""

from __future__ import annotations

import os

from agent_runtime.foundation.paths import get_data_dir, get_skills_dir


def test_get_skills_dir_lands_under_data_dir(monkeypatch, tmp_path):
    data_dir = tmp_path / "runtime_data"
    monkeypatch.setenv("AGENT_RUNTIME_DATA_DIR", str(data_dir))

    skills_dir = get_skills_dir()

    assert skills_dir == os.path.realpath(str(data_dir / "skills"))
    assert os.path.isdir(skills_dir)


def test_get_skills_dir_shares_data_root(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_RUNTIME_DATA_DIR", str(tmp_path))

    assert os.path.dirname(get_skills_dir()) == get_data_dir()
