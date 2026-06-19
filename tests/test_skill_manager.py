"""Phase 1: the read-only ``SkillManager`` discovery + ``load_skill`` path."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from agent_runtime.extensions.skills.skill_manager import (
    SkillInfo,
    SkillManager,
    build_skills_prompt,
)


def _make_skill(root: Path, name: str, *, body: str = "# body\n", legacy: bool = False) -> Path:
    skill_dir = root / name
    skill_dir.mkdir(parents=True)
    (skill_dir / ("skill.md" if legacy else "SKILL.md")).write_text(body, encoding="utf-8")
    return skill_dir


@pytest.fixture
def isolated_runtime(monkeypatch, tmp_path):
    """Point the runtime data dir at a temp root so skills_root/skills.json are isolated."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setenv("AGENT_RUNTIME_DATA_DIR", str(data_dir))
    return data_dir


def test_local_discovery_default_active(isolated_runtime, tmp_path):
    root = tmp_path / "skills"
    _make_skill(
        root,
        "greet",
        body="---\nname: greet\ndescription: Say hi.\n---\n# Greet\n",
    )
    manager = SkillManager(skills_root=str(root))

    skills = manager.list_skills()
    assert [s.name for s in skills] == ["greet"]
    greet = skills[0]
    assert greet.active is True
    assert greet.source_type == "local"
    assert greet.description == "Say hi."
    assert greet.path.endswith("greet")


def test_scan_writes_nothing(isolated_runtime, tmp_path):
    root = tmp_path / "skills"
    _make_skill(root, "greet", body="---\nname: greet\ndescription: x\n---\n")
    manager = SkillManager(skills_root=str(root))

    manager.list_skills()

    # No skills.json is created by the read path; missing ⇒ default active.
    assert not (isolated_runtime / "skills.json").exists()


def test_skills_json_active_false_filters_local_and_plugin(isolated_runtime, tmp_path):
    root = tmp_path / "skills"
    plugin_root = tmp_path / "pluginskills"
    _make_skill(root, "a", body="---\nname: a\ndescription: local a\n---\n")
    _make_skill(plugin_root, "b", body="---\nname: b\ndescription: plugin b\n---\n")
    # opt both out
    (isolated_runtime / "skills.json").write_text(
        json.dumps({"skills": {"a": {"active": False}, "b": {"active": False}}}),
        encoding="utf-8",
    )

    manager = SkillManager(skills_root=str(root), extra_skill_dirs=[plugin_root])
    active = {s.name for s in manager.list_skills(active_only=True)}
    assert active == set()
    # active_only=False still surfaces them, flagged inactive
    all_skills = {s.name: s for s in manager.list_skills()}
    assert all_skills["a"].active is False
    assert all_skills["b"].active is False
    assert all_skills["b"].source_type == "plugin"
    assert all_skills["b"].readonly is True


def test_local_wins_over_plugin_on_name_clash(isolated_runtime, tmp_path):
    root = tmp_path / "skills"
    plugin_root = tmp_path / "pluginskills"
    _make_skill(root, "shared", body="---\nname: shared\ndescription: LOCAL\n---\n")
    _make_skill(plugin_root, "shared", body="---\nname: shared\ndescription: PLUGIN\n---\n")
    _make_skill(plugin_root, "onlyplugin", body="---\nname: onlyplugin\ndescription: P\n---\n")

    manager = SkillManager(skills_root=str(root), extra_skill_dirs=[plugin_root])
    by_name = {s.name: s for s in manager.list_skills()}
    assert by_name["shared"].source_type == "local"
    assert by_name["shared"].description == "LOCAL"
    assert by_name["onlyplugin"].source_type == "plugin"
    assert manager.is_plugin_skill("onlyplugin") is True
    assert manager.is_plugin_skill("shared") is False


def test_legacy_skill_md_read_but_not_renamed(isolated_runtime, tmp_path):
    root = tmp_path / "skills"
    skill_dir = _make_skill(
        root, "legacy", body="---\nname: legacy\ndescription: legacy desc\n---\nbody", legacy=True
    )

    manager = SkillManager(skills_root=str(root))
    skills = manager.list_skills()
    assert [s.name for s in skills] == ["legacy"]
    assert skills[0].description == "legacy desc"
    # The legacy file is untouched: no entry is renamed/created (rename is a write /
    # platform job). Checking on-disk entry names is correct on both case-sensitive
    # and case-insensitive filesystems.
    assert {p.name for p in skill_dir.iterdir()} == {"skill.md"}
    assert (skill_dir / "skill.md").read_text(encoding="utf-8").startswith("---")


def test_build_skills_prompt_sanitizes_description_and_path(isolated_runtime, tmp_path):
    skill = SkillInfo(
        name="weird",
        description="has `backticks` and\x00control",
        path="/tmp/sk`ills/weird",
        active=True,
    )
    prompt = build_skills_prompt([skill])
    # The malicious backticks injected via description/path are stripped (the template's
    # own formatting backticks are unrelated and may remain).
    assert "has backticks and control" in prompt  # backticks removed, control → space
    assert "sk`ills" not in prompt  # path backtick stripped
    assert "skills/weird" in prompt
    assert "\x00" not in prompt


def test_mtime_cache_hits_then_invalidates(isolated_runtime, tmp_path):
    root = tmp_path / "skills"
    _make_skill(root, "greet", body="---\nname: greet\ndescription: v1\n---\n")
    manager = SkillManager(skills_root=str(root))

    scans = {"n": 0}
    original_scan = manager._scan

    def counting_scan():
        scans["n"] += 1
        return original_scan()

    manager._scan = counting_scan  # type: ignore[method-assign]

    manager.list_skills()
    manager.list_skills()
    assert scans["n"] == 1  # second call served from cache

    # Adding a skill dir changes skills_root mtime ⇒ cache invalidates.
    _make_skill(root, "wave", body="---\nname: wave\ndescription: v2\n---\n")
    # bump mtime deterministically in case of coarse fs granularity
    os.utime(root, None)
    skills = manager.list_skills()
    assert scans["n"] == 2
    assert {s.name for s in skills} == {"greet", "wave"}


def test_mtime_cache_invalidates_on_skills_json_change(isolated_runtime, tmp_path):
    root = tmp_path / "skills"
    _make_skill(root, "greet", body="---\nname: greet\ndescription: d\n---\n")
    manager = SkillManager(skills_root=str(root))

    manager.list_skills()
    assert {s.name for s in manager.list_skills(active_only=True)} == {"greet"}

    # Disable via skills.json ⇒ mtime change invalidates cache.
    (isolated_runtime / "skills.json").write_text(
        json.dumps({"skills": {"greet": {"active": False}}}), encoding="utf-8"
    )
    os.utime(isolated_runtime / "skills.json", None)
    assert manager.list_skills(active_only=True) == []


def test_load_skill_returns_instructions(isolated_runtime, tmp_path):
    root = tmp_path / "skills"
    body = "---\nname: greet\ndescription: hi\n---\n# Greet\nWave politely.\n"
    _make_skill(root, "greet", body=body)

    manager = SkillManager(skills_root=str(root))
    assert manager.load_skill("greet") == body


def test_load_skill_rejects_invalid_and_unknown(isolated_runtime, tmp_path):
    root = tmp_path / "skills"
    _make_skill(root, "greet", body="---\nname: greet\ndescription: hi\n---\n")
    manager = SkillManager(skills_root=str(root))

    with pytest.raises(ValueError):
        manager.load_skill("../other")
    with pytest.raises(ValueError):
        manager.load_skill("a!!")
    with pytest.raises(FileNotFoundError):
        manager.load_skill("nope")


def test_load_skill_rejects_inactive(isolated_runtime, tmp_path):
    root = tmp_path / "skills"
    _make_skill(root, "greet", body="---\nname: greet\ndescription: hi\n---\n")
    (isolated_runtime / "skills.json").write_text(
        json.dumps({"skills": {"greet": {"active": False}}}), encoding="utf-8"
    )
    manager = SkillManager(skills_root=str(root))

    with pytest.raises(PermissionError):
        manager.load_skill("greet")
