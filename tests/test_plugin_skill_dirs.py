"""Phase 4: plugins declare bundled skill directories; the host wires them into the
skills layer as read-only scan roots. The two layers never import each other."""

from __future__ import annotations

from agent_runtime.extensions.plugins.base import Plugin
from agent_runtime.extensions.plugins.context import PluginContext
from agent_runtime.extensions.plugins.contributions import collect_contribution
from agent_runtime.extensions.plugins.metadata import PluginMetadata
from agent_runtime.extensions.plugins.store import InMemoryPluginStore
from agent_runtime.extensions.skills.skill_manager import SkillManager


def _ctx(name: str = "p") -> PluginContext:
    return PluginContext(
        metadata=PluginMetadata(name=name, author="a", desc="d", version="1"),
        plugin_store=InMemoryPluginStore(),
    )


def _write_skill(root, name: str, desc: str) -> None:
    skill_dir = root / name
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {desc}\n---\n# {name}\n", encoding="utf-8"
    )


def test_plugin_declared_skill_dirs_collected(tmp_path):
    skills_dir = tmp_path / "plugin_skills"
    _write_skill(skills_dir, "foo", "a plugin skill.")

    class _SkillPlugin(Plugin):
        name, author, desc, version = "skillplugin", "a", "d", "1"
        skills_dirs = [skills_dir]

    contribution = collect_contribution(_SkillPlugin(_ctx()))

    assert contribution.skill_dirs == [skills_dir]


def test_plugin_without_skills_dirs_contributes_empty():
    class _Plain(Plugin):
        name, author, desc, version = "plain", "a", "d", "1"

    contribution = collect_contribution(_Plain(_ctx()))

    assert contribution.skill_dirs == []


def test_injected_plugin_skills_discovered_readonly_and_local_priority(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setenv("AGENT_RUNTIME_DATA_DIR", str(data_dir))

    local_root = tmp_path / "skills"
    plugin_root = tmp_path / "plugin_skills"
    _write_skill(local_root, "shared", "LOCAL")  # local overrides plugin on clash
    _write_skill(plugin_root, "shared", "PLUGIN")
    _write_skill(plugin_root, "onlyplugin", "plugin only")

    class _SkillPlugin(Plugin):
        name, author, desc, version = "skillplugin", "a", "d", "1"
        skills_dirs = [plugin_root]

    contribution = collect_contribution(_SkillPlugin(_ctx()))

    # Host-side wiring: the skills layer only sees a path list, never the plugin type.
    manager = SkillManager(skills_root=str(local_root), extra_skill_dirs=contribution.skill_dirs)
    by_name = {s.name: s for s in manager.list_skills()}

    assert by_name["shared"].source_type == "local"
    assert by_name["shared"].description == "LOCAL"
    assert by_name["onlyplugin"].source_type == "plugin"
    assert by_name["onlyplugin"].readonly is True
    assert manager.is_plugin_skill("onlyplugin") is True
