"""Read-only skill discovery + on-demand instruction loading.

This module ports the *read path* of a mature skills subsystem into the runtime,
stripped of every concern that does not belong to local agent execution:

* no remote-container execution coupling,
* no remote skill-marketplace sync,
* no install / delete / toggle / persist write paths — those are platform
  responsibilities. The runtime treats ``skills_root`` and ``skills.json`` as
  externally-written inputs it only reads.

What survives is the faithful read path: skill-name normalization, ``skill.md`` →
``SKILL.md`` probing (read-only, never renamed), YAML frontmatter description
parsing, prompt/display sanitization, and the skill-inventory prompt builder. A new
``load_skill(name)`` powers the ``Skill(name)`` tool — load instructions by name,
the symmetric counterpart of name-based discovery.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from agent_runtime.foundation.paths import get_data_dir, get_skills_dir

__all__ = [
    "SkillInfo",
    "SkillManager",
    "build_skills_prompt",
]

SKILLS_CONFIG_FILENAME = "skills.json"
DEFAULT_SKILLS_CONFIG: dict[str, dict] = {"skills": {}}

_SKILL_NAME_RE = re.compile(r"^[\w.-]+$")


def _normalize_skill_name(name: str | None) -> str:
    raw = str(name or "")
    return re.sub(r"\s+", "_", raw.strip())


def _normalize_skill_markdown_path(skill_dir: Path) -> Path | None:
    """Return the instruction-file path for a skill directory, probing only.

    Canonical ``SKILL.md`` wins; a legacy ``skill.md`` is recognized so its
    description stays readable, but it is **never renamed** — renaming is a write
    and therefore a platform job. Returns ``None`` when neither file exists.
    """
    canonical = skill_dir / "SKILL.md"
    if canonical.is_file():
        return canonical
    legacy = skill_dir / "skill.md"
    if legacy.is_file():
        return legacy
    return None


@dataclass
class SkillInfo:
    """A discovered skill's metadata (no execution-environment fields)."""

    name: str
    description: str
    path: str
    """Skill directory (forward slashes); shown in the inventory and used to locate
    ancillary files (``scripts/`` / ``references/`` / ``assets/``)."""

    active: bool
    source_type: str = "local"
    """``local`` (under ``skills_root``) or ``plugin`` (under an injected extra dir)."""

    source_label: str = "local"
    local_exists: bool = True
    plugin_name: str = ""
    readonly: bool = False


def _parse_frontmatter_description(text: str) -> str:
    """Extract the ``description`` value from YAML frontmatter.

    Expects the standard SKILL.md format used by Anthropic Claude Skills and the
    OpenAI Codex CLI::

        ---
        name: my-skill
        description: What this skill does and when to use it.
        ---
    """
    if not text.startswith("---"):
        return ""
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return ""
    end_idx = None
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end_idx = i
            break
    if end_idx is None:
        return ""

    frontmatter = "\n".join(lines[1:end_idx])
    try:
        payload = yaml.safe_load(frontmatter) or {}
    except yaml.YAMLError:
        return ""
    if not isinstance(payload, dict):
        return ""

    description = payload.get("description", "")
    if not isinstance(description, str):
        return ""
    return description.strip()


# Regex for sanitizing values rendered into the prompt — only allow safe characters
# to prevent prompt injection via crafted skill paths or descriptions.
_SAFE_PATH_RE = re.compile(r"[^\w./ ,()'\-]", re.UNICODE)
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x1F\x7F]")


def _sanitize_prompt_path_for_prompt(path: str) -> str:
    if not path:
        return ""
    path = path.replace("\\", "/")
    path = path.replace("`", "")
    path = _CONTROL_CHARS_RE.sub("", path)
    return _SAFE_PATH_RE.sub("", path)


def _sanitize_prompt_description(description: str) -> str:
    description = description.replace("`", "")
    description = _CONTROL_CHARS_RE.sub(" ", description)
    return " ".join(description.split())


def _sanitize_skill_display_name(name: str) -> str:
    if _SKILL_NAME_RE.fullmatch(name):
        return name
    return "<invalid_skill_name>"


def build_skills_prompt(skills: list[SkillInfo]) -> str:
    """Build the skills section of the system prompt.

    Generates a markdown-formatted skill inventory for the LLM. Only ``name`` and
    ``description`` are shown upfront; the LLM loads the full ``SKILL.md``
    instructions on demand via the ``Skill(name)`` tool (progressive disclosure).
    """
    skills_lines: list[str] = []
    for skill in skills:
        display_name = _sanitize_skill_display_name(skill.name)

        description = skill.description or "No description"
        description = _sanitize_prompt_description(description)
        if not description:
            description = "No description"

        rendered_path = _sanitize_prompt_path_for_prompt(skill.path)
        if not rendered_path:
            rendered_path = "<skills_root>/<skill_name>"

        skills_lines.append(f"- **{display_name}**: {description}\n  Directory: `{rendered_path}`")
    skills_block = "\n".join(skills_lines)

    return (
        "## Skills\n\n"
        "You have specialized skills — reusable instruction bundles stored "
        "in `SKILL.md` files. Each skill has a **name** and a **description** "
        "that tells you what it does and when to use it.\n\n"
        "### Available skills\n\n"
        f"{skills_block}\n\n"
        "### Skill rules\n\n"
        "1. **Discovery** — The list above is the complete skill inventory "
        "for this session. Full instructions live in each skill's "
        "`SKILL.md` file.\n"
        "2. **When to trigger** — Use a skill if the user names it "
        "explicitly, or if the task clearly matches the skill's description. "
        "*Never silently skip a matching skill* — either use it or briefly "
        "explain why you chose not to.\n"
        "3. **Mandatory grounding** — Before executing any skill you MUST "
        "first load its `SKILL.md` instructions by calling the `Skill` tool "
        'with the skill\'s **name** (e.g. `Skill(name="<skill_name>")`). '
        "Never rely on memory or assumptions about a skill's content.\n"
        "4. **Progressive disclosure** — Load only what is directly "
        "referenced from `SKILL.md`:\n"
        "   - If `scripts/` exist, prefer running or patching them over "
        "rewriting code from scratch.\n"
        "   - If `references/`, `assets/`, or templates exist, reuse them.\n"
        "   - Do NOT bulk-load every file in the skill directory.\n"
        "5. **Coordination** — When multiple skills apply, pick the minimal "
        "set needed. Announce which skill(s) you are using and why "
        "(one short line).\n"
        "6. **Context hygiene** — Avoid deep reference chasing; open only "
        "files that are directly linked from `SKILL.md`. The directory path "
        "shown for each skill is for locating its ancillary files.\n"
        "7. **Failure handling** — If a skill cannot be applied, state the "
        "issue clearly and continue with the best alternative.\n"
    )


class SkillManager:
    """Read-only skill discovery + instruction loading.

    ``skills_root`` and ``skills.json`` are externally-written inputs the manager
    only reads — it never installs, deletes, toggles, or persists. Drop
    ``<name>/SKILL.md`` into ``skills_root`` (default ``get_skills_dir()``) to make a
    skill discoverable; it is active by default unless ``skills.json`` opts it out.
    """

    def __init__(
        self,
        skills_root: str | None = None,
        extra_skill_dirs: list[Path] | None = None,
        config_path: str | None = None,
    ) -> None:
        self.skills_root = skills_root or get_skills_dir()
        self.extra_skill_dirs: list[Path] = [Path(d) for d in (extra_skill_dirs or [])]
        self.config_path = config_path or os.path.join(get_data_dir(), SKILLS_CONFIG_FILENAME)
        os.makedirs(self.skills_root, exist_ok=True)
        # mtime cache: list_skills runs every LLM step, so the parsed inventory is
        # cached keyed on the mtimes of skills_root + skills.json (+ extra dirs).
        self._cache_key_state: tuple | None = None
        self._cache_value: list[SkillInfo] | None = None

    # —— discovery ——

    def list_skills(self, *, active_only: bool = False) -> list[SkillInfo]:
        """List discovered skills, local taking priority over plugin on name clash.

        Reads ``skills.json`` for the active opt-out flag (missing ⇒ active). Pure
        read: writes nothing. Results are mtime-cached across the LLM-step loop.
        """
        key = self._cache_key()
        if self._cache_value is not None and self._cache_key_state == key:
            all_skills = self._cache_value
        else:
            all_skills = self._scan()
            self._cache_key_state = key
            self._cache_value = all_skills
        if active_only:
            return [skill for skill in all_skills if skill.active]
        return list(all_skills)

    def is_plugin_skill(self, name: str) -> bool:
        """True if ``name`` is discovered only under an injected extra dir."""
        local_dir = Path(self.skills_root) / name
        if local_dir.is_dir() and _normalize_skill_markdown_path(local_dir) is not None:
            return False
        for extra_dir in self.extra_skill_dirs:
            candidate = extra_dir / name
            if candidate.is_dir() and _normalize_skill_markdown_path(candidate) is not None:
                return True
        return False

    def load_skill(self, name: str) -> str:
        """Load a named skill's ``SKILL.md`` instruction text (read-only).

        Validates the name, resolves the skill directory (local priority), probes
        for ``SKILL.md`` / legacy ``skill.md``, honors the ``skills.json`` active
        opt-out, and returns the instruction text. Raises on invalid name,
        unknown skill, or a disabled skill.
        """
        if not name or not _SKILL_NAME_RE.fullmatch(name):
            raise ValueError(f"Invalid skill name: {name!r}")
        resolved = self._resolve_skill_dir(name)
        if resolved is None:
            raise FileNotFoundError(f"Skill not found: {name!r}")
        skill_dir, _source = resolved
        skill_md = _normalize_skill_markdown_path(skill_dir)
        if skill_md is None:
            raise FileNotFoundError(f"Skill has no SKILL.md: {name!r}")
        active = self._load_config().get("skills", {}).get(name, {}).get("active", True)
        if not active:
            raise PermissionError(f"Skill is disabled: {name!r}")
        return skill_md.read_text(encoding="utf-8")

    # —— internals ——

    def _scan(self) -> list[SkillInfo]:
        config = self._load_config()
        skill_configs = config.get("skills", {})
        skills_by_name: dict[str, SkillInfo] = {}

        root = Path(self.skills_root)
        if root.is_dir():
            for entry in sorted(root.iterdir(), key=lambda item: item.name):
                if not entry.is_dir():
                    continue
                skill_md = _normalize_skill_markdown_path(entry)
                if skill_md is None:
                    continue
                name = entry.name
                active = bool(skill_configs.get(name, {}).get("active", True))
                skills_by_name[name] = SkillInfo(
                    name=name,
                    description=self._read_description(skill_md),
                    path=str(entry).replace("\\", "/"),
                    active=active,
                    source_type="local",
                    source_label="local",
                    local_exists=True,
                )

        for extra_dir in self.extra_skill_dirs:
            if not extra_dir.is_dir():
                continue
            for entry in sorted(extra_dir.iterdir(), key=lambda item: item.name):
                if not entry.is_dir():
                    continue
                skill_md = _normalize_skill_markdown_path(entry)
                if skill_md is None:
                    continue
                name = entry.name
                if name in skills_by_name:
                    continue  # local wins over plugin on name clash
                active = bool(skill_configs.get(name, {}).get("active", True))
                plugin_name = extra_dir.name
                skills_by_name[name] = SkillInfo(
                    name=name,
                    description=self._read_description(skill_md),
                    path=str(entry).replace("\\", "/"),
                    active=active,
                    source_type="plugin",
                    source_label=plugin_name,
                    local_exists=True,
                    plugin_name=plugin_name,
                    readonly=True,
                )

        return [skills_by_name[name] for name in sorted(skills_by_name)]

    def _resolve_skill_dir(self, name: str) -> tuple[Path, str] | None:
        """Return ``(skill_dir, source_type)`` for a name, local priority."""
        local_dir = Path(self.skills_root) / name
        if local_dir.is_dir() and _normalize_skill_markdown_path(local_dir) is not None:
            return local_dir, "local"
        for extra_dir in self.extra_skill_dirs:
            candidate = extra_dir / name
            if candidate.is_dir() and _normalize_skill_markdown_path(candidate) is not None:
                return candidate, "plugin"
        return None

    def _load_config(self) -> dict:
        """Read ``skills.json`` (never writes; missing/invalid ⇒ empty defaults)."""
        if not os.path.exists(self.config_path):
            return DEFAULT_SKILLS_CONFIG.copy()
        try:
            with open(self.config_path, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            return DEFAULT_SKILLS_CONFIG.copy()
        if not isinstance(data, dict) or "skills" not in data:
            return DEFAULT_SKILLS_CONFIG.copy()
        return data

    @staticmethod
    def _read_description(skill_md: Path) -> str:
        try:
            content = skill_md.read_text(encoding="utf-8")
        except OSError:
            return ""
        return _parse_frontmatter_description(content)

    def _cache_key(self) -> tuple:
        """mtimes of skills_root + skills.json + extra dirs (cache invalidation key)."""
        parts: list[tuple] = []

        def mtime(path: str) -> int | None:
            try:
                return os.stat(path).st_mtime_ns
            except OSError:
                return None

        parts.append(("root", mtime(self.skills_root)))
        parts.append(("cfg", mtime(self.config_path)))
        for extra_dir in self.extra_skill_dirs:
            parts.append(("extra", str(extra_dir), mtime(str(extra_dir))))
        return tuple(parts)
