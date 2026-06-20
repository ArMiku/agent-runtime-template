"""Neutral plugin metadata.

Framework-agnostic identity for one plugin: required identity fields plus optional
descriptive ones, with no platform / dashboard / host-specific fields (no
``support_platforms`` / ``pages`` / ``i18n`` / ``logo_path``). Runtime-only bookkeeping
(the loaded class, instance, activation state, config) is carried here too but is never
sourced from a yaml file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = ["PluginMetadata", "REQUIRED_METADATA_FIELDS"]

REQUIRED_METADATA_FIELDS = ("name", "author", "desc", "version")


@dataclass
class PluginMetadata:
    """Identity + descriptive metadata for one plugin."""

    # —— Required identity ——
    name: str
    author: str
    desc: str
    version: str

    # —— Optional descriptive ——
    repo: str | None = None
    display_name: str | None = None
    short_desc: str | None = None
    runtime_version: str | None = None
    """Optional runtime-version hint."""

    # —— Runtime-only bookkeeping (never from yaml) ——
    module_path: str | None = None
    plugin_cls: type | None = None
    instance: Any | None = None
    activated: bool = False
    config: dict | None = field(default=None)

    def missing_required_fields(self) -> list[str]:
        """Return the names of required fields that are empty/missing."""
        missing = []
        for name in REQUIRED_METADATA_FIELDS:
            value = getattr(self, name, None)
            if value is None or (isinstance(value, str) and not value.strip()):
                missing.append(name)
        return missing
