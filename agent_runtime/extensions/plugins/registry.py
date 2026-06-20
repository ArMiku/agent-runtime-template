"""Process-level plugin registry.

Plugin subclasses auto-register their *class* here on definition (via
``Plugin.__init_subclass__``), keyed by ``__module__``. ``PluginManager`` later records
the live instances. The default process-level singleton is shared across runtime
instances (accepted for this change, see design §9); a fresh instance can be constructed
for isolation when needed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import Plugin

__all__ = ["PluginRegistry"]


class PluginRegistry:
    """Registry of plugin classes (by module) and their live instances (by name)."""

    _default: "PluginRegistry | None" = None

    def __init__(self) -> None:
        self._classes: dict[str, type[Plugin]] = {}
        self._instances: dict[str, Plugin] = {}

    @classmethod
    def default(cls) -> "PluginRegistry":
        """Return the shared process-level registry, creating it on first use."""
        if cls._default is None:
            cls._default = cls()
        return cls._default

    # —— class registration (auto, via __init_subclass__) ——
    def register_class(self, plugin_cls: type[Plugin]) -> None:
        """Register a plugin *class*, keyed by its defining module."""
        self._classes[plugin_cls.__module__] = plugin_cls

    def classes(self) -> list[type[Plugin]]:
        return list(self._classes.values())

    def get_class(self, module_path: str) -> "type[Plugin] | None":
        return self._classes.get(module_path)

    # —— instance registration (by PluginManager) ——
    def register_instance(self, plugin: Plugin) -> None:
        """Record a live plugin instance, keyed by its metadata name."""
        self._instances[plugin.metadata.name] = plugin

    def unregister_instance(self, name: str) -> None:
        self._instances.pop(name, None)

    def get_instance(self, name: str) -> "Plugin | None":
        """Look up another plugin instance by name (cross-plugin reference)."""
        return self._instances.get(name)

    def instances(self) -> list[Plugin]:
        return list(self._instances.values())
