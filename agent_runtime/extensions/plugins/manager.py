"""``PluginManager`` — plugin discovery, lifecycle, and auto-registration.

A plugin is a **development-time capability-injection unit**: write a ``Plugin`` subclass
and register it (code injection — the main path). Online install / download / update and
``requirements.txt`` auto-install are out of scope (that is "plugin marketplace"
semantics). An optional directory-load path uses ``importlib`` for local modules — no
download.
"""

from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path

from agent_runtime import logger

from .base import Plugin
from .context import PluginContext
from .contributions import PluginContribution, collect_contribution
from .metadata import PluginMetadata
from .registry import PluginRegistry
from .store import InMemoryPluginStore, PluginStore

__all__ = ["PluginManager", "MetadataValidationError"]


class MetadataValidationError(ValueError):
    """Raised when a plugin's metadata is missing required fields."""


class PluginManager:
    """Registers plugins (code injection), drives their lifecycle, collects contributions."""

    def __init__(
        self,
        *,
        registry: PluginRegistry | None = None,
        plugin_store: PluginStore | None = None,
        context_store=None,
    ) -> None:
        self.registry = registry or PluginRegistry.default()
        self.plugin_store = plugin_store or InMemoryPluginStore()
        self.context_store = context_store
        self.contributions: list[PluginContribution] = []
        self._plugins: dict[str, Plugin] = {}

    # —— main path: code injection ——
    async def register(
        self,
        plugin_cls: type[Plugin],
        *,
        metadata: PluginMetadata | None = None,
        config: dict | None = None,
    ) -> PluginContribution:
        """Instantiate ``plugin_cls``, inject context, call ``initialize``, collect contribution."""
        resolved = self._resolve_metadata(plugin_cls, metadata, config)
        context = PluginContext(
            metadata=resolved,
            config=config if config is not None else resolved.config,
            registry=self.registry,
            context_store=self.context_store,
            plugin_store=self.plugin_store,
        )
        plugin = plugin_cls(context)
        return await self._activate(plugin, resolved)

    async def register_instance(
        self,
        plugin: Plugin,
        *,
        metadata: PluginMetadata | None = None,
    ) -> PluginContribution:
        """Register an already-constructed plugin instance."""
        resolved = metadata or getattr(plugin, "metadata", None)
        if resolved is None:
            resolved = self._resolve_metadata(type(plugin), None, None)
        self._reject_if_incomplete(resolved)
        return await self._activate(plugin, resolved)

    async def _activate(self, plugin: Plugin, metadata: PluginMetadata) -> PluginContribution:
        plugin.metadata = metadata
        await plugin.initialize()
        metadata.activated = True
        metadata.instance = plugin
        contribution = collect_contribution(plugin)
        self.contributions.append(contribution)
        self._plugins[metadata.name] = plugin
        self.registry.register_instance(plugin)
        logger.info(f"Registered plugin: {metadata.name} (tools={len(contribution.tools)})")
        return contribution

    async def unload(self, name: str) -> None:
        """Call ``terminate`` and drop the plugin's contribution."""
        plugin = self._plugins.pop(name, None)
        if plugin is None:
            return
        try:
            await plugin.terminate()
        except Exception as e:  # noqa: BLE001
            logger.error(f"Error terminating plugin {name}: {e}", exc_info=True)
        self.contributions = [c for c in self.contributions if c.plugin is not plugin]
        self.registry.unregister_instance(name)
        plugin.metadata.activated = False

    async def reload(
        self,
        name: str,
        *,
        metadata: PluginMetadata | None = None,
        config: dict | None = None,
    ) -> PluginContribution:
        """Terminate → reconstruct from the same class → initialize."""
        plugin = self._plugins.get(name)
        if plugin is None:
            raise KeyError(f"Plugin {name} is not registered.")
        plugin_cls = type(plugin)
        await self.unload(name)
        return await self.register(plugin_cls, metadata=metadata, config=config)

    # —— optional: directory load via importlib (no download) ——
    async def load_from_directory(
        self,
        directory: str | Path,
        *,
        config: dict | None = None,
    ) -> list[PluginContribution]:
        """Import plugin modules from a local directory and register found ``Plugin`` subclasses.

        Imports ``main.py`` / ``<dirname>.py`` from each immediate subdirectory (and any
        top-level ``.py`` files), then registers each ``Plugin`` subclass. No download,
        no requirements auto-install.
        """
        directory = Path(directory)
        contributions: list[PluginContribution] = []
        if not directory.is_dir():
            logger.warning(f"Plugin directory does not exist: {directory}")
            return contributions

        for module_path in self._discover_module_files(directory):
            module = self._import_module_file(module_path)
            if module is None:
                continue
            for plugin_cls in self._plugin_classes_in_module(module):
                try:
                    contributions.append(await self.register(plugin_cls, config=config))
                except MetadataValidationError as e:
                    logger.warning(f"Skipped plugin {plugin_cls.__name__}: {e}")
        return contributions

    @staticmethod
    def _discover_module_files(directory: Path) -> list[Path]:
        files: list[Path] = []
        for sub in sorted(directory.iterdir()):
            if sub.is_dir():
                main_py = sub / "main.py"
                same_named = sub / f"{sub.name}.py"
                if main_py.exists():
                    files.append(main_py)
                elif same_named.exists():
                    files.append(same_named)
            elif sub.suffix == ".py" and sub.name != "__init__.py":
                files.append(sub)
        return files

    @staticmethod
    def _import_module_file(module_path: Path):
        module_name = f"_agent_runtime_plugin_{module_path.stem}_{abs(hash(str(module_path)))}"
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            logger.warning(f"Could not load module spec for {module_path}")
            return None
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Exception as e:  # noqa: BLE001
            logger.error(f"Failed to import plugin module {module_path}: {e}", exc_info=True)
            return None
        return module

    @staticmethod
    def _plugin_classes_in_module(module) -> list[type[Plugin]]:
        """Return ``Plugin`` subclasses *defined in* this module."""
        found: list[type[Plugin]] = []
        for _, member in inspect.getmembers(module, inspect.isclass):
            if issubclass(member, Plugin) and member is not Plugin and member.__module__ == module.__name__:
                found.append(member)
        return found

    # —— metadata resolution / validation ——
    def _resolve_metadata(
        self,
        plugin_cls: type[Plugin],
        metadata: PluginMetadata | None,
        config: dict | None,
    ) -> PluginMetadata:
        if metadata is not None:
            resolved = metadata
        elif isinstance(getattr(plugin_cls, "metadata", None), PluginMetadata):
            resolved = plugin_cls.metadata  # type: ignore[assignment]
        else:
            resolved = PluginMetadata(
                name=getattr(plugin_cls, "name", None) or "",
                author=getattr(plugin_cls, "author", None) or "",
                desc=getattr(plugin_cls, "desc", None) or "",
                version=getattr(plugin_cls, "version", None) or "",
                repo=getattr(plugin_cls, "repo", None),
                display_name=getattr(plugin_cls, "display_name", None),
                short_desc=getattr(plugin_cls, "short_desc", None),
            )
        resolved.module_path = plugin_cls.__module__
        resolved.plugin_cls = plugin_cls
        if config is not None:
            resolved.config = config
        self._reject_if_incomplete(resolved)
        return resolved

    @staticmethod
    def _reject_if_incomplete(metadata: PluginMetadata) -> None:
        missing = metadata.missing_required_fields()
        if missing:
            raise MetadataValidationError(f"Plugin metadata is missing required field(s): {', '.join(missing)}")
