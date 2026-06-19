"""Framework-agnostic plugin mechanism for the agent runtime (design.md).

A unified, event-free way to extend the agent loop: subclass :class:`Plugin`, contribute
tools with ``@tool`` and loop hooks with ``@on_llm_request`` / ``@on_agent_begin`` /
``@on_agent_done`` / ``@on_tool_start`` / ``@on_tool_end`` (every hook targets
``run_context``, never an ``event``). :class:`PluginManager` registers plugins by code
injection, drives their lifecycle, and collects each plugin's
:class:`PluginContribution` (tools + hooks). :class:`CompositeAgentRunHooks` aggregates
many plugins' hooks into the single ``BaseAgentRunHooks`` the runner expects.

Plugins reach conversation context and persistence through :class:`PluginContext`: the
current run's live messages (via the hook's ``run_context``), cross-run history (via the
existing ``ContextStore`` seam), and private KV storage (via the injectable
:class:`PluginStore`, defaulting to the DB-free :class:`InMemoryPluginStore`).

Nothing in ``core`` / ``provider`` / ``message`` / ``media`` imports this package; the
dependency only points inward (the four-seam architecture is preserved).
"""

from __future__ import annotations

from .base import Plugin
from .context import PluginContext
from .contributions import PluginContribution, collect_contribution
from .decorators import (
    on_agent_begin,
    on_agent_done,
    on_llm_request,
    on_tool_end,
    on_tool_start,
    tool,
)
from .hooks import CompositeAgentRunHooks
from .manager import MetadataValidationError, PluginManager
from .metadata import PluginMetadata
from .registry import PluginRegistry
from .session import is_plugin_enabled_for_session
from .store import InMemoryPluginStore, PluginStore

__all__ = [
    # core types
    "Plugin",
    "PluginContext",
    "PluginMetadata",
    "PluginManager",
    "PluginRegistry",
    "PluginContribution",
    "collect_contribution",
    "MetadataValidationError",
    # persistence seam
    "PluginStore",
    "InMemoryPluginStore",
    # hooks aggregation
    "CompositeAgentRunHooks",
    # decorators
    "tool",
    "on_llm_request",
    "on_agent_begin",
    "on_agent_done",
    "on_tool_start",
    "on_tool_end",
    # session enable/disable
    "is_plugin_enabled_for_session",
]
