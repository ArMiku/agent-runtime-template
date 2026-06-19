"""What a plugin contributes (design.md §4).

``PluginManager`` instantiates a plugin, then ``collect_contribution`` scans its
decorated methods (``@tool`` / ``@on_*``) and produces a ``PluginContribution``: the
clean ``FunctionTool`` objects to feed ``func_tool`` / executor, and the hook methods
grouped by event for ``CompositeAgentRunHooks`` to aggregate.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass, field

from agent_runtime.core.tool import FunctionTool

from .base import Plugin
from .decorators import (
    HOOK_MARKER_ATTR,
    TOOL_MARKER_ATTR,
    build_tool_parameters,
)

__all__ = ["PluginContribution", "collect_contribution"]


@dataclass
class PluginContribution:
    """The tools + hooks one plugin contributes."""

    plugin: Plugin
    tools: list[FunctionTool] = field(default_factory=list)
    hook_methods: dict[str, list[Callable]] = field(default_factory=dict)


def collect_contribution(plugin: Plugin) -> PluginContribution:
    """Scan a plugin instance for ``@tool`` / ``@on_*`` methods → ``PluginContribution``.

    Bound methods are used so a tool handler's first positional argument is
    ``run_context`` (``self`` is already bound) — matching the executor's
    ``tool.handler(run_context, **params)`` call, and hooks are awaited directly by the
    aggregator with no ``event``.
    """
    contribution = PluginContribution(plugin=plugin)

    # Iterate the class to find decorated *functions* (markers live on the function),
    # then bind each to the instance.
    for attr_name, member in inspect.getmembers(type(plugin), predicate=inspect.isfunction):
        tool_marker = getattr(member, TOOL_MARKER_ATTR, None)
        hook_event = getattr(member, HOOK_MARKER_ATTR, None)

        if tool_marker is not None:
            bound = getattr(plugin, attr_name)
            tool_name = tool_marker["name"]
            parameters, description = build_tool_parameters(member, tool_name=tool_name)
            contribution.tools.append(
                FunctionTool(
                    name=tool_name,
                    description=description,
                    parameters=parameters,
                    handler=bound,
                )
            )

        if hook_event is not None:
            bound = getattr(plugin, attr_name)
            contribution.hook_methods.setdefault(hook_event, []).append(bound)

    return contribution
