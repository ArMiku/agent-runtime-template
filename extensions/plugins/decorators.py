"""Plugin decorators (design.md §3.3).

All decorators target ``run_context`` — never an ``event``. ``@tool`` marks a method to
be collected into a clean :class:`~agent_runtime.core.tool.FunctionTool` (its handler's
first positional argument is ``run_context``), parsing parameters from the docstring with
``docstring_parser`` + the ``PY_TO_JSON_TYPE`` map. The ``@on_*`` decorators only tag a
method with the hook event it serves; the actual wiring happens in ``contributions.py`` /
``hooks.py``.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any, TypeVar, overload

import docstring_parser

__all__ = [
    "tool",
    "on_llm_request",
    "on_agent_begin",
    "on_agent_done",
    "on_tool_start",
    "on_tool_end",
    "HOOK_EVENTS",
    "TOOL_MARKER_ATTR",
    "HOOK_MARKER_ATTR",
    "build_tool_parameters",
]

# JSON-schema types accepted in tool parameter annotations.
SUPPORTED_TYPES = ["string", "number", "object", "array", "boolean"]
PY_TO_JSON_TYPE = {
    "int": "number",
    "float": "number",
    "bool": "boolean",
    "str": "string",
    "dict": "object",
    "list": "array",
    "tuple": "array",
    "set": "array",
}

HOOK_EVENTS = (
    "on_llm_request",
    "on_agent_begin",
    "on_agent_done",
    "on_tool_start",
    "on_tool_end",
)

# Attributes stamped onto decorated methods so the contribution scanner can find them.
TOOL_MARKER_ATTR = "__plugin_tool__"
HOOK_MARKER_ATTR = "__plugin_hook_event__"


def build_tool_parameters(func: Callable, *, tool_name: str) -> tuple[dict[str, Any], str]:
    """Parse a function's docstring into (JSON-schema parameters, description).

    Parses Google-style ``Args:`` entries (``name(type): description``) into a JSON-schema
    ``properties`` map. Raises ``ValueError`` on a parameter missing a type or using an
    unsupported type.
    """
    doc = func.__doc__ or ""
    docstring = docstring_parser.parse(doc)
    properties: dict[str, Any] = {}
    for arg in docstring.params:
        type_name = arg.type_name
        if not type_name:
            raise ValueError(
                f"Plugin tool {func.__module__}.{tool_name} parameter '{arg.arg_name}' is missing a type annotation."
            )
        sub_type_name = None
        match = re.match(r"(\w+)\[(\w+)\]", type_name)
        if match:
            type_name = match.group(1)
            sub_type_name = match.group(2)
        type_name = PY_TO_JSON_TYPE.get(type_name, type_name)
        if sub_type_name:
            sub_type_name = PY_TO_JSON_TYPE.get(sub_type_name, sub_type_name)
        if type_name not in SUPPORTED_TYPES or (sub_type_name and sub_type_name not in SUPPORTED_TYPES):
            raise ValueError(
                f"Plugin tool {func.__module__}.{tool_name} has an unsupported parameter type: {arg.type_name}"
            )
        prop: dict[str, Any] = {"type": type_name, "description": arg.description or ""}
        if sub_type_name and type_name == "array":
            prop["items"] = {"type": sub_type_name}
        properties[arg.arg_name] = prop

    parameters = {"type": "object", "properties": properties}
    description = docstring.description.strip() if docstring.description else ""
    return parameters, description


F = TypeVar("F", bound=Callable[..., Any])


@overload
def tool(func: F) -> F: ...
@overload
def tool(*, name: str | None = None) -> Callable[[F], F]: ...
def tool(func: F | None = None, *, name: str | None = None) -> F | Callable[[F], F]:
    """Mark a plugin method as a tool.

    Usage: ``@tool`` or ``@tool(name="...")``. The decorated method's signature is
    ``(self, run_context, **params)``; ``params`` are parsed from the docstring.
    """

    def decorator(target: F) -> F:
        setattr(target, TOOL_MARKER_ATTR, {"name": name or target.__name__})
        return target

    # Bare ``@tool`` (called with the function directly).
    if func is not None:
        return decorator(func)
    return decorator


def _hook_decorator(event: str) -> Callable[[F], F]:
    def decorator(func: F) -> F:
        setattr(func, HOOK_MARKER_ATTR, event)
        return func

    return decorator


on_llm_request = _hook_decorator("on_llm_request")
on_agent_begin = _hook_decorator("on_agent_begin")
on_agent_done = _hook_decorator("on_agent_done")
on_tool_start = _hook_decorator("on_tool_start")
on_tool_end = _hook_decorator("on_tool_end")
