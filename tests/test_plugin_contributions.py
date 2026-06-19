"""Phase 2 tests: decorators (@tool / @on_*) and contribution collection."""

from __future__ import annotations

import mcp
import pytest

from agent_runtime.core import FunctionToolExecutor
from agent_runtime.core.run_context import ContextWrapper
from agent_runtime.core.session_context import SessionContext
from agent_runtime.extensions.plugins.base import Plugin
from agent_runtime.extensions.plugins.context import PluginContext
from agent_runtime.extensions.plugins.contributions import collect_contribution
from agent_runtime.extensions.plugins.decorators import (
    on_agent_begin,
    on_llm_request,
    tool,
)
from agent_runtime.extensions.plugins.metadata import PluginMetadata
from agent_runtime.extensions.plugins.store import InMemoryPluginStore


def _ctx(name: str = "p") -> PluginContext:
    return PluginContext(
        metadata=PluginMetadata(name=name, author="a", desc="d", version="1"),
        plugin_store=InMemoryPluginStore(),
    )


class _DemoPlugin(Plugin):
    @tool
    async def echo(self, run_context, text: str) -> str:
        """Echo the input.

        Args:
            text(string): the text to echo
        """
        return text

    @tool(name="add_numbers")
    async def add(self, run_context, a: int, b: int) -> str:
        """Add two numbers.

        Args:
            a(int): first
            b(int): second
        """
        return str(a + b)

    @on_llm_request
    async def inject(self, run_context) -> None:
        run_context.messages.append("injected")

    @on_agent_begin
    async def began(self, run_context) -> None:
        ...


async def test_tool_produces_executable_function_tool() -> None:
    plugin = _DemoPlugin(_ctx())
    contribution = collect_contribution(plugin)

    tools = {t.name: t for t in contribution.tools}
    assert set(tools) == {"echo", "add_numbers"}

    echo = tools["echo"]
    assert echo.description == "Echo the input."
    assert echo.parameters["properties"]["text"] == {"type": "string", "description": "the text to echo"}

    # Handler is bound: first positional arg is run_context, self already bound.
    # Invoke through the real executor (the production call path).
    run_context = ContextWrapper(context=SessionContext(session_id="t"), messages=[])
    executor = FunctionToolExecutor(provider=None)  # type: ignore[arg-type]  # echo needs no provider
    results = [item async for item in executor.execute(echo, run_context, text="hi")]
    block = results[0].content[0]
    assert isinstance(block, mcp.types.TextContent)
    assert block.text == "hi"


async def test_tool_custom_name_and_type_mapping() -> None:
    plugin = _DemoPlugin(_ctx())
    contribution = collect_contribution(plugin)
    add = {t.name: t for t in contribution.tools}["add_numbers"]
    # int -> number
    assert add.parameters["properties"]["a"]["type"] == "number"
    assert add.parameters["properties"]["b"]["type"] == "number"


async def test_hook_methods_classified_by_event() -> None:
    plugin = _DemoPlugin(_ctx())
    contribution = collect_contribution(plugin)
    assert set(contribution.hook_methods) == {"on_llm_request", "on_agent_begin"}
    assert len(contribution.hook_methods["on_llm_request"]) == 1

    # The bound hook actually mutates run_context.
    run_context = ContextWrapper(context=SessionContext(session_id="t"), messages=[])
    await contribution.hook_methods["on_llm_request"][0](run_context)
    assert run_context.messages == ["injected"]


def test_tool_missing_type_raises() -> None:
    class _Bad(Plugin):
        @tool
        async def broken(self, run_context, text) -> str:
            """No type for the arg.

            Args:
                text: missing type
            """
            return text

    with pytest.raises(ValueError, match="missing a type annotation"):
        collect_contribution(_Bad(_ctx()))
