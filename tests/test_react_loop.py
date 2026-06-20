"""ReAct loop scenarios (spec: 支持推理的 ReAct 循环).

Covers 6.1 (streaming passthrough), 6.2 (non-streaming termination on a no-tool
response), 6.3 (tool-call then text round-trip), and the max_step truncation branch.
"""

from __future__ import annotations

import pytest

from agent_runtime.core.tool import FunctionTool, ToolSet

from .fakes import FakeProvider, llm_text, llm_tool_call, run


class EchoTool(FunctionTool):
    """Echoes its single ``text`` argument back — a deterministic call-path tool."""

    def __init__(self) -> None:
        super().__init__(
            name="echo",
            description="Echo the given text back.",
            parameters={
                "type": "object",
                "properties": {"text": {"type": "string", "description": "text to echo"}},
                "required": ["text"],
            },
        )

    async def call(self, context, **kwargs) -> str:  # noqa: ANN001
        return f"echo:{kwargs['text']}"


def _echo_tools() -> ToolSet:
    ts = ToolSet()
    ts.add_tool(EchoTool())
    return ts


@pytest.mark.asyncio
async def test_non_streaming_terminates_on_text():
    """6.2: a no-tool assistant response terminates the loop and is returned."""
    provider = FakeProvider([llm_text("Hello there.")])
    final, _ = await run(provider, None, "hi", streaming=False)
    assert final is not None
    assert final.completion_text == "Hello there."


@pytest.mark.asyncio
async def test_streaming_passes_through_completion_and_reasoning():
    """6.1: streaming path exercises text_chat_stream; completion + reasoning pass through."""
    provider = FakeProvider(
        [llm_text("Streaming reply.", reasoning="<think>reasoning</think>")],
        chunk_text=True,
    )
    final, _ = await run(provider, None, "hi", streaming=True)
    assert final is not None
    assert "Streaming reply." in (final.completion_text or "")
    # reasoning_content must be visible on the final response (non-streaming capture path)
    assert final.reasoning_content


@pytest.mark.asyncio
async def test_tool_call_then_text_round_trip():
    """6.3: model calls a tool first, then returns plain text; loop runs two steps."""
    provider = FakeProvider(
        [
            llm_tool_call("echo", {"text": "world"}, call_id="c1"),
            llm_text("The echo was: echo:world"),
        ]
    )
    final, _ = await run(provider, _echo_tools(), "echo 'world'")
    assert final is not None
    # The provider was called twice (tool-call step + final text step).
    assert len(provider.calls) == 2
    assert "echo:world" in final.completion_text


@pytest.mark.asyncio
async def test_max_step_truncation_injects_summary_prompt():
    """When max_step is hit, the runner drops tools + injects a 'summarise' prompt,
    then runs one more step whose (text) response becomes the final reply."""
    # Two tool-call steps exhaust max_step; the forced summary step yields text.
    provider = FakeProvider(
        [
            llm_tool_call("echo", {"text": "x"}),
            llm_tool_call("echo", {"text": "y"}),
            llm_text("Summary after hitting the step limit."),
        ]
    )
    final, _ = await run(provider, _echo_tools(), "loop", max_step=2)
    assert final is not None
    assert "Summary after hitting the step limit." in final.completion_text
