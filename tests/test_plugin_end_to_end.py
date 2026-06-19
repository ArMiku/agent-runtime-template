"""Phase 5 end-to-end test: the example plugin drives the runtime through the mechanism.

Asserts the full seam set works together: register → initialize → tool enters func_tool →
the ``@on_llm_request`` hook's injected system message reaches the provider request body.
"""

from __future__ import annotations

import pytest

from agent_runtime.examples.plugin_demo import INJECTED_SYSTEM_NOTE, main


async def test_example_plugin_end_to_end() -> None:
    result = await main()
    assert "echo" in result["tools"]
    assert result["provider_saw_injected_message"] is True
    assert result["final_text"] == "done"


async def test_injected_note_constant_used() -> None:
    # Guard against the example silently diverging from the asserted note.
    assert INJECTED_SYSTEM_NOTE.startswith("[example]")
