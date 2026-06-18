"""6.4: AgentResponse.data.chain is a MessageChain consumable without a platform adapter."""

from __future__ import annotations

import pytest

from agent_runtime.message.message_event_result import MessageChain
from agent_runtime.tests.fakes import FakeProvider, llm_text, run


@pytest.mark.asyncio
async def test_final_response_chain_is_message_chain():
    provider = FakeProvider([llm_text("The answer is 42.")])
    _final, responses = await run(provider, None, "what is the answer?")

    chains = [r.data.get("chain") for r in responses if r.data.get("chain") is not None]
    assert chains, "expected at least one AgentResponse carrying a chain"
    last_chain = chains[-1]
    assert isinstance(last_chain, MessageChain)
    # Consumable with no platform adapter: plain text is retrievable directly.
    assert "The answer is 42." in last_chain.get_plain_text()
