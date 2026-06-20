"""End-to-end example: the read-only skills subsystem driving the runtime.

Proves the skills closed loop is self-consistent for local execution:

* **discovery** — drop ``greet/SKILL.md`` into the skills root and ``SkillManager`` finds it;
* **inventory push** — ``SkillsPromptHook`` injects the active-skill inventory into the
  leading system message (sentinel-delimited, exactly one segment) before each LLM step;
* **instruction pull** — the ``Skill(name)`` tool loads a named skill's ``SKILL.md``
  instructions on demand, through the real tool-execution path.

Ancillary files (``scripts/`` / ``references/`` / ``assets/`` referenced from a SKILL.md)
are read by a separate general-purpose READ capability, delivered by an independent
change; this example only exercises instruction loading.

Run it::

    python -m examples.skills_demo
"""

from __future__ import annotations

import asyncio
import os
import tempfile

from agent_runtime.core import FunctionToolExecutor, SessionContext
from agent_runtime.core.run_context import ContextWrapper
from agent_runtime.core.runners.tool_loop_agent_runner import ToolLoopAgentRunner
from agent_runtime.core.tool import ToolSet
from agent_runtime.extensions.skills import (
    SkillManager,
    SkillsPromptHook,
    build_skill_tool,
)
from agent_runtime.provider.entities import LLMResponse, ProviderRequest
from agent_runtime.provider.provider import Provider

# Sentinel the hook wraps the inventory in (one segment per system message).
INVENTORY_SENTINEL = "<!-- skills-inventory -->"

GREET_SKILL_MD = (
    "---\n"
    "name: greet\n"
    "description: Greet the user warmly and concisely.\n"
    "---\n"
    "# Greet\n\n"
    "Wave and say hello in one short line.\n"
)


class _RecordingProvider(Provider):
    """Records the contexts it chats over, then returns scripted replies."""

    def __init__(self, script: list[LLMResponse]) -> None:
        super().__init__({"id": "demo", "type": "demo", "max_context_tokens": 0, "modalities": []}, {})
        self._script = list(script)
        self._idx = 0
        self.seen_contexts: list = []

    def get_current_key(self) -> str:
        return "demo-key"

    def set_key(self, key: str) -> None: ...

    async def get_models(self) -> list[str]:
        return ["demo-model"]

    async def text_chat(self, *args, **kwargs) -> LLMResponse:
        self.seen_contexts = kwargs.get("contexts") or []
        resp = self._script[min(self._idx, len(self._script) - 1)]
        self._idx += 1
        return resp

    async def text_chat_stream(self, *args, **kwargs):
        yield await self.text_chat(*args, **kwargs)


def _seed_greet_skill(data_dir: str) -> None:
    skill_dir = os.path.join(data_dir, "skills", "greet")
    os.makedirs(skill_dir, exist_ok=True)
    with open(os.path.join(skill_dir, "SKILL.md"), "w", encoding="utf-8") as f:
        f.write(GREET_SKILL_MD)


async def main() -> dict:
    # Isolate runtime data so the example is self-contained.
    data_dir = tempfile.mkdtemp(prefix="skills_demo_")
    os.environ["AGENT_RUNTIME_DATA_DIR"] = data_dir
    _seed_greet_skill(data_dir)

    skill_manager = SkillManager()
    skill_tool = build_skill_tool(skill_manager)

    tools = ToolSet([skill_tool])
    # Script: first the model calls Skill(name="greet"), then it finishes.
    provider = _RecordingProvider(
        [
            LLMResponse(
                role="assistant",
                tools_call_name=["Skill"],
                tools_call_args=[{"name": "greet"}],
                tools_call_ids=["call_1"],
            ),
            LLMResponse(role="assistant", completion_text="Hello! 👋"),
        ]
    )
    request = ProviderRequest(prompt="greet me", system_prompt="", func_tool=tools)
    run_context = ContextWrapper(context=SessionContext(session_id="demo"), messages=[])

    runner = ToolLoopAgentRunner()
    await runner.reset(
        provider=provider,
        request=request,
        run_context=run_context,
        tool_executor=FunctionToolExecutor(provider),
        agent_hooks=SkillsPromptHook(skill_manager),
    )
    async for _ in runner.step_until_done(max_step=5):
        pass

    # 1. Inventory push: the provider saw the inventory in the leading system message,
    #    with exactly one sentinel-delimited segment.
    system_text = "\n".join(
        str(getattr(m, "content", "")) for m in provider.seen_contexts if getattr(m, "role", None) == "system"
    )
    inventory_in_message = "greet" in system_text and "## Skills" in system_text
    segment_count = system_text.count(INVENTORY_SENTINEL)

    # 2. Instruction pull: the Skill tool's result (a tool-role message) carries the
    #    SKILL.md instructions — loaded by name through the real execution path.
    tool_results = [str(getattr(m, "content", "")) for m in run_context.messages if getattr(m, "role", None) == "tool"]
    instructions_loaded = any("Greet" in text for text in tool_results)

    result = {
        "discovered_skills": [s.name for s in skill_manager.list_skills()],
        "inventory_in_system_message": inventory_in_message,
        "inventory_segment_count": segment_count,
        "skill_tool_loaded_instructions": instructions_loaded,
    }
    print(result)
    return result


if __name__ == "__main__":
    asyncio.run(main())
