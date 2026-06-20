"""End-to-end example: fs tools closing the skills loop.

Proves the path-addressed file access makes the skills subsystem self-consistent for
local execution:

* **instruction pull** — ``Skill(name)`` loads a skill's ``SKILL.md``;
* **discovery** — ``list_dir`` reveals the skill directory, including its ``scripts/``
  ancillary folder;
* **read** — ``read_file`` reads the referenced ancillary file through the real
  tool-execution path, with line numbers and metadata.

The fs tool set is built with the default fence (the skills directory), so the skill's
own ancillary files are readable out of the box — no extra wiring.

Run it::

    python -m examples.fs_demo
"""

from __future__ import annotations

import asyncio
import os
import tempfile

from agent_runtime.core import FunctionToolExecutor, SessionContext
from agent_runtime.core.run_context import ContextWrapper
from agent_runtime.core.runners.tool_loop_agent_runner import ToolLoopAgentRunner
from agent_runtime.core.tool import ToolSet
from agent_runtime.extensions.fs import build_fs_tools
from agent_runtime.extensions.skills import (
    SkillManager,
    SkillsPromptHook,
    build_skill_tool,
)
from agent_runtime.foundation.paths import get_skills_dir
from agent_runtime.provider.entities import LLMResponse, ProviderRequest
from agent_runtime.provider.provider import Provider

GREET_SKILL_MD = (
    "---\n"
    "name: greet\n"
    "description: Greet the user by running the bundled script.\n"
    "---\n"
    "# Greet\n\n"
    "Run `scripts/run.sh` to produce the greeting.\n"
)

GREET_SCRIPT = "#!/usr/bin/env bash\necho hello from the greet skill\n"


class _RecordingProvider(Provider):
    """Records the contexts it chats over, then returns scripted replies."""

    def __init__(self, script: list[LLMResponse]) -> None:
        super().__init__(
            {"id": "demo", "type": "demo", "max_context_tokens": 0, "modalities": []}, {}
        )
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


def _seed_greet_skill(data_dir: str) -> str:
    """Drop greet/SKILL.md + greet/scripts/run.sh under the skills root; return greet dir."""
    greet_dir = os.path.join(get_skills_dir(), "greet")
    scripts_dir = os.path.join(greet_dir, "scripts")
    os.makedirs(scripts_dir, exist_ok=True)
    with open(os.path.join(greet_dir, "SKILL.md"), "w", encoding="utf-8") as f:
        f.write(GREET_SKILL_MD)
    with open(os.path.join(scripts_dir, "run.sh"), "w", encoding="utf-8") as f:
        f.write(GREET_SCRIPT)
    return greet_dir


async def main() -> dict:
    # Isolate runtime data so the example is self-contained.
    data_dir = tempfile.mkdtemp(prefix="fs_demo_")
    os.environ["AGENT_RUNTIME_DATA_DIR"] = data_dir
    greet_dir = _seed_greet_skill(data_dir)
    script_path = os.path.join(greet_dir, "scripts", "run.sh")

    skill_manager = SkillManager()
    skill_tool = build_skill_tool(skill_manager)
    # Default fence includes the skills directory, so the ancillary file is readable.
    fs_tools = build_fs_tools()
    tools = ToolSet([skill_tool, *fs_tools])

    # Script: load instructions -> list the skill dir -> read the ancillary script -> done.
    provider = _RecordingProvider(
        [
            LLMResponse(
                role="assistant",
                tools_call_name=["Skill"],
                tools_call_args=[{"name": "greet"}],
                tools_call_ids=["call_1"],
            ),
            LLMResponse(
                role="assistant",
                tools_call_name=["list_dir"],
                tools_call_args=[{"path": greet_dir}],
                tools_call_ids=["call_2"],
            ),
            LLMResponse(
                role="assistant",
                tools_call_name=["read_file"],
                tools_call_args=[{"path": script_path}],
                tools_call_ids=["call_3"],
            ),
            LLMResponse(role="assistant", completion_text="Done."),
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
    async for _ in runner.step_until_done(max_step=8):
        pass

    tool_results = [
        str(getattr(m, "content", ""))
        for m in run_context.messages
        if getattr(m, "role", None) == "tool"
    ]

    result = {
        "discovered_skills": [s.name for s in skill_manager.list_skills()],
        # 1. Instruction pull: SKILL.md body loaded by name.
        "skill_instructions_loaded": any("Run `scripts/run.sh`" in t for t in tool_results),
        # 2. Discovery: list_dir exposes the scripts/ folder.
        "list_dir_shows_scripts_dir": any("scripts\t[dir]" in t for t in tool_results),
        # 3. Read: the ancillary script is read with line numbers + metadata.
        "read_file_loaded_script": any(
            "1\t#!/usr/bin/env bash" in t and "encoding: utf-8" in t for t in tool_results
        ),
    }
    print(result)
    return result


if __name__ == "__main__":
    asyncio.run(main())
