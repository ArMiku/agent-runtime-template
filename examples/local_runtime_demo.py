"""End-to-end example: one-shot assembly vs hand-wiring, proven equivalent.

The same scenario — a drop-in ``greet`` skill plus a plugin tool — is run twice:

* **Tier 3 (one-shot)** — ``await build_local_agent(provider, ...)`` then ``agent.run()``;
  the caller writes no ``runner.reset`` and no step loop.
* **Tier 1 (hand-wired)** — import the fine-grained builders, assemble the tools/hooks by
  hand, drive ``runner.step_until_done`` directly.

Both paths are asserted to produce the same observable results (discovered skills, tool
names, inventory injection, plugin tool/hook firing, final response), demonstrating the
two usages are the *same wiring* — the one-shot path just owns the composition root for
you.

Run it::

    python -m examples.local_runtime_demo
"""

from __future__ import annotations

import asyncio
import os
import tempfile

from agent_runtime.core import FunctionToolExecutor, SessionContext
from agent_runtime.core.run_context import ContextWrapper
from agent_runtime.core.runners.tool_loop_agent_runner import ToolLoopAgentRunner
from agent_runtime.core.tool import FunctionTool, ToolSet
from agent_runtime.extensions.fs import build_fs_tools
from agent_runtime.extensions.plugins.base import Plugin
from agent_runtime.extensions.plugins.contributions import PluginContribution
from agent_runtime.extensions.skills import (
    SkillManager,
    SkillsPromptHook,
    build_skill_tool,
)
from agent_runtime.foundation.paths import get_skills_dir
from agent_runtime.local_runtime import build_local_agent
from agent_runtime.provider.entities import LLMResponse, ProviderRequest
from agent_runtime.provider.provider import Provider

GREET_SKILL_MD = "---\nname: greet\ndescription: Greet the user.\n---\n# Greet\n\nSay hello warmly.\n"


class _ScriptedProvider(Provider):
    """Returns scripted replies; a fresh instance per path keeps the scripts independent."""

    def __init__(self, script: list[LLMResponse]) -> None:
        super().__init__({"id": "demo", "type": "demo", "max_context_tokens": 0, "modalities": []}, {})
        self._script = list(script)
        self._idx = 0

    def get_current_key(self) -> str:
        return "demo-key"

    def set_key(self, key: str) -> None: ...

    async def get_models(self) -> list[str]:
        return ["demo-model"]

    async def text_chat(self, *args, **kwargs) -> LLMResponse:
        resp = self._script[min(self._idx, len(self._script) - 1)]
        self._idx += 1
        return resp

    async def text_chat_stream(self, *args, **kwargs):
        yield await self.text_chat(*args, **kwargs)


def _script() -> list[LLMResponse]:
    return [
        LLMResponse(
            role="assistant",
            tools_call_name=["Skill"],
            tools_call_args=[{"name": "greet"}],
            tools_call_ids=["c1"],
        ),
        LLMResponse(
            role="assistant",
            tools_call_name=["ping"],
            tools_call_args=[{}],
            tools_call_ids=["c2"],
        ),
        LLMResponse(role="assistant", completion_text="Done."),
    ]


def _seed_greet_skill() -> None:
    greet_dir = os.path.join(get_skills_dir(), "greet")
    os.makedirs(greet_dir, exist_ok=True)
    with open(os.path.join(greet_dir, "SKILL.md"), "w", encoding="utf-8") as f:
        f.write(GREET_SKILL_MD)


def _make_contribution() -> tuple[PluginContribution, list[str]]:
    """A plugin contributing one tool (``ping``) and one ``on_llm_request`` hook.

    Returns the contribution plus the shared list the hook appends to, so the caller can
    observe whether the hook fired.
    """
    fired: list[str] = []

    async def _ping(run_context, **kwargs) -> str:
        return "pong"

    async def _hook(run_context) -> None:
        fired.append("hook")

    class _DemoPlugin(Plugin):
        name, author, desc, version = "demo", "a", "d", "1"

    ping_tool = FunctionTool(
        name="ping",
        description="Reply pong.",
        parameters={"type": "object", "properties": {}},
        handler=_ping,
    )
    contribution = PluginContribution(
        plugin=_DemoPlugin.__new__(_DemoPlugin),
        tools=[ping_tool],
        hook_methods={"on_llm_request": [_hook]},
    )
    return contribution, fired


def _observe(run_context) -> dict:
    tool_results = [str(getattr(m, "content", "")) for m in run_context.messages if getattr(m, "role", None) == "tool"]
    system_msg = run_context.messages[0] if run_context.messages else None
    system_content = getattr(system_msg, "content", "") if system_msg else ""
    return {
        "inventory_injected": "greet" in system_content and "## Skills" in system_content,
        "skill_loaded": any("Say hello warmly." in t for t in tool_results),
        "plugin_tool_ran": any("pong" in t for t in tool_results),
    }


async def _run_oneshot() -> dict:
    """Tier 3: build_local_agent → agent.run(). No reset, no loop written by the caller."""
    contribution, fired = _make_contribution()
    provider = _ScriptedProvider(_script())
    agent = await build_local_agent(provider, prompt="greet me", contributions=[contribution])
    final = await agent.run(max_step=8)
    await agent.aclose()

    observed = _observe(agent.run_context)
    observed["tool_names"] = sorted(agent.basics.tools.names())
    observed["discovered_skills"] = sorted(s.name for s in agent.basics.skill_manager.list_skills())
    observed["plugin_hook_fired"] = fired == ["hook"] or "hook" in fired
    observed["final_text"] = final.completion_text if final else None
    return observed


async def _run_manual() -> dict:
    """Tier 1: hand-wire the same builders and drive the runner directly."""
    contribution, fired = _make_contribution()
    provider = _ScriptedProvider(_script())

    mgr = SkillManager()
    tools = ToolSet([build_skill_tool(mgr)])
    for tool in contribution.tools:
        tools.add_tool(tool)
    tools.merge(ToolSet(build_fs_tools(allowed_roots=[mgr.skills_root])))

    from agent_runtime.core.hooks_chain import ChainedAgentRunHooks
    from agent_runtime.extensions.plugins import CompositeAgentRunHooks

    hooks = ChainedAgentRunHooks(SkillsPromptHook(mgr), CompositeAgentRunHooks([contribution]))

    request = ProviderRequest(prompt="greet me", system_prompt="", func_tool=tools)
    run_context = ContextWrapper(context=SessionContext(session_id="manual"), messages=[])
    runner = ToolLoopAgentRunner()
    await runner.reset(
        provider=provider,
        request=request,
        run_context=run_context,
        tool_executor=FunctionToolExecutor(provider),
        agent_hooks=hooks,
    )
    async for _ in runner.step_until_done(max_step=8):
        pass
    final = runner.get_final_llm_resp()

    observed = _observe(run_context)
    observed["tool_names"] = sorted(tools.names())
    observed["discovered_skills"] = sorted(s.name for s in mgr.list_skills())
    observed["plugin_hook_fired"] = fired == ["hook"] or "hook" in fired
    observed["final_text"] = final.completion_text if final else None
    return observed


async def main() -> dict:
    data_dir = tempfile.mkdtemp(prefix="local_runtime_demo_")
    os.environ["AGENT_RUNTIME_DATA_DIR"] = data_dir
    _seed_greet_skill()

    oneshot = await _run_oneshot()
    manual = await _run_manual()

    equivalent = oneshot == manual
    result = {
        "oneshot": oneshot,
        "manual": manual,
        "equivalent": equivalent,
    }
    print(result)
    assert equivalent, "one-shot and hand-wired paths diverged"
    return result


if __name__ == "__main__":
    asyncio.run(main())
