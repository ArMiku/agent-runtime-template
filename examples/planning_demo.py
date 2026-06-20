"""End-to-end example: the planning extension driving a multi-step ReAct run.

Proves the plan-and-execute closed loop emerges from the existing ReAct loop with no
control-flow changes:

* **plan write** — the model calls ``write_todos`` to create a plan; it lands in
  independent state (the ``PluginStore``), keyed by ``session_id``, not in the message
  stream;
* **plan injection** — ``PlanningHook`` injects the live plan into the leading system
  message (sentinel-delimited) before each LLM step, so the model always sees current
  progress even after context compaction;
* **premature-finish veto** — when the model tries to finish with a todo still
  unfinished, the kernel's ``on_before_complete`` hook refuses the completion and the run
  continues; once the plan is all-completed, the finish is admitted.

Run it::

    python -m examples.planning_demo
"""

from __future__ import annotations

import asyncio

from agent_runtime.core import FunctionToolExecutor, SessionContext
from agent_runtime.core.run_context import ContextWrapper
from agent_runtime.core.runners.tool_loop_agent_runner import ToolLoopAgentRunner
from agent_runtime.extensions.planning import load_plan
from agent_runtime.extensions.plugins.store import InMemoryPluginStore
from agent_runtime.local_runtime import build_local_agent_basics
from agent_runtime.provider.entities import LLMResponse, ProviderRequest
from agent_runtime.provider.provider import Provider

# Sentinel the planning hook wraps the live plan in (one segment per system message).
PLAN_SENTINEL = "<!-- todo-state -->"


class _ScriptedProvider(Provider):
    """Returns scripted replies and records the system text it last chatted over."""

    def __init__(self, script: list[LLMResponse]) -> None:
        super().__init__({"id": "demo", "type": "demo", "max_context_tokens": 0, "modalities": []}, {})
        self._script = list(script)
        self._idx = 0
        self.last_system_text = ""

    def get_current_key(self) -> str:
        return "demo-key"

    def set_key(self, key: str) -> None: ...

    async def get_models(self) -> list[str]:
        return ["demo-model"]

    async def text_chat(self, *args, **kwargs) -> LLMResponse:
        contexts = kwargs.get("contexts") or []
        self.last_system_text = "\n".join(
            str(getattr(m, "content", "")) for m in contexts if getattr(m, "role", None) == "system"
        )
        resp = self._script[min(self._idx, len(self._script) - 1)]
        self._idx += 1
        return resp

    async def text_chat_stream(self, *args, **kwargs):
        yield await self.text_chat(*args, **kwargs)


def _write_todos(todos: list[dict]) -> LLMResponse:
    """A scripted ``write_todos`` tool call carrying the full plan."""
    return LLMResponse(
        role="assistant",
        tools_call_name=["write_todos"],
        tools_call_args=[{"todos": todos}],
        tools_call_ids=["call_plan"],
    )


async def main() -> dict:
    # A persistent store would be host-injected here; InMemory keeps the example local.
    store = InMemoryPluginStore()
    basics = build_local_agent_basics(include_fs=False, include_planning=True, plugin_store=store)

    # Script: plan with two steps → finish the first → PREMATURELY try to finish (vetoed)
    # → mark all complete → finish (admitted).
    provider = _ScriptedProvider(
        [
            _write_todos(
                [
                    {"content": "Research the topic", "status": "in_progress"},
                    {"content": "Write the summary", "status": "pending"},
                ]
            ),
            _write_todos(
                [
                    {"content": "Research the topic", "status": "completed"},
                    {"content": "Write the summary", "status": "in_progress"},
                ]
            ),
            LLMResponse(role="assistant", completion_text="All set!"),  # premature → vetoed
            _write_todos(
                [
                    {"content": "Research the topic", "status": "completed"},
                    {"content": "Write the summary", "status": "completed"},
                ]
            ),
            LLMResponse(role="assistant", completion_text="Done — summary delivered."),  # admitted
        ]
    )

    request = ProviderRequest(prompt="research and summarize", system_prompt="", func_tool=basics.tools)
    run_context = ContextWrapper(context=SessionContext(session_id="planning-demo"), messages=[])
    runner = ToolLoopAgentRunner()
    await runner.reset(
        provider=provider,
        request=request,
        run_context=run_context,
        tool_executor=FunctionToolExecutor(provider),
        agent_hooks=basics.hooks,
    )

    # Count completions the model attempted vs. how many steps actually ran: the veto
    # forces an extra round, so the no-tool "All set!" does not end the run.
    steps = 0
    async for _ in runner.step_until_done(max_step=20):
        steps += 1

    final = runner.get_final_llm_resp()
    final_plan = await load_plan(store, "planning-demo")

    result = {
        "final_text": final.completion_text if final else None,
        "plan_injected_in_system_message": PLAN_SENTINEL in provider.last_system_text,
        "final_plan": [(t.content, t.status.value) for t in final_plan],
        "all_completed": all(t.status.value == "completed" for t in final_plan),
    }
    print(result)
    return result


if __name__ == "__main__":
    asyncio.run(main())
