"""End-to-end example: the plan-and-execute runner driving a multi-step task.

Proves the explicit ``PLAN → EXEC → REPLAN`` state machine (route 1) end to end, orthogonal to
the emergent-todo planning route:

* **PLAN** — one ``tool_choice="required"`` call forces a ``submit_plan`` tool call; its args
  become the todo plan, snapshotted to the phase key and mirrored via ``save_plan``.
* **EXEC** — each todo is delegated to an isolated child ReAct runner; only the todo's result
  summary is harvested back into the main context (the child's tool traffic stays isolated).
* **REPLAN** — one structured call revises the plan; the cursor advances, or the run completes.
* **Recovery** — a second agent built on the same ``session_id`` + persistent store resumes from
  the phase-key snapshot, skipping already-completed todos.

Run it::

    python -m examples.plan_execute_demo
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Sequence

from agent_runtime.extensions.planning import load_plan
from agent_runtime.extensions.plugins.store import InMemoryPluginStore
from agent_runtime.local_runtime import build_local_agent
from agent_runtime.provider.entities import LLMResponse
from agent_runtime.provider.provider import Provider

SESSION_ID = "plan-execute-demo"


class _ScriptedProvider(Provider):
    """Returns scripted replies in call order (planner, executor, replanner all draw from one queue)."""

    def __init__(self, script: Sequence[LLMResponse]) -> None:
        super().__init__({"id": "demo", "type": "demo", "max_context_tokens": 0, "modalities": []}, {})
        self._script = list(script)
        self._idx = 0

    def get_current_key(self) -> str:
        return "demo-key"

    def set_key(self, key: str) -> None: ...

    async def get_models(self) -> list[str]:
        return ["demo-model"]

    async def text_chat(self, *args, **kwargs) -> LLMResponse:  # noqa: ARG002
        resp = self._script[min(self._idx, len(self._script) - 1)]
        self._idx += 1
        return resp

    async def text_chat_stream(self, *args, **kwargs) -> AsyncGenerator[LLMResponse, None]:  # noqa: ARG002
        yield await self.text_chat()


def _submit_plan(todos: list[dict]) -> LLMResponse:
    return LLMResponse(
        role="assistant",
        tools_call_name=["submit_plan"],
        tools_call_args=[{"todos": todos}],
        tools_call_ids=["call_plan"],
    )


def _plan(research_status: str, summary_status: str) -> LLMResponse:
    return _submit_plan([
        {"content": "Research the topic", "status": research_status},
        {"content": "Write the summary", "status": summary_status},
    ])


async def _print_phase_evolution(agent, max_step: int) -> list:
    """Drive the agent, printing each response's type (so phase evolution is visible)."""
    events: list[str] = []
    step = 0
    async for resp in agent.step_until_done(max_step=max_step):
        step += 1
        text = ""
        chain = resp.data.get("chain") if isinstance(resp.data, dict) else None
        if chain is not None:
            text = chain.get_plain_text()[:60].replace("\n", " ")
        label = {"llm_result": "plan/exec result", "tool_call_result": "exec tool"}.get(resp.type, resp.type)
        print(f"  step {step:>2} [{resp.type:<16} {label}]: {text}")
        events.append(resp.type)
    return events


async def main() -> dict:
    store = InMemoryPluginStore()

    # --- Run 1: a full plan-execute run of a two-step task. ------------------
    # Call order: PLAN(submit_plan) → EXEC(research result) → REPLAN → EXEC(summary result)
    # → REPLAN(all complete) → DONE.
    provider = _ScriptedProvider([
        _plan("pending", "pending"),
        LLMResponse(role="assistant", completion_text="Found 3 authoritative sources on the topic."),
        _plan("completed", "pending"),
        LLMResponse(role="assistant", completion_text="Summary: the topic has three key aspects."),
        _plan("completed", "completed"),
    ])
    agent = await build_local_agent(
        provider,
        prompt="research the topic and write a summary",
        session_id=SESSION_ID,
        include_fs=False,
        runner_type="plan_execute",
        plugin_store=store,
    )
    print("=== Run 1: plan-execute (PLAN → EXEC → REPLAN → EXEC → REPLAN → DONE) ===")
    events = await _print_phase_evolution(agent, max_step=20)

    final = agent.get_final_llm_resp()
    final_plan = await load_plan(store, SESSION_ID)
    result = {
        "events": events,
        "final_text": final.completion_text if final else None,
        "final_plan": [(t.content, t.status.value) for t in final_plan],
        "all_completed": all(t.status.value == "completed" for t in final_plan),
    }
    print("\nRun 1 result:", result)

    # --- Run 2: recovery. A fresh agent on the same session_id + store resumes from the
    # persisted phase-key snapshot. The prior run already finished, so recovery re-loads the
    # completed plan and the new agent confirms completion in a single replan step — it does
    # NOT re-PLAN or re-EXECUTE the todos. Progress survived in the store, not the process.
    print("\n=== Run 2: recovery (same session_id + store, fresh process/agent) ===")
    provider2 = _ScriptedProvider([_plan("completed", "completed")])
    agent2 = await build_local_agent(
        provider2,
        prompt="research the topic and write a summary",
        session_id=SESSION_ID,
        include_fs=False,
        runner_type="plan_execute",
        plugin_store=store,
    )
    recovered_plan = await load_plan(store, SESSION_ID)
    print("Recovered plan from store:", [(t.content, t.status.value) for t in recovered_plan])
    recovery_steps = await _print_phase_evolution(agent2, max_step=10)
    print(f"Recovery finished in {len(recovery_steps)} step(s) (no re-PLAN/re-EXEC of finished todos).")
    return result


if __name__ == "__main__":
    asyncio.run(main())
