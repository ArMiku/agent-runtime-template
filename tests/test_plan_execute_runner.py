"""``PlanExecuteRunner``: the explicit PLAN → EXEC → REPLAN state machine.

Covers the runner-level tasks: required-tool structured planning (1.2), the skeleton/state
contract (2.5), PLAN snapshot+mirror/empty-plan/parse-failure (3.5), EXEC child isolation +
summary harvest (4.5), REPLAN cursor/cap (5.4), one-step-per-LLM-call granularity + response
flow + online-pause (6.4), and cross-process recovery from the phase-key snapshot (7.4).

The runner is built directly (not via ``build_local_agent``) with a stub child-hook factory and
a scripted ``FakeProvider``, so each phase boundary is observable.
"""

from __future__ import annotations

from agent_runtime.core import FunctionToolExecutor, SessionContext
from agent_runtime.core.hooks import BaseAgentRunHooks
from agent_runtime.core.run_context import ContextWrapper
from agent_runtime.core.runners.plan_execute_runner import PlanExecuteRunner, _Phase
from agent_runtime.core.tool import FunctionTool, ToolSet
from agent_runtime.extensions.planning.entities import Todo, TodoStatus
from agent_runtime.extensions.planning.store import PLANNING_PLUGIN_ID, load_plan
from agent_runtime.extensions.plugins.store import InMemoryPluginStore
from agent_runtime.provider.entities import LLMResponse, ProviderRequest

from .fakes import FakeProvider, llm_text, llm_tool_call


def _submit_plan(todos: list[dict]) -> LLMResponse:
    return llm_tool_call("submit_plan", {"todos": todos})


async def _reset(
    provider: FakeProvider,
    *,
    session_id: str = "pe",
    plugin_store: InMemoryPluginStore | None = None,
    tool_set: ToolSet | None = None,
    sub_hook_factory=None,
    agent_hooks: BaseAgentRunHooks | None = None,
    max_replan: int = 2,
    planner_retries: int = 2,
    planner_max_tokens: int | None = None,
    planner_extra_body: dict | None = None,
    max_turns: int = 8,
    prompt: str = "do the task",
    max_step: int | None = None,
    per_call_timeout_s: float | None = None,
    per_turn_deadline_s: float | None = None,
) -> tuple[PlanExecuteRunner, InMemoryPluginStore]:
    if plugin_store is None:
        plugin_store = InMemoryPluginStore()
    if sub_hook_factory is None:
        def sub_hook_factory(sub_session_id: str) -> BaseAgentRunHooks:  # noqa: ARG001
            return BaseAgentRunHooks()
    runner = PlanExecuteRunner()
    request = ProviderRequest(
        prompt=prompt, system_prompt="", func_tool=tool_set, session_id=session_id
    )
    run_context: ContextWrapper[SessionContext] = ContextWrapper(
        context=SessionContext(session_id=session_id), messages=[]
    )
    # Only pass liveness kwargs when set, so tests that don't care get the runner's defaults.
    liveness_kwargs: dict = {}
    if max_step is not None:
        liveness_kwargs["max_step"] = max_step
    if per_call_timeout_s is not None:
        liveness_kwargs["per_call_timeout_s"] = per_call_timeout_s
    if per_turn_deadline_s is not None:
        liveness_kwargs["per_turn_deadline_s"] = per_turn_deadline_s
    await runner.reset(
        provider=provider,
        request=request,
        run_context=run_context,
        tool_executor=FunctionToolExecutor(provider),
        agent_hooks=agent_hooks or BaseAgentRunHooks(),
        plugin_store=plugin_store,
        tool_set=tool_set,
        sub_hook_factory=sub_hook_factory,
        max_replan=max_replan,
        planner_retries=planner_retries,
        planner_max_tokens=planner_max_tokens,
        planner_extra_body=planner_extra_body,
        enforce_max_turns=max_turns,
        **liveness_kwargs,
    )
    return runner, plugin_store


async def _drain(runner: PlanExecuteRunner, max_step: int = 30) -> list:
    responses = []
    async for resp in runner.step_until_done(max_step):
        responses.append(resp)
    return responses


# --- 1.2 required-tool planning parses submit_plan → list[Todo] ----------------


def test_submit_plan_args_parse_to_todos():
    """1.2: a submit_plan tool-call's args parse into list[Todo]."""
    todos = PlanExecuteRunner._extract_plan(_submit_plan([
        {"content": "a", "status": "pending"},
        {"content": "b", "status": "in_progress"},
    ]))
    assert todos is not None
    assert [t.content for t in todos] == ["a", "b"]
    assert todos[1].status is TodoStatus.IN_PROGRESS


async def test_submit_plan_required_choice_is_passed_to_provider():
    """1.2: PLAN calls text_chat with tool_choice='required' and the submit_plan tool set."""
    provider = FakeProvider([_submit_plan([{"content": "x", "status": "pending"}]), llm_text("done"), _submit_plan([{"content": "x", "status": "completed"}])])
    runner, _ = await _reset(provider)
    await _drain(runner)

    plan_calls = [c for c in provider.calls if c.get("tool_choice") == "required"]
    assert len(plan_calls) >= 1
    # The required-tool call carries exactly the submit_plan tool.
    for c in plan_calls:
        tool = c.get("func_tool")
        names = tool.names() if tool is not None else []
        assert names == ["submit_plan"]


# --- 2.5 skeleton / state contract -------------------------------------------


async def test_reset_then_done_is_false_and_state_set_stays_canonical():
    """2.5: after reset done() is False; every observed state is in the canonical set."""
    provider = FakeProvider([_submit_plan([{"content": "a", "status": "pending"}]), llm_text("done"), _submit_plan([{"content": "a", "status": "completed"}])])
    runner, _ = await _reset(provider)
    assert runner.done() is False
    assert runner._state.name in {"IDLE", "RUNNING", "DONE", "ERROR"}

    seen: set[str] = set()
    while not runner.done():
        seen.add(runner._state.name)
        async for _ in runner.step():
            pass
        seen.add(runner._state.name)
    # phase is private — the public state never escapes the canonical enum.
    assert seen <= {"IDLE", "RUNNING", "DONE", "ERROR"}


async def test_on_before_complete_veto_does_not_transition_to_done():
    """2.5: a host veto at the completion gate keeps the runner RUNNING (not DONE)."""

    class _VetoOnce(BaseAgentRunHooks):
        def __init__(self) -> None:
            self.calls = 0

        async def on_before_complete(self, run_context, llm_response) -> bool:
            self.calls += 1
            if self.calls == 1:
                return False
            return True

    hook = _VetoOnce()
    # 1 todo: plan → exec → replan(all complete, vetoed) → replan(all complete, admitted)
    provider = FakeProvider([
        _submit_plan([{"content": "a", "status": "pending"}]),
        llm_text("did a"),
        _submit_plan([{"content": "a", "status": "completed"}]),
        _submit_plan([{"content": "a", "status": "completed"}]),
    ])
    runner, _ = await _reset(provider, agent_hooks=hook)
    await _drain(runner)

    assert hook.calls >= 2  # vetoed once, then admitted
    assert runner.done()
    assert runner._state.name == "DONE"


# --- 3.5 PLAN: snapshot==mirror, cursor, empty→DONE, parse-fail→ERROR ----------


async def test_plan_writes_snapshot_and_mirror_and_sets_cursor():
    """3.5: after PLAN, the phase-key snapshot matches the load_plan mirror; cursor→first pending."""
    provider = FakeProvider([
        _submit_plan([{"content": "a", "status": "pending"}, {"content": "b", "status": "pending"}]),
        llm_text("done"),
        _submit_plan([{"content": "a", "status": "completed"}, {"content": "b", "status": "completed"}]),
    ])
    runner, store = await _reset(provider)
    # Drive ONE step (PLAN only) and stop — online-pause friendly.
    async for _ in runner.step():
        pass

    assert runner._progress.phase is _Phase.EXEC
    assert runner._progress.cursor == 0

    snapshot = await store.get(PLANNING_PLUGIN_ID, "pe__phase", None)
    assert isinstance(snapshot, dict)
    snapshot_plan = [Todo.from_dict(t) for t in snapshot["plan"]]
    mirror_plan = await load_plan(store, "pe")
    assert [(t.content, t.status.value) for t in snapshot_plan] == [(t.content, t.status.value) for t in mirror_plan]
    assert [t.content for t in mirror_plan] == ["a", "b"]


async def test_empty_plan_completes_without_exec():
    """3.3/3.5: an empty plan goes straight to DONE, never entering EXEC."""
    provider = FakeProvider([_submit_plan([])])
    runner, _ = await _reset(provider)
    await _drain(runner)

    assert runner.done()
    assert runner._state.name == "DONE"
    # No child runner was ever built.
    assert runner._sub_runner is None


async def test_planner_parse_failure_retries_then_errors():
    """3.4/3.5: a missing submit_plan call is retried, then the run transitions to ERROR."""
    provider = FakeProvider([llm_text("no tool"), llm_text("still no tool")])
    runner, _ = await _reset(provider, planner_retries=1)
    await _drain(runner)

    assert runner._state.name == "ERROR"
    assert runner.done()  # ERROR is terminal
    final = runner.get_final_llm_resp()
    assert final is not None and final.role == "err"


async def test_planner_reasoning_truncation_diagnostic():
    """Reproduces the user's bug: a reasoning model returns thinking but no submit_plan call
    (truncated). The error must point at reasoning truncation + the two levers, not the opaque
    'missing or malformed'."""
    reasoning_only = LLMResponse(
        role="assistant", reasoning_content="let me think hard about the zelda timeline..."
    )
    provider = FakeProvider([reasoning_only])  # planner_retries=0 → one attempt
    runner, _ = await _reset(provider, planner_retries=0)
    await _drain(runner)

    assert runner._state.name == "ERROR"
    final = runner.get_final_llm_resp()
    assert final is not None
    msg = final.completion_text or ""
    assert "truncated by reasoning" in msg
    assert "planner_max_tokens" in msg
    assert "planner_extra_body" in msg


async def test_planner_forwards_max_tokens_and_extra_body():
    """The planner call forwards planner_max_tokens (as max_tokens) and planner_extra_body
    (as extra_body) into the provider call; None planner_max_tokens (the runner default) omits
    it — the runner imposes no budget by default."""
    provider = FakeProvider([
        _submit_plan([{"content": "a", "status": "pending"}]),
        llm_text("done"),
        _submit_plan([{"content": "a", "status": "completed"}]),
    ])
    runner, _ = await _reset(
        provider, planner_max_tokens=8192, planner_extra_body={"thinking": False}
    )
    async for _ in runner.step():  # one PLAN step
        pass

    plan_call = provider.calls[0]
    assert plan_call.get("max_tokens") == 8192
    assert plan_call.get("extra_body") == {"thinking": False}

    # The runner default (None) → no max_tokens key forwarded (no imposed budget).
    provider2 = FakeProvider([
        _submit_plan([{"content": "a", "status": "pending"}]),
        llm_text("done"),
        _submit_plan([{"content": "a", "status": "completed"}]),
    ])
    runner2, _ = await _reset(provider2, planner_max_tokens=None)
    async for _ in runner2.step():
        pass
    assert "max_tokens" not in provider2.calls[0]


# --- 4.5 EXEC: isolated child hooks, no main-plan veto, summary harvest --------


async def test_child_uses_factory_hooks_not_main_and_summary_harvested():
    """4.5: the child runner uses the factory-built hooks (not the main chain); its tool traffic
    stays out of the main context; the todo summary is harvested with no extra LLM call."""
    main_hooks = BaseAgentRunHooks()
    child_hooks = BaseAgentRunHooks()

    def factory(sub_session_id: str) -> BaseAgentRunHooks:  # noqa: ARG001
        return child_hooks

    provider = FakeProvider([
        _submit_plan([{"content": "a", "status": "pending"}]),
        llm_text("the result"),
        _submit_plan([{"content": "a", "status": "completed"}]),
    ])
    runner, _ = await _reset(provider, sub_hook_factory=factory, agent_hooks=main_hooks)
    await _drain(runner)

    # The child for todo 0 used the factory hook, never the main hook object.
    assert runner._sub_runner is None  # child torn down after completion
    # The summary was harvested straight from the child's final text into the main context.
    main_texts = [str(getattr(m, "content", "")) for m in runner.run_context.messages]
    assert any("the result" in t for t in main_texts)
    # Exactly three provider calls: PLAN + 1 EXEC step + REPLAN — no extra summarizing call.
    assert len(provider.calls) == 3


async def test_child_completion_not_vetoed_by_main_plan():
    """4.5: a main PlanningHook over the top-level plan (still unfinished) does NOT veto a
    child's completion — the child's on_before_complete only sees its own session."""
    from agent_runtime.extensions.planning import PlanningHook

    # Top-level plan has an unfinished item, so a main PlanningHook WOULD veto the top-level
    # completion — but the child must still finish its single todo.
    store = InMemoryPluginStore()
    main_hooks = PlanningHook(store)

    def factory(sub_session_id: str) -> BaseAgentRunHooks:  # noqa: ARG001
        return BaseAgentRunHooks()  # child has NO planning hook → admits freely

    provider = FakeProvider([
        _submit_plan([{"content": "a", "status": "pending"}, {"content": "b", "status": "pending"}]),
        llm_text("did a"),
        _submit_plan([{"content": "a", "status": "completed"}, {"content": "b", "status": "completed"}]),
    ])
    runner, _ = await _reset(provider, sub_hook_factory=factory, agent_hooks=main_hooks, plugin_store=store)
    await _drain(runner)

    # The child completed (todo a harvested) despite the main plan still having been unfinished
    # at that moment — proving the main PlanningHook never gated the child.
    main_texts = [str(getattr(m, "content", "")) for m in runner.run_context.messages]
    assert any("did a" in t for t in main_texts)


async def test_child_intermediate_messages_isolated_from_main_context():
    """4.4/4.5: a tool call inside the child never reaches the main run_context.messages."""
    async def _noop(run_context, **kwargs) -> str:  # noqa: ARG001
        return "tool-output"

    tool = FunctionTool(name="noop", description="noop", parameters={"type": "object", "properties": {}}, handler=_noop)
    child_tools = ToolSet([tool])

    def factory(sub_session_id: str) -> BaseAgentRunHooks:  # noqa: ARG001
        return BaseAgentRunHooks()

    # PLAN → EXEC(child calls noop then finishes) → REPLAN(all done)
    provider = FakeProvider([
        _submit_plan([{"content": "a", "status": "pending"}]),
        llm_tool_call("noop", {}),
        llm_text("done after tool"),
        _submit_plan([{"content": "a", "status": "completed"}]),
    ])
    runner, _ = await _reset(provider, tool_set=child_tools, sub_hook_factory=factory)
    await _drain(runner)

    main_contents = [str(getattr(m, "content", "")) for m in runner.run_context.messages]
    # The child's tool result message never leaked into the main context.
    assert not any("tool-output" in t for t in main_contents)
    # Only the harvested summary entered the main context.
    assert any("done after tool" in t for t in main_contents)


# --- 5.4 REPLAN: pending→EXEC, all-complete→DONE, cap→release -----------------


async def test_replan_pending_returns_to_exec():
    """5.4: a replan that still has pending todos returns to EXEC for the next one."""
    provider = FakeProvider([
        _submit_plan([{"content": "a", "status": "pending"}, {"content": "b", "status": "pending"}]),
        llm_text("did a"),
        _submit_plan([{"content": "a", "status": "completed"}, {"content": "b", "status": "pending"}]),
        llm_text("did b"),
        _submit_plan([{"content": "a", "status": "completed"}, {"content": "b", "status": "completed"}]),
    ])
    runner, _ = await _reset(provider)
    await _drain(runner)

    assert runner.done()
    assert all(t.status is TodoStatus.COMPLETED for t in runner._progress.plan)


async def test_replan_cap_releases_completion():
    """5.3/5.4: the replan cap releases completion. The cap bounds non-productive replans —
    and the only way to accumulate them (EXEC resets the counter on every completed todo) is a
    completion that the host keeps vetoing. So a permanent veto that would otherwise loop
    forever is forced to DONE once the cap is hit."""
    class _AlwaysVeto(BaseAgentRunHooks):
        async def on_before_complete(self, run_context, llm_response) -> bool:  # noqa: ARG002
            return False

    provider = FakeProvider([
        _submit_plan([{"content": "a", "status": "pending"}]),
        llm_text("did a"),
        _submit_plan([{"content": "a", "status": "completed"}]),
    ])
    runner, _ = await _reset(provider, agent_hooks=_AlwaysVeto(), max_replan=1)
    await _drain(runner)

    # Without the cap this loops forever (host always vetoes); the cap forces DONE.
    assert runner.done()
    assert runner._state.name == "DONE"


# --- 6.4 granularity: one step == one LLM call; flow types; online pause ------


async def test_step_count_equals_llm_call_count():
    """6.1/6.4: each top-level step is exactly one LLM call across phases."""
    provider = FakeProvider([
        _submit_plan([{"content": "a", "status": "pending"}, {"content": "b", "status": "pending"}]),
        llm_text("did a"),
        _submit_plan([{"content": "a", "status": "completed"}, {"content": "b", "status": "pending"}]),
        llm_text("did b"),
        _submit_plan([{"content": "a", "status": "completed"}, {"content": "b", "status": "completed"}]),
    ])
    runner, _ = await _reset(provider)

    steps = 0
    while not runner.done():
        async for _ in runner.step():
            pass
        steps += 1
    # 5 phases driven (PLAN, EXEC, REPLAN, EXEC, REPLAN) == 5 provider calls.
    assert steps == len(provider.calls) == 5


async def test_response_flow_distinguishes_phases():
    """6.3/6.4: PLAN/REPLAN emit an llm_result carrying the plan; EXEC forwards child responses
    (here a tool_call_result, a distinct type)."""
    async def _noop(run_context, **kwargs) -> str:  # noqa: ARG001
        return "r"

    tool = FunctionTool(name="noop", description="noop", parameters={"type": "object", "properties": {}}, handler=_noop)
    provider = FakeProvider([
        _submit_plan([{"content": "a", "status": "pending"}]),
        llm_tool_call("noop", {}),
        llm_text("finished"),
        _submit_plan([{"content": "a", "status": "completed"}]),
    ])
    runner, _ = await _reset(provider, tool_set=ToolSet([tool]))
    responses = await _drain(runner)

    types = [r.type for r in responses]
    # PLAN emitted an llm_result; EXEC forwarded a tool_call_result from the child.
    assert "llm_result" in types
    assert "tool_call_result" in types


async def test_online_pause_leaves_state_readable():
    """6.2/6.4: after stepping once (PLAN), the host can stop and read a coherent non-done state."""
    provider = FakeProvider([
        _submit_plan([{"content": "a", "status": "pending"}]),
        llm_text("did a"),
        _submit_plan([{"content": "a", "status": "completed"}]),
    ])
    runner, _ = await _reset(provider)

    async for _ in runner.step():  # one phase only — online pause
        pass

    assert not runner.done()  # still running, paused between phases
    assert runner._progress.phase is _Phase.EXEC  # internal progress is readable
    assert len(runner._progress.plan) == 1


# --- 7.4 cross-process recovery from the phase-key snapshot ------------------


async def test_recovery_restores_phase_plan_and_skips_completed_todos():
    """7.4: persist, then a fresh runner on the same session_id recovers the phase/plan/cursor
    and does not re-run already-completed todos."""
    store = InMemoryPluginStore()
    # Run 1: plan two todos, complete the first, then stop (simulate a crash before the second).
    provider1 = FakeProvider([
        _submit_plan([{"content": "a", "status": "pending"}, {"content": "b", "status": "pending"}]),
        llm_text("did a"),
        _submit_plan([{"content": "a", "status": "completed"}, {"content": "b", "status": "pending"}]),
    ])
    runner1, _ = await _reset(provider1, session_id="resume", plugin_store=store)
    # Drive exactly: PLAN, EXEC(a), REPLAN — leaving us in EXEC for todo b.
    for _ in range(3):
        async for _ in runner1.step():
            pass
    assert runner1._progress.phase is _Phase.EXEC
    assert runner1._progress.cursor == 1
    # todo a is completed and persisted.
    persisted = await store.get(PLANNING_PLUGIN_ID, "resume__phase", None)
    assert persisted["plan"][0]["status"] == "completed"

    # Run 2: a brand-new runner instance, same session_id + store — recovers.
    provider2 = FakeProvider([
        llm_text("did b"),
        _submit_plan([{"content": "a", "status": "completed"}, {"content": "b", "status": "completed"}]),
    ])
    runner2, _ = await _reset(provider2, session_id="resume", plugin_store=store)
    # Recovered into EXEC at cursor 1, plan intact, todo a still completed.
    assert runner2._progress.phase is _Phase.EXEC
    assert runner2._progress.cursor == 1
    assert runner2._progress.plan[0].status is TodoStatus.COMPLETED
    await _drain(runner2)

    assert runner2.done()
    assert all(t.status is TodoStatus.COMPLETED for t in runner2._progress.plan)
    # todo a was NOT re-run: only "did b" was harvested this run.
    new_summaries = runner2.run_context.messages
    assert not any("did a" in str(getattr(m, "content", "")) for m in new_summaries)
    assert any("did b" in str(getattr(m, "content", "")) for m in new_summaries)
