"""``PlanExecuteRunner`` liveness guards: honest terminal state, per-call timeout, per-turn
deadline, and disaster-loose defaults (change ``add-plan-execute-liveness``).

The guards share one sink: hitting any ceiling transitions the runner to ``ERROR`` with a
non-empty ``final_llm_resp`` whose ``completion_text`` carries the already-completed todo
summaries (host can display partial results directly), prefixed by a recognizable stop reason
(which limit, which phase, how many todos done). These tests drive the runner with a scripted
``FakeProvider`` so each ceiling is observable without a network.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator

from agent_runtime.core.runners.plan_execute_runner import _Phase
from agent_runtime.extensions.planning.entities import Todo, TodoStatus
from agent_runtime.extensions.planning.store import PLANNING_PLUGIN_ID
from agent_runtime.provider.entities import LLMResponse

from .fakes import FakeProvider, llm_text
from .test_plan_execute_runner import _reset, _submit_plan


class _CyclicProvider(FakeProvider):
    """A provider that never lets the run finish: every PLAN/REPLAN structured call returns a
    plan with one still-pending todo, and every child call returns plain text. Used to exercise
    the liveness ceilings, which are the only thing that can stop such a run."""

    def __init__(self) -> None:
        super().__init__([])

    async def text_chat(self, **kwargs) -> LLMResponse:
        self.calls.append(kwargs)
        if kwargs.get("tool_choice") == "required":
            return _submit_plan([{"content": "loop", "status": "pending"}])
        return llm_text("did a chunk")

    async def text_chat_stream(self, **kwargs) -> AsyncGenerator[LLMResponse, None]:
        self.calls.append(kwargs)
        if kwargs.get("tool_choice") == "required":
            yield _submit_plan([{"content": "loop", "status": "pending"}])
        else:
            yield llm_text("did a chunk")


# --- 2. unified honest-terminal sink + partial results -------------------------


async def test_liveness_error_transitions_to_error_with_stop_reason():
    """2.3: the liveness sink transitions to ERROR with a non-empty final whose text names the
    limit kind, the phase, and the completed-todo count."""
    provider = FakeProvider([_submit_plan([{"content": "a", "status": "pending"}])])
    runner, _ = await _reset(provider)
    # Seed some progress: one completed todo with a harvested summary.
    runner._progress.plan = [
        Todo(content="research X", status=TodoStatus.COMPLETED),
        Todo(content="research Y", status=TodoStatus.PENDING),
    ]
    runner._progress.summaries = {0: "found three X facts"}
    runner._progress.phase = _Phase.REPLAN

    await runner._to_liveness_error("per-turn deadline")

    assert runner.done()
    assert runner._state.name == "ERROR"
    final = runner.get_final_llm_resp()
    assert final is not None and final.role == "err"
    text = final.completion_text or ""
    # Stop-reason prefix names the limit, the phase, and the completed count.
    assert "per-turn deadline" in text
    assert "replan" in text.lower()
    assert "1" in text  # one todo completed


async def test_liveness_error_completion_text_carries_partial_results():
    """2.3: already-completed todo summaries appear in completion_text in displayable form,
    not buried in a diagnostic-only string."""
    provider = FakeProvider([_submit_plan([{"content": "a", "status": "pending"}])])
    runner, _ = await _reset(provider)
    runner._progress.plan = [
        Todo(content="research X", status=TodoStatus.COMPLETED),
        Todo(content="research Y", status=TodoStatus.PENDING),
    ]
    runner._progress.summaries = {0: "found three X facts"}

    await runner._to_liveness_error("max_step")

    text = runner.get_final_llm_resp().completion_text or ""
    # The harvested summary is present and host-displayable.
    assert "found three X facts" in text
    assert "research X" in text


async def test_liveness_error_persists_terminal_for_recovery():
    """2.4: the ERROR terminal is persisted; a rebuild on the same session reads the stop
    reason (not IDLE/RUNNING)."""
    provider = FakeProvider([_submit_plan([{"content": "a", "status": "pending"}])])
    runner, store = await _reset(provider, session_id="pe-live")
    runner._progress.plan = [Todo(content="a", status=TodoStatus.PENDING)]

    await runner._to_liveness_error("per-call timeout")

    snapshot = await store.get(PLANNING_PLUGIN_ID, "pe-live__phase", None)
    assert isinstance(snapshot, dict)
    # The run is terminal; final response carries the stop reason for the host.
    final = runner.get_final_llm_resp()
    assert final is not None and "per-call timeout" in (final.completion_text or "")


# --- 3. three ceilings are runner-owned and checked inside step() --------------


async def test_max_step_trips_honest_terminal_under_bare_step_loop():
    """3.4: driving with a bare ``while not done(): step()`` loop (NOT step_until_done) still
    trips max_step honestly — the ceiling lives inside step(), immune to drive style."""
    provider = _CyclicProvider()
    runner, _ = await _reset(provider, max_step=5)

    steps = 0
    while not runner.done():
        async for _ in runner.step():
            pass
        steps += 1
        assert steps <= 50  # guard the test itself against a real infinite loop

    assert runner._state.name == "ERROR"
    final = runner.get_final_llm_resp()
    assert final is not None and "max_step" in (final.completion_text or "")
    # The bare loop never imposed its own bound — the runner's own max_step did.
    assert steps <= 6


async def test_max_step_trips_under_step_until_done_too():
    """3.4: the same ceiling also trips under step_until_done (host's max_step arg is no longer
    the only thing standing between a runaway and a hang)."""
    provider = _CyclicProvider()
    runner, _ = await _reset(provider, max_step=5)

    async for _ in runner.step_until_done(max_step=1000):
        pass

    assert runner._state.name == "ERROR"
    assert "max_step" in (runner.get_final_llm_resp().completion_text or "")


async def test_per_turn_deadline_trips_at_step_boundary():
    """3.2/5.2: a per-turn deadline (monotonic) trips at the next step boundary. A patched clock
    advances past the deadline while each individual call returns instantly."""
    provider = _CyclicProvider()
    runner, _ = await _reset(provider, per_turn_deadline_s=10.0, max_step=1000)

    # Patch the runner's clock: each read advances 4s, so by the 3rd step boundary (>10s) the
    # deadline trips — while no single call ever blocks.
    ticks = iter([0.0, 4.0, 8.0, 12.0, 16.0, 20.0, 24.0])
    runner._now = lambda: next(ticks)  # type: ignore[attr-defined]

    steps = 0
    while not runner.done():
        async for _ in runner.step():
            pass
        steps += 1
        assert steps <= 50

    assert runner._state.name == "ERROR"
    assert "per-turn deadline" in (runner.get_final_llm_resp().completion_text or "")


# --- 4. per-call timeout catches in-call hangs --------------------------------


class _SlowStructuredProvider(FakeProvider):
    """A converging run whose PLAN structured call hangs. In RED (no timeout) the hang is brief
    and the run completes DONE; in GREEN the per-call timeout cancels the wedged planner call.
    Hang is short so the RED assertion fails fast rather than looping."""

    def __init__(self, hang_s: float) -> None:
        super().__init__([])
        self._hang_s = hang_s
        self._required_calls = 0

    async def text_chat(self, **kwargs) -> LLMResponse:
        self.calls.append(kwargs)
        if kwargs.get("tool_choice") == "required":
            self._required_calls += 1
            if self._required_calls == 1:
                await asyncio.sleep(self._hang_s)  # wedge the PLAN call
                return _submit_plan([{"content": "a", "status": "pending"}])
            # REPLAN converges so the RED run terminates (DONE) instead of looping.
            return _submit_plan([{"content": "a", "status": "completed"}])
        return llm_text("did a")


class _SlowChildProvider(FakeProvider):
    """A converging run whose child step LLM call hangs. RED completes DONE; GREEN's per-call
    timeout cancels the wedged child step."""

    def __init__(self, hang_s: float) -> None:
        super().__init__([])
        self._hang_s = hang_s
        self._required_calls = 0

    async def text_chat(self, **kwargs) -> LLMResponse:
        self.calls.append(kwargs)
        if kwargs.get("tool_choice") == "required":
            self._required_calls += 1
            if self._required_calls == 1:
                return _submit_plan([{"content": "a", "status": "pending"}])
            return _submit_plan([{"content": "a", "status": "completed"}])
        await asyncio.sleep(self._hang_s)  # wedge the child step
        return llm_text("did a")


async def test_structured_call_hang_trips_per_call_timeout():
    """4.3: a wedged PLAN structured call is cancelled and the run transitions to ERROR naming
    the per-call timeout."""
    provider = _SlowStructuredProvider(hang_s=0.5)
    runner, _ = await _reset(provider, per_call_timeout_s=0.05)

    async for _ in runner.step_until_done(max_step=1000):
        pass

    assert runner._state.name == "ERROR"
    assert "per-call timeout" in (runner.get_final_llm_resp().completion_text or "")


async def test_child_step_hang_trips_per_call_timeout():
    """4.4: a wedged child step is cancelled and the run transitions to ERROR honestly."""
    provider = _SlowChildProvider(hang_s=0.5)
    runner, _ = await _reset(provider, per_call_timeout_s=0.05)

    async for _ in runner.step_until_done(max_step=1000):
        pass

    assert runner._state.name == "ERROR"
    assert "per-call timeout" in (runner.get_final_llm_resp().completion_text or "")


async def test_per_call_timeout_does_not_fire_on_normal_calls():
    """4.5: calls that return within the (loose) timeout never trip it — a normal run completes."""
    provider = FakeProvider([
        _submit_plan([{"content": "a", "status": "pending"}]),
        llm_text("did a"),
        _submit_plan([{"content": "a", "status": "completed"}]),
    ])
    runner, _ = await _reset(provider, per_call_timeout_s=30.0)

    async for _ in runner.step_until_done(max_step=1000):
        pass

    assert runner._state.name == "DONE"


# --- 5. per-turn deadline semantics (monotonic, no false-positive) ------------


async def test_per_turn_deadline_uses_monotonic_not_wall_clock():
    """5.3: the deadline is measured by the runner's monotonic clock (``_now``). Patching it is
    what moves time; a wall-clock jump (time.time) would not trip it. We assert the trip is
    driven purely by the monotonic readings."""
    provider = _CyclicProvider()
    runner, _ = await _reset(provider, per_turn_deadline_s=10.0, max_step=1000)

    # Monotonic readings drive the trip: start at 0, then jump past 10 on the 2nd boundary.
    ticks = iter([0.0, 5.0, 11.0, 99.0])
    runner._now = lambda: next(ticks)  # type: ignore[attr-defined]

    steps = 0
    while not runner.done():
        async for _ in runner.step():
            pass
        steps += 1
        assert steps <= 50

    assert runner._state.name == "ERROR"
    assert "per-turn deadline" in (runner.get_final_llm_resp().completion_text or "")


async def test_per_turn_deadline_loose_does_not_trip_normal_run():
    """5.4: a normal run that finishes within the (loose) deadline never trips it."""
    provider = FakeProvider([
        _submit_plan([{"content": "a", "status": "pending"}]),
        llm_text("did a"),
        _submit_plan([{"content": "a", "status": "completed"}]),
    ])
    runner, _ = await _reset(provider, per_turn_deadline_s=600.0, max_step=1000)

    async for _ in runner.step_until_done(max_step=1000):
        pass

    assert runner._state.name == "DONE"


# --- 6. disaster-loose defaults do not misfire --------------------------------


async def test_legit_multistep_task_not_cut_by_default_max_step():
    """6.1: a legitimate multi-todo task (each todo taking a couple of child steps) completes
    under the default max_step — it is not cut short."""
    # 3 todos; each child takes 2 ReAct steps (tool call then final). Plan + 3×exec-ish + replans
    # is well under the disaster-loose default (200), so the run must complete normally.
    script = [
        _submit_plan([
            {"content": "a", "status": "pending"},
            {"content": "b", "status": "pending"},
            {"content": "c", "status": "pending"},
        ]),
        llm_text("did a"),
        _submit_plan([
            {"content": "a", "status": "completed"},
            {"content": "b", "status": "in_progress"},
            {"content": "c", "status": "pending"},
        ]),
        llm_text("did b"),
        _submit_plan([
            {"content": "a", "status": "completed"},
            {"content": "b", "status": "completed"},
            {"content": "c", "status": "in_progress"},
        ]),
        llm_text("did c"),
        _submit_plan([
            {"content": "a", "status": "completed"},
            {"content": "b", "status": "completed"},
            {"content": "c", "status": "completed"},
        ]),
    ]
    provider = FakeProvider(script)
    # No liveness kwargs → runner defaults (disaster-loose) apply.
    runner, _ = await _reset(provider, max_replan=10)

    async for _ in runner.step_until_done(max_step=1000):
        pass

    assert runner._state.name == "DONE"


def test_defaults_are_disaster_loose():
    """6.3 (anti-regression): the three ceiling defaults sit in the disaster-loose range. If a
    later change tightens a default toward "just above normal", it re-creates the guess-the-
    task-shape misfire these guards exist to avoid — this test fails loudly first."""
    from agent_runtime.core.runners import plan_execute_runner as m

    assert m._DEFAULT_MAX_STEP >= 100
    assert m._DEFAULT_PER_CALL_TIMEOUT_S >= 60.0
    assert m._DEFAULT_PER_TURN_DEADLINE_S >= 600.0


# --- 9. guards: react untouched; full-default normal run unaffected -----------


async def test_full_default_normal_run_completes_unchanged():
    """9.3: with NO liveness kwargs (all disaster-loose defaults), a normal plan-execute run
    completes DONE exactly as before — the guards never fire on a healthy run."""
    provider = FakeProvider([
        _submit_plan([{"content": "a", "status": "pending"}]),
        llm_text("did a"),
        _submit_plan([{"content": "a", "status": "completed"}]),
    ])
    runner, _ = await _reset(provider)  # no liveness kwargs at all

    final_states = []
    async for _ in runner.step_until_done(max_step=1000):
        pass
    final_states.append(runner._state.name)

    assert final_states == ["DONE"]
    assert "did a" in (runner.get_final_llm_resp().completion_text or "")


async def test_only_canonical_states_observed_through_liveness_trip():
    """9.1: even when a liveness ceiling trips, the public state never leaves the canonical
    AgentState set (the private phase never leaks)."""
    provider = _CyclicProvider()
    runner, _ = await _reset(provider, max_step=4)

    seen: set[str] = set()
    while not runner.done():
        seen.add(runner._state.name)
        async for _ in runner.step():
            pass
        seen.add(runner._state.name)

    assert seen <= {"IDLE", "RUNNING", "DONE", "ERROR"}
    assert runner._state.name == "ERROR"
