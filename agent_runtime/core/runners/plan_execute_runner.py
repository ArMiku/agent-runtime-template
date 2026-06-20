"""Plan-and-execute runner: an explicit ``PLAN → EXEC → REPLAN`` state machine.

A second control-flow paradigm sibling to the ReAct :class:`ToolLoopAgentRunner`, selected
at assembly time via ``runner_type="plan_execute"``. Where ReAct lets planning *emerge* from
the tool loop, this runner makes the plan an explicit, auditable state machine:

* **PLAN** — one structured LLM call (``tool_choice="required"`` + a lone ``submit_plan``
  tool) produces the full todo plan; parsed straight into ``list[Todo]``.
* **EXEC** — the current todo is delegated to a child :class:`ToolLoopAgentRunner` (has-a).
  The child owns an isolated run context and an isolated sub-hook chain, so its verbose tool
  traffic never enters the main context — only the todo's result summary is harvested back.
* **REPLAN** — one structured call revises the remaining plan; if anything is still pending,
  the cursor advances and EXEC resumes, otherwise the run completes (host hooks may veto).

Architectural commitments (see the change design):

* **One ``step()`` == one LLM call** (the "C2" granularity). EXEC drives the child exactly
  one step and returns — it never loops ``while not sub.done()`` — so a host can pause online
  with the same granularity as ReAct.
* **The plan's source of truth is the phase-key snapshot** (inline, single atomic ``put``);
  ``save_plan`` writes a mirror to the planning extension's key so hosts and ``include_planning``
  children read one plan via ``load_plan``. Write order is snapshot-first.
* **``phase`` is private.** The public surface stays ``AgentState{IDLE, RUNNING, DONE, ERROR}``;
  ``done()`` depends only on terminal ``AgentState``, never on ``phase``.
* **Decoupled from the planning extension's hooks/tools.** This module reuses only the shared
  pure storage/data functions (``store.py`` / ``entities.py``); the isolated sub-hook chain is
  built by the assembly layer and injected as a factory.
"""

from __future__ import annotations

import asyncio
import enum
import time
import typing as T
from collections.abc import Callable
from dataclasses import dataclass, field

from agent_runtime import logger
from agent_runtime.core.hooks import BaseAgentRunHooks
from agent_runtime.core.message import Message
from agent_runtime.core.response import AgentResponse, AgentResponseData
from agent_runtime.core.run_context import ContextWrapper, TContext
from agent_runtime.core.runners.base import AgentState, BaseAgentRunner
from agent_runtime.core.runners.tool_loop_agent_runner import ToolLoopAgentRunner
from agent_runtime.core.session_context import SessionContext
from agent_runtime.core.tool import ToolSet
from agent_runtime.core.tool_executor import BaseFunctionToolExecutor
from agent_runtime.extensions.planning.entities import Todo, TodoStatus
from agent_runtime.extensions.planning.store import PLANNING_PLUGIN_ID, save_plan
from agent_runtime.message.message_event_result import MessageChain
from agent_runtime.provider.entities import LLMResponse, ProviderRequest
from agent_runtime.provider.provider import Provider

from .plan_execute_prompts import (
    PLANNER_SYSTEM_PROMPT,
    REPLANNER_SYSTEM_PROMPT,
    build_submit_plan_tool,
    render_plan_checklist,
)

__all__ = ["PlanExecuteRunner"]

# Replan safety cap, analogous to the planning extension's ``_MAX_REMINDERS``. It bounds
# *non-productive* replans: every completed todo resets the count, so a long healthy plan is
# never cut short — only a replanner that stops making progress is forced to completion.
_DEFAULT_MAX_REPLAN = 2
_DEFAULT_PLANNER_RETRIES = 2

# Liveness ceilings (change ``add-plan-execute-liveness``). All three defaults are deliberately
# DISASTER-LOOSE — set so high that a normal task never touches them. A tight threshold (whether
# in steps or seconds) re-creates the very "guess the task's shape" failure these guards replace:
# "this kind of task shouldn't take >N minutes" is the same misfire as "shouldn't take >N steps".
# Convergence is the planner's job (its prompt produces bounded todos); these guards only cut a
# *runaway* short and report it honestly. See the change design.
_DEFAULT_MAX_STEP = 200  # ~10x a legitimate multi-step research plan (≈21 steps)
_DEFAULT_PER_CALL_TIMEOUT_S = 150.0  # a single LLM call far exceeds this only when wedged
_DEFAULT_PER_TURN_DEADLINE_S = 1500.0  # 25 min — a deep research/heavy-test task fits well inside


class _Phase(str, enum.Enum):
    """Private runner phase. Never leaks into the public ``AgentState``."""

    PLAN = "plan"
    EXEC = "exec"
    REPLAN = "replan"


class _PlanParseError(Exception):
    """Raised when a required-tool planner/replanner call yields no usable plan."""


class _PerCallTimeout(Exception):
    """Raised when a single LLM call exceeds ``per_call_timeout_s`` (in-call hang)."""


def _phase_key(session_id: str) -> str:
    """Persistence key for run progress, namespaced off the plan mirror key."""
    return f"{session_id}__phase"


@dataclass
class _Progress:
    """Serializable run progress, persisted under the phase key with the plan inlined."""

    phase: _Phase = _Phase.PLAN
    plan: list[Todo] = field(default_factory=list)
    cursor: int = 0
    summaries: dict[int, str] = field(default_factory=dict)
    replan_count: int = 0

    def to_dict(self) -> dict:
        return {
            "phase": self.phase.value,
            "plan": [t.to_dict() for t in self.plan],
            "cursor": self.cursor,
            "summaries": {str(k): v for k, v in self.summaries.items()},
            "replan_count": self.replan_count,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "_Progress":
        try:
            phase = _Phase(data.get("phase", _Phase.PLAN.value))
        except ValueError:
            phase = _Phase.PLAN
        raw_plan = data.get("plan", [])
        plan = [Todo.from_dict(item) for item in raw_plan if isinstance(item, dict)]
        summaries: dict[int, str] = {}
        raw_summaries = data.get("summaries", {})
        if isinstance(raw_summaries, dict):
            for k, v in raw_summaries.items():
                try:
                    summaries[int(k)] = str(v)
                except (TypeError, ValueError):
                    continue
        return cls(
            phase=phase,
            plan=plan,
            cursor=int(data.get("cursor", 0) or 0),
            summaries=summaries,
            replan_count=int(data.get("replan_count", 0) or 0),
        )


class PlanExecuteRunner(BaseAgentRunner[TContext]):
    """Explicit plan-and-execute control flow over the shared plan storage seam."""

    def __init__(self) -> None:
        # Fully initialized in ``reset``; defined here so attribute access pre-reset is explicit.
        self._state: AgentState = AgentState.IDLE
        self.final_llm_resp: LLMResponse | None = None
        self._progress: _Progress = _Progress()
        self._sub_runner: ToolLoopAgentRunner | None = None
        self._sub_session_id: str | None = None

    # ------------------------------------------------------------------ reset

    async def reset(
        self,
        provider: Provider,
        request: ProviderRequest,
        run_context: ContextWrapper[TContext],
        tool_executor: BaseFunctionToolExecutor[TContext],
        agent_hooks: BaseAgentRunHooks[TContext],
        *,
        plugin_store: T.Any,
        tool_set: ToolSet | None,
        sub_hook_factory: Callable[[str], BaseAgentRunHooks],
        max_replan: int = _DEFAULT_MAX_REPLAN,
        planner_retries: int = _DEFAULT_PLANNER_RETRIES,
        planner_max_tokens: int | None = None,
        planner_extra_body: dict | None = None,
        max_step: int = _DEFAULT_MAX_STEP,
        per_call_timeout_s: float | None = _DEFAULT_PER_CALL_TIMEOUT_S,
        per_turn_deadline_s: float | None = _DEFAULT_PER_TURN_DEADLINE_S,
        streaming: bool = False,
        enforce_max_turns: int = -1,
        **kwargs: T.Any,
    ) -> None:
        """Reset the runner to its initial (or recovered) state.

        Plan-execute-specific kwargs:

        * ``plugin_store`` — the KV seam for the phase snapshot and the plan mirror.
        * ``tool_set`` — the tools handed to each EXEC child runner (skills/fs/…).
        * ``sub_hook_factory`` — builds the isolated child hook chain for a given child
          ``session_id``. Built by the assembly layer, so this module stays decoupled from
          the planning/skills hook implementations.
        * ``max_replan`` / ``planner_retries`` — safety caps (see module constants).
        * ``planner_max_tokens`` — output cap forwarded as ``max_tokens`` on the planner/replanner
          structured call. The runner imposes **no** default budget: model context windows differ
          by orders of magnitude, so the budget is the host's policy, not this library's. Hosts
          driving a reasoning/thinking model set this (or ``planner_extra_body``) so the planner
          isn't truncated before emitting ``submit_plan``; ``None`` (default) omits it.
        * ``planner_extra_body`` — dict forwarded as the SDK ``extra_body`` on that call, e.g.
          ``{"thinking": False}`` (GLM/DeepSeek) or ``{"reasoning_effort": "low"}`` (OpenAI) to
          tame reasoning. Do not put an ``extra_body`` key inside it.
        * ``max_step`` / ``per_call_timeout_s`` / ``per_turn_deadline_s`` — liveness ceilings,
          checked inside ``step()`` so they trip regardless of how the host drives the runner
          (``step_until_done`` *or* a bare ``while not done(): step()`` loop). Tripping any of them
          transitions to ``ERROR`` with the already-completed summaries preserved (see
          ``_to_liveness_error``). All three default to disaster-loose values: a tight threshold
          would re-create the "guess the task's shape" misfire these guards exist to avoid — task
          convergence is the planner's job, not the guard's. ``per_call_timeout_s`` /
          ``per_turn_deadline_s`` accept ``None`` to disable that ceiling.
        """
        self.provider = provider
        self.req = request
        self.run_context = run_context
        self.tool_executor = tool_executor
        self.agent_hooks = agent_hooks
        self.plugin_store = plugin_store
        self.tool_set = tool_set
        self._sub_hook_factory = sub_hook_factory
        self._max_replan = max_replan
        self._planner_retries = planner_retries
        self._planner_max_tokens = planner_max_tokens
        self._planner_extra_body = dict(planner_extra_body) if planner_extra_body else {}
        self._max_step = max_step
        self._per_call_timeout_s = per_call_timeout_s
        self._per_turn_deadline_s = per_turn_deadline_s
        self._step_count = 0
        self._turn_start: float | None = None
        self._streaming = streaming
        self._enforce_max_turns = enforce_max_turns

        self._submit_plan_tool = build_submit_plan_tool()
        self.final_llm_resp = None
        self._sub_runner = None
        self._sub_session_id = None
        self._state = AgentState.IDLE

        # Seed the main context with the task (mirrors ToolLoopAgentRunner.reset). Done only
        # when the caller handed us an empty context, so a pre-populated/resumed context wins.
        if not self.run_context.messages:
            messages: list[Message] = []
            if request.system_prompt:
                messages.append(Message(role="system", content=request.system_prompt))
            if request.prompt is not None:
                messages.append(Message(role="user", content=request.prompt))
            self.run_context.messages = messages

        # Recover persisted progress (task 7.3): the phase-key snapshot is authoritative.
        session_id = self._session_id()
        persisted = await self.plugin_store.get(PLANNING_PLUGIN_ID, _phase_key(session_id), None)
        if isinstance(persisted, dict):
            self._progress = _Progress.from_dict(persisted)
            # Resync the mirror from the authoritative snapshot in case it lagged a crash.
            await save_plan(self.plugin_store, session_id, self._progress.plan)
            logger.info(
                "plan-execute recovered session %s at phase %s (cursor=%s, replans=%s).",
                session_id,
                self._progress.phase.value,
                self._progress.cursor,
                self._progress.replan_count,
            )
        else:
            self._progress = _Progress()

        # If recovered mid-EXEC, rebuild the child for the in-flight todo (already-completed
        # todos before the cursor are not re-run).
        if self._progress.phase == _Phase.EXEC and self._first_pending(self._progress.plan) is not None:
            self._sub_runner = await self._build_sub_runner(self._progress.cursor)

    # ----------------------------------------------------------- abstract API

    async def step(self) -> T.AsyncGenerator[AgentResponse, None]:
        """Process one step: exactly one LLM call, dispatched on the private phase."""
        if self._state == AgentState.IDLE:
            try:
                await self.agent_hooks.on_agent_begin(self.run_context)
            except Exception as e:  # noqa: BLE001 - a hook failure never breaks the run
                logger.error(f"Error in on_agent_begin hook: {e}", exc_info=True)

        if self._state in (AgentState.DONE, AgentState.ERROR):
            return  # terminal — keep done() stable, produce nothing

        # Liveness ceilings live HERE (not in step_until_done) so they trip no matter how the host
        # drives us — step_until_done or a bare ``while not done(): step()`` loop. The count/clock
        # ceilings are checked at the step boundary; the per-call timeout wraps the LLM calls below.
        if self._turn_start is None:
            self._turn_start = self._now()
        self._step_count += 1
        tripped = self._check_step_boundary_ceilings()
        if tripped is not None:
            await self._to_liveness_error(tripped)
            return

        self._transition_state(AgentState.RUNNING)

        if self._progress.phase == _Phase.PLAN:
            async for resp in self._step_plan():
                yield resp
        elif self._progress.phase == _Phase.EXEC:
            async for resp in self._step_exec():
                yield resp
        else:  # _Phase.REPLAN
            async for resp in self._step_replan():
                yield resp

    async def step_until_done(self, max_step: int) -> T.AsyncGenerator[AgentResponse, None]:
        """Drive ``step()`` until done (or the step budget is spent)."""
        step_count = 0
        while not self.done() and step_count < max_step:
            step_count += 1
            async for resp in self.step():
                yield resp

    def done(self) -> bool:
        """Terminal on DONE or ERROR — mirrors ``ToolLoopAgentRunner``.

        The spec phrases this as "``done()`` only looks at ``_state == DONE``"; read as "``done()``
        depends solely on terminal ``AgentState``, never on the private ``phase``". Matching the
        ReAct runner (DONE *or* ERROR) keeps ``step_until_done`` parity and lets a planner-failure
        ERROR terminate the run instead of spinning against the step budget.
        """
        return self._state in (AgentState.DONE, AgentState.ERROR)

    def get_final_llm_resp(self) -> LLMResponse | None:
        return self.final_llm_resp

    # ------------------------------------------------------------------ PLAN

    async def _step_plan(self) -> T.AsyncGenerator[AgentResponse, None]:
        # One structured call. PLAN/REPLAN deliberately bypass the main on_llm_request hook
        # (the planner reasons over the task, not the skill-injected message stream).
        try:
            plan = await self._call_structured(PLANNER_SYSTEM_PROMPT, self._planner_user_prompt())
        except _PerCallTimeout:
            await self._to_liveness_error("per-call timeout")
            return
        except _PlanParseError as e:
            await self._to_error(str(e))
            yield AgentResponse(
                type="err",
                data=AgentResponseData(chain=MessageChain().message(f"plan-execute planner failed: {e}")),
            )
            return

        self._progress.plan = plan
        self._progress.replan_count = 0
        cursor = self._first_pending(plan)
        if cursor is None:
            # Empty plan → nothing executable → complete immediately (task 3.3).
            self._progress.cursor = 0
            await self._persist()
            await self._complete()
            return

        self._progress.cursor = cursor
        self._progress.phase = _Phase.EXEC
        await self._persist()
        yield AgentResponse(
            type="llm_result",
            data=AgentResponseData(chain=MessageChain().message(render_plan_checklist(plan))),
        )

    # ------------------------------------------------------------------ EXEC

    async def _step_exec(self) -> T.AsyncGenerator[AgentResponse, None]:
        if self._sub_runner is None:
            self._sub_runner = await self._build_sub_runner(self._progress.cursor)

        # C2 discipline: drive the child exactly ONE step, pass through every response it
        # yields (streaming/llm_result/tool), then return. Never loop while-not-done here —
        # that would collapse many LLM calls into one top-level step and break the granularity.
        # The child's single step is wrapped in the per-call timeout: a wedged in-call hang is
        # cancelled here (the cooperative step-boundary check could never see it).
        try:
            async for resp in self._iter_with_per_call_timeout(self._sub_runner.step()):
                yield resp
        except _PerCallTimeout:
            await self._to_liveness_error("per-call timeout")
            return

        if self._sub_runner.done():
            await self._on_sub_done()

    async def _on_sub_done(self) -> None:
        """Harvest the completed todo's summary into the main context and move to REPLAN."""
        cursor = self._progress.cursor
        sub = self._sub_runner
        plan = self._progress.plan

        if cursor < len(plan):
            plan[cursor].status = TodoStatus.COMPLETED

        # Direct reuse of the child's final text as the summary — zero extra LLM call.
        summary = ""
        final = sub.get_final_llm_resp() if sub is not None else None
        if final is not None and final.completion_text:
            summary = final.completion_text
        self._progress.summaries[cursor] = summary

        todo_content = plan[cursor].content if cursor < len(plan) else f"step {cursor}"
        # The summary enters the MAIN context only (task 4.3); the child's intermediate tool
        # traffic stayed in the child's isolated context and never reaches here.
        self.run_context.messages.append(
            Message(role="user", content=f"[Step {cursor + 1} done: {todo_content}]\n{summary}"),
        )

        self._sub_runner = None
        self._sub_session_id = None
        self._progress.replan_count = 0  # forward progress resets the non-productive cap
        self._progress.phase = _Phase.REPLAN
        await self._persist()

    # ---------------------------------------------------------------- REPLAN

    async def _step_replan(self) -> T.AsyncGenerator[AgentResponse, None]:
        # Cap (task 5.3): too many consecutive non-productive replans → release completion.
        if self._progress.replan_count >= self._max_replan:
            await self._force_complete()
            return

        self._progress.replan_count += 1
        try:
            plan = await self._call_structured(REPLANNER_SYSTEM_PROMPT, self._replan_user_prompt())
        except _PerCallTimeout:
            await self._to_liveness_error("per-call timeout")
            return
        except _PlanParseError as e:
            await self._to_error(str(e))
            yield AgentResponse(
                type="err",
                data=AgentResponseData(chain=MessageChain().message(f"plan-execute replanner failed: {e}")),
            )
            return

        self._progress.plan = plan
        await self._persist()
        yield AgentResponse(
            type="llm_result",
            data=AgentResponseData(chain=MessageChain().message(render_plan_checklist(plan))),
        )

        cursor = self._first_pending(plan)
        if cursor is None:
            # All complete → attempt completion. A host hook may veto (→ another REPLAN round,
            # bounded by the cap above).
            await self._complete()
        else:
            self._progress.cursor = cursor
            self._progress.phase = _Phase.EXEC
            await self._persist()

    # ----------------------------------------------------------- completion

    async def _complete(self) -> None:
        """Attempt DONE through the ``on_before_complete`` gate; a veto returns to REPLAN."""
        final = self._build_final_response()
        try:
            admit = await self.agent_hooks.on_before_complete(self.run_context, final)
        except Exception as e:  # noqa: BLE001 - a hook raising is treated as an admit
            logger.error(f"Error in on_before_complete hook: {e}", exc_info=True)
            admit = True

        if admit is False:
            # Host wants another round — the replanner reconsiders (bounded by the replan cap).
            self._progress.phase = _Phase.REPLAN
            await self._persist()
            return

        await self._finish_done(final)

    async def _force_complete(self) -> None:
        """Release completion unconditionally (replan cap reached) — bypasses the veto gate."""
        logger.warning(
            "plan-execute reached the replan cap (%s); releasing completion.", self._max_replan
        )
        await self._finish_done(self._build_final_response())

    async def _finish_done(self, final: LLMResponse) -> None:
        """Commit the DONE transition, fire ``on_agent_done``, and persist the final snapshot."""
        self.final_llm_resp = final
        self._transition_state(AgentState.DONE)
        await self._persist()
        try:
            await self.agent_hooks.on_agent_done(self.run_context, final)
        except Exception as e:  # noqa: BLE001 - a hook failure never breaks the run
            logger.error(f"Error in on_agent_done hook: {e}", exc_info=True)

    async def _to_error(self, reason: str) -> None:
        self.final_llm_resp = LLMResponse(role="err", completion_text=f"plan-execute error: {reason}")
        self._transition_state(AgentState.ERROR)
        await self._persist()
        try:
            await self.agent_hooks.on_agent_done(self.run_context, self.final_llm_resp)
        except Exception as e:  # noqa: BLE001
            logger.error(f"Error in on_agent_done hook: {e}", exc_info=True)

    async def _to_liveness_error(self, limit_kind: str) -> None:
        """Honest terminal for a tripped liveness ceiling (max_step / per-call / per-turn).

        Unlike a planner-failure ``_to_error``, this is not a defect: the run was cut short by a
        guard, but the todos completed so far are real partial results. So the ``completion_text``
        leads with a recognizable stop reason (which limit, which phase, how many todos done) and
        then carries the already-completed summaries in host-displayable form (reusing
        ``_build_final_response``), so a host can surface "stopped, but here's what got done"
        rather than discarding a real partial result. ``role="err"`` still marks it as cut short —
        distinct from ``_force_complete``'s DONE ("replan cap reached but the plan basically met
        its goal").
        """
        completed = sum(1 for t in self._progress.plan if t.status is TodoStatus.COMPLETED)
        reason = (
            f"plan-execute stopped: {limit_kind} reached during {self._progress.phase.value} "
            f"phase ({completed} todo(s) completed). Partial results below."
        )
        partial = self._build_final_response().completion_text or ""
        logger.warning("plan-execute liveness ceiling tripped: %s.", limit_kind)
        self.final_llm_resp = LLMResponse(role="err", completion_text=f"{reason}\n\n{partial}")
        self._transition_state(AgentState.ERROR)
        await self._persist()
        try:
            await self.agent_hooks.on_agent_done(self.run_context, self.final_llm_resp)
        except Exception as e:  # noqa: BLE001 - a hook failure never breaks the run
            logger.error(f"Error in on_agent_done hook: {e}", exc_info=True)

    def _build_final_response(self) -> LLMResponse:
        """Assemble the final response from harvested summaries (no extra LLM call)."""
        plan = self._progress.plan
        if not plan:
            return LLMResponse(role="assistant", completion_text="Plan executed (no steps).")
        lines = []
        for i, todo in enumerate(plan):
            summary = self._progress.summaries.get(i)
            if summary:
                lines.append(f"- {todo.content}: {summary}")
            else:
                lines.append(f"- {todo.content} ({todo.status.value})")
        return LLMResponse(role="assistant", completion_text="Plan executed.\n" + "\n".join(lines))

    # ----------------------------------------------------------- persistence

    async def _persist(self) -> None:
        """Write the phase snapshot (authoritative) first, then the save_plan mirror."""
        session_id = self._session_id()
        await self.plugin_store.put(PLANNING_PLUGIN_ID, _phase_key(session_id), self._progress.to_dict())
        await save_plan(self.plugin_store, session_id, self._progress.plan)

    # ----------------------------------------------------------- child runner

    async def _build_sub_runner(self, cursor: int) -> ToolLoopAgentRunner:
        """Construct an isolated child ReAct runner for the todo at ``cursor``."""
        sub_session_id = f"{self._session_id()}__todo_{cursor}"
        self._sub_session_id = sub_session_id
        todo = self._progress.plan[cursor]

        sub_run_context: ContextWrapper[SessionContext] = ContextWrapper(
            context=SessionContext(session_id=sub_session_id),
            messages=[],
        )
        sub_hooks = self._sub_hook_factory(sub_session_id)
        sub_request = ProviderRequest(
            prompt=(
                "Execute exactly this step of the overall task, then report a concise result.\n\n"
                f"Overall task: {self._task_text()}\n\n"
                f"Step to complete: {todo.content}"
            ),
            system_prompt=(
                "You are the executor in a plan-and-execute pipeline. Carry out the assigned "
                "step with the available tools, then give a brief result summary."
            ),
            func_tool=self.tool_set,
            session_id=sub_session_id,
        )
        sub = ToolLoopAgentRunner()
        await sub.reset(
            provider=self.provider,
            request=sub_request,
            run_context=sub_run_context,
            tool_executor=self.tool_executor,
            agent_hooks=sub_hooks,
            streaming=self._streaming,
            enforce_max_turns=self._enforce_max_turns,
        )
        return sub

    # ----------------------------------------------------------- planner I/O

    async def _call_structured(self, system_prompt: str, user_prompt: str) -> list[Todo]:
        """One ``tool_choice="required"`` call → parse ``submit_plan`` → ``list[Todo]``.

        Retries on parse failure up to ``_planner_retries`` extra attempts; raises
        :class:`_PlanParseError` on final failure so the caller transitions to ERROR.
        """
        tool_set = ToolSet([self._submit_plan_tool])
        planner_kwargs: dict[str, T.Any] = {}
        if self._planner_max_tokens is not None:
            planner_kwargs["max_tokens"] = self._planner_max_tokens
        if self._planner_extra_body:
            # Forwarded as the SDK extra_body (e.g. {"thinking": False} / {"reasoning_effort": "low"}).
            planner_kwargs["extra_body"] = dict(self._planner_extra_body)
        last_err = "unknown"
        for _ in range(self._planner_retries + 1):
            resp = await self._with_per_call_timeout(
                self.provider.text_chat(
                    prompt=user_prompt,
                    system_prompt=system_prompt,
                    func_tool=tool_set,
                    tool_choice="required",
                    model=self.req.model,
                    session_id=self.req.session_id,
                    **planner_kwargs,
                )
            )
            if resp.role == "err":
                last_err = resp.completion_text or "provider error"
                continue
            todos = self._extract_plan(resp)
            if todos is not None:
                return todos
            # The model returned content but no submit_plan call — on a reasoning model this
            # is almost always truncation: the chain-of-thought ate the output budget before
            # the tool call was emitted. Point the user at the two levers.
            if resp.reasoning_content or resp.completion_text:
                produced = "reasoning" if resp.reasoning_content else "text"
                last_err = (
                    f"submit_plan tool call missing or malformed, but the model produced {produced} "
                    "output — likely truncated by reasoning/thinking before the tool call "
                    "(finish_reason=length). Fix: raise planner_max_tokens, or disable thinking "
                    "via planner_extra_body={'reasoning_effort':'low'} (OpenAI) / "
                    "{'thinking': False} (GLM/DeepSeek)."
                )
            else:
                last_err = "submit_plan tool call missing or malformed"
        raise _PlanParseError(last_err)

    @staticmethod
    def _extract_plan(resp: LLMResponse) -> list[Todo] | None:
        """Return the parsed plan, or None if no usable ``submit_plan`` call was emitted.

        An empty ``todos`` list is a *valid* plan (nothing to do) — only a missing call or a
        non-list ``todos`` field is treated as a parse failure (→ retry).
        """
        for name, arg in zip(resp.tools_call_name or [], resp.tools_call_args or []):
            if name == "submit_plan" and isinstance(arg, dict):
                raw = arg.get("todos")
                if isinstance(raw, list):
                    return [Todo.from_dict(item) for item in raw if isinstance(item, dict)]
        return None

    def _planner_user_prompt(self) -> str:
        return f"Task:\n{self._task_text()}\n\nProduce the plan."

    def _replan_user_prompt(self) -> str:
        summaries = []
        for i in sorted(self._progress.summaries):
            content = (
                self._progress.plan[i].content
                if i < len(self._progress.plan)
                else f"step {i + 1}"
            )
            summaries.append(f"- Step {i + 1} ({content}): {self._progress.summaries[i]}")
        summaries_text = "\n".join(summaries) if summaries else "(none yet)"
        return (
            f"Task:\n{self._task_text()}\n\n"
            f"Current plan:\n{render_plan_checklist(self._progress.plan)}\n\n"
            f"Completed-step summaries:\n{summaries_text}\n\n"
            "Revise the remaining plan."
        )

    # ----------------------------------------------------------- helpers

    def _session_id(self) -> str:
        return getattr(self.run_context.context, "session_id", "") or ""

    def _now(self) -> float:
        """Monotonic clock for the per-turn deadline (run-duration semantics, immune to wall-clock
        jumps). A method so tests can patch it deterministically."""
        return time.monotonic()

    async def _with_per_call_timeout(self, coro: T.Awaitable[T.Any]) -> T.Any:
        """Await a single LLM-call coroutine under the per-call timeout. Translates a timeout into
        ``_PerCallTimeout`` (cancelling the wedged call); ``None`` disables the ceiling."""
        if self._per_call_timeout_s is None:
            return await coro
        try:
            return await asyncio.wait_for(coro, timeout=self._per_call_timeout_s)
        except (TimeoutError, asyncio.TimeoutError) as e:  # noqa: UP041 - 3.10 compat
            raise _PerCallTimeout from e

    async def _iter_with_per_call_timeout(
        self, gen: T.AsyncGenerator[AgentResponse, None]
    ) -> T.AsyncGenerator[AgentResponse, None]:
        """Drive an async-generator step (the child's ``step()``) under the per-call timeout,
        applied per chunk so streaming is preserved: each ``__anext__`` must arrive within the
        timeout, else the wedged call is cancelled and ``_PerCallTimeout`` is raised."""
        if self._per_call_timeout_s is None:
            async for resp in gen:
                yield resp
            return
        while True:
            try:
                resp = await asyncio.wait_for(gen.__anext__(), timeout=self._per_call_timeout_s)
            except StopAsyncIteration:
                return
            except (TimeoutError, asyncio.TimeoutError) as e:  # noqa: UP041 - 3.10 compat
                await gen.aclose()
                raise _PerCallTimeout from e
            yield resp

    def _check_step_boundary_ceilings(self) -> str | None:
        """Return the name of a tripped step-boundary ceiling (max_step / per-turn deadline), or
        None. The per-call timeout is NOT here — it wraps the in-step LLM calls, since a call
        wedged mid-step never returns to this boundary for a cooperative check to fire."""
        if self._step_count > self._max_step:
            return "max_step"
        if self._per_turn_deadline_s is not None and self._turn_start is not None:
            if self._now() - self._turn_start > self._per_turn_deadline_s:
                return "per-turn deadline"
        return None

    def _task_text(self) -> str:
        return self.req.prompt or ""

    @staticmethod
    def _first_pending(plan: list[Todo]) -> int | None:
        """Index of the first non-completed todo, or None when all are completed."""
        for i, todo in enumerate(plan):
            if todo.status is not TodoStatus.COMPLETED:
                return i
        return None
