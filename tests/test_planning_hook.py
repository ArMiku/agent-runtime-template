"""``PlanningHook``: live-plan injection (idempotent, compaction-proof, skills-coexisting)
and the completion veto with its per-session reminder cap (tasks 5.3, 5.4, 7.2, 7.3)."""

from __future__ import annotations

from agent_runtime.core.message import Message
from agent_runtime.core.run_context import ContextWrapper
from agent_runtime.core.session_context import SessionContext
from agent_runtime.extensions.planning.entities import Todo, TodoStatus
from agent_runtime.extensions.planning.store import save_plan
from agent_runtime.extensions.planning.todo_hook import PlanningHook
from agent_runtime.extensions.plugins.store import InMemoryPluginStore


def _ctx(system: str | None = None, session_id: str = "s1") -> ContextWrapper[SessionContext]:
    messages = []
    if system is not None:
        messages.append(Message(role="system", content=system))
    return ContextWrapper(context=SessionContext(session_id=session_id), messages=messages)


# —— injection (on_llm_request) ——


async def test_plan_injected_when_present():
    store = InMemoryPluginStore()
    await save_plan(store, "s1", [Todo(content="do the thing", status=TodoStatus.IN_PROGRESS)])
    ctx = _ctx(system="You are helpful.")

    await PlanningHook(store).on_llm_request(ctx)

    content = ctx.messages[0].content
    assert ctx.messages[0].role == "system"
    assert content.startswith("You are helpful.")
    assert "do the thing" in content
    assert content.count(PlanningHook._OPEN) == 1
    assert content.count(PlanningHook._CLOSE) == 1


async def test_injection_is_idempotent_across_steps():
    store = InMemoryPluginStore()
    await save_plan(store, "s1", [Todo(content="step")])
    hook = PlanningHook(store)
    ctx = _ctx(system="base")

    await hook.on_llm_request(ctx)
    first = ctx.messages[0].content
    await hook.on_llm_request(ctx)
    await hook.on_llm_request(ctx)
    second = ctx.messages[0].content

    assert second == first
    assert second.count(PlanningHook._OPEN) == 1


async def test_reinjected_from_state_after_compaction():
    """7.2: even if the write_todos tool message was compacted away, the plan is re-read
    from independent state and re-injected. We simulate compaction by giving the hook a
    fresh context with no planning message in history — only the store has the plan."""
    store = InMemoryPluginStore()
    await save_plan(store, "s1", [Todo(content="survives compaction")])

    # A brand-new context (history wiped by compaction) — nothing about the plan in messages.
    ctx = _ctx(system="base")
    await PlanningHook(store).on_llm_request(ctx)

    assert "survives compaction" in ctx.messages[0].content


async def test_coexists_with_skills_sentinel():
    """5.3: planning replaces only its own sentinel block; a skills block is untouched."""
    store = InMemoryPluginStore()
    await save_plan(store, "s1", [Todo(content="plan item")])
    skills_block = "<!-- skills-inventory -->\n## Skills\nstuff\n<!-- /skills-inventory -->"
    ctx = _ctx(system=f"base\n\n{skills_block}")

    await PlanningHook(store).on_llm_request(ctx)

    content = ctx.messages[0].content
    # Skills block survives verbatim; plan block added separately.
    assert skills_block in content
    assert "plan item" in content
    assert content.count(PlanningHook._OPEN) == 1


async def test_no_plan_removes_segment():
    store = InMemoryPluginStore()
    await save_plan(store, "s1", [Todo(content="x")])
    hook = PlanningHook(store)
    ctx = _ctx(system="base")
    await hook.on_llm_request(ctx)
    assert PlanningHook._OPEN in ctx.messages[0].content

    # Plan cleared → the segment must be removed, not left stale.
    await save_plan(store, "s1", [])
    await hook.on_llm_request(ctx)
    assert PlanningHook._OPEN not in ctx.messages[0].content


async def test_injection_failure_isolated():
    class _BoomStore:
        async def get(self, *a, **k):
            raise RuntimeError("boom")

    ctx = _ctx(system="base")
    # Must not raise — a planning failure never breaks the run.
    await PlanningHook(_BoomStore()).on_llm_request(ctx)  # type: ignore[arg-type]
    assert ctx.messages[0].content == "base"


# —— completion veto (on_before_complete) ——


async def test_veto_when_unfinished():
    store = InMemoryPluginStore()
    await save_plan(store, "s1", [Todo(content="open", status=TodoStatus.PENDING)])
    ctx = _ctx(session_id="s1")

    admit = await PlanningHook(store).on_before_complete(ctx, None)
    assert admit is False
    # A reminder was appended so the next round has guidance.
    assert any(m.role == "user" for m in ctx.messages)


async def test_admit_when_all_completed():
    store = InMemoryPluginStore()
    await save_plan(store, "s1", [Todo(content="done", status=TodoStatus.COMPLETED)])
    ctx = _ctx(session_id="s1")

    assert await PlanningHook(store).on_before_complete(ctx, None) is True


async def test_admit_when_no_plan():
    store = InMemoryPluginStore()
    ctx = _ctx(session_id="s1")
    assert await PlanningHook(store).on_before_complete(ctx, None) is True


async def test_reminder_cap_admits_after_limit():
    """5.4: once the per-session cap is hit, completion is admitted despite open todos."""
    store = InMemoryPluginStore()
    await save_plan(store, "s1", [Todo(content="open", status=TodoStatus.PENDING)])
    hook = PlanningHook(store, max_reminders=2)
    ctx = _ctx(session_id="s1")

    assert await hook.on_before_complete(ctx, None) is False  # reminder 1
    assert await hook.on_before_complete(ctx, None) is False  # reminder 2
    assert await hook.on_before_complete(ctx, None) is True  # cap reached → admit


async def test_reminder_cap_is_per_session():
    store = InMemoryPluginStore()
    await save_plan(store, "s1", [Todo(content="open")])
    await save_plan(store, "s2", [Todo(content="open")])
    hook = PlanningHook(store, max_reminders=1)

    assert await hook.on_before_complete(_ctx(session_id="s1"), None) is False
    assert await hook.on_before_complete(_ctx(session_id="s1"), None) is True  # s1 capped
    # s2 has its own independent count — still vetoes the first time.
    assert await hook.on_before_complete(_ctx(session_id="s2"), None) is False


# —— manual replanning through the shared write channel (task 7.3) ——


async def test_manual_edit_then_injection_reflects_change():
    """7.3: a host edits the plan via write_plan (same channel as the LLM), then the next
    injection shows the edited plan."""
    store = InMemoryPluginStore()
    hook = PlanningHook(store)
    await hook.write_plan("s1", [Todo(content="original")])

    # Host reads, edits, writes back through the same channel.
    current = await hook.read_plan("s1")
    current.append(Todo(content="human-added", status=TodoStatus.IN_PROGRESS))
    await hook.write_plan("s1", current)

    ctx = _ctx(system="base", session_id="s1")
    await hook.on_llm_request(ctx)
    content = ctx.messages[0].content
    assert "original" in content and "human-added" in content
