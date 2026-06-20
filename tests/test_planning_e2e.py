"""End-to-end: a ReAct run with planning enabled is held back from finishing while a todo
is unfinished, then completes once the plan is done (integration of tool + hook + veto)."""

from __future__ import annotations

from agent_runtime.core import FunctionToolExecutor
from agent_runtime.core.run_context import ContextWrapper
from agent_runtime.core.runners.tool_loop_agent_runner import ToolLoopAgentRunner
from agent_runtime.core.session_context import SessionContext
from agent_runtime.extensions.planning.store import load_plan
from agent_runtime.extensions.plugins.store import InMemoryPluginStore
from agent_runtime.local_runtime import build_local_agent_basics
from agent_runtime.provider.entities import ProviderRequest

from .fakes import FakeProvider, llm_text, llm_tool_call


async def test_planning_run_vetoes_then_completes():
    """The model: (1) writes a plan with one in_progress item, (2) prematurely tries to
    finish → vetoed, (3) writes an all-completed plan, (4) finishes → admitted."""
    store = InMemoryPluginStore()
    basics = build_local_agent_basics(include_fs=False, include_planning=True, plugin_store=store)

    script = [
        llm_tool_call("write_todos", {"todos": [{"content": "task", "status": "in_progress"}]}),
        llm_text("all done!"),  # premature — one todo still in_progress → vetoed
        llm_tool_call("write_todos", {"todos": [{"content": "task", "status": "completed"}]}),
        llm_text("actually done now"),  # admitted — plan complete
    ]
    provider = FakeProvider(script)
    request = ProviderRequest(prompt="go", system_prompt="", func_tool=basics.tools)
    run_context = ContextWrapper(context=SessionContext(session_id="e2e"), messages=[])
    runner = ToolLoopAgentRunner()
    await runner.reset(
        provider=provider,
        request=request,
        run_context=run_context,
        tool_executor=FunctionToolExecutor(provider),
        agent_hooks=basics.hooks,
    )

    async for _ in runner.step_until_done(max_step=20):
        pass

    final = runner.get_final_llm_resp()
    assert final is not None and final.completion_text == "actually done now"
    # Final plan state is the all-completed one.
    plan = await load_plan(store, "e2e")
    assert [t.status.value for t in plan] == ["completed"]
