"""Composition root: assemble skills + fs + plugins into a runnable local agent.

This package-top module is the *one* place allowed to depend outward on every extension
plus the runner/provider entities (the extensions never import each other). It delivers
the upper two tiers of the assembly stack; the fine-grained Tier-1
builders stay untouched and importable, so hand-wiring remains a first-class path.

* **Tier 2** — :func:`build_local_agent_basics` returns a :class:`LocalAgentBasics`
  bundle (``skill_manager`` + ``tools`` + ``hooks``) with no provider/runner. For callers
  that want to own the runner lifecycle.
* **Tier 3** — :func:`build_local_agent` wraps Tier 2 plus
  ``ProviderRequest``/``ContextWrapper``/``FunctionToolExecutor``/``ToolLoopAgentRunner``
  and ``await runner.reset(...)`` into a ready-to-run :class:`LocalAgent`. The caller
  supplies only the ``provider`` (host credentials) and calls ``await agent.run()``.

The factories are thin wiring over the existing builders — zero logic is duplicated. The
fs and plugins extensions are imported lazily, so a skills-only caller never pulls them in.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from agent_runtime.core import FunctionToolExecutor
from agent_runtime.core.hooks import BaseAgentRunHooks
from agent_runtime.core.hooks_chain import ChainedAgentRunHooks
from agent_runtime.core.response import AgentResponse
from agent_runtime.core.run_context import ContextWrapper
from agent_runtime.core.runners.base import BaseAgentRunner
from agent_runtime.core.runners.tool_loop_agent_runner import ToolLoopAgentRunner
from agent_runtime.core.session_context import SessionContext
from agent_runtime.core.tool import ToolSet
from agent_runtime.extensions.skills import (
    SkillManager,
    SkillsPromptHook,
    build_skill_tool,
)
from agent_runtime.provider.entities import LLMResponse, ProviderRequest
from agent_runtime.provider.provider import Provider

if TYPE_CHECKING:
    from agent_runtime.extensions.planning import PlanningHook
    from agent_runtime.extensions.plugins import PluginContribution
    from agent_runtime.extensions.plugins.store import PluginStore

__all__ = [
    "LocalAgentBasics",
    "build_local_agent_basics",
    "LocalAgent",
    "build_local_agent",
]


# --- Tier 2: bundle (tools + hooks, no runner) -------------------------------


@dataclass
class LocalAgentBasics:
    """The skills/fs/plugins bundle: discovery manager, tools, and a single hook.

    Deliberately free of ``provider``/``runner``/``request`` — this is the layer for
    callers that want their own runner lifecycle. ``tools`` is mutable (``add_tool`` /
    ``merge``) and ``hooks`` is replaceable, so a one-shot bundle can still be tweaked.
    ``planning_hook`` is set only when ``include_planning`` is enabled; a host uses it to
    read/edit the live plan (manual replanning) through the same write channel the LLM uses.
    """

    skill_manager: SkillManager
    tools: ToolSet
    hooks: BaseAgentRunHooks
    planning_hook: "PlanningHook | None" = None


def build_local_agent_basics(
    *,
    skills_root: str | Path | None = None,
    contributions: Sequence["PluginContribution"] = (),
    include_fs: bool = True,
    include_planning: bool = False,
    plugin_store: "PluginStore | None" = None,
    extra_allowed_roots: Sequence[str | Path] = (),
    runner_type: str = "react",
) -> LocalAgentBasics:
    """Wire skills (+ optional fs + optional plugin contributions) into one bundle.

    Only connects existing builders — no skills/fs/plugins logic is reimplemented:

    1. aggregate each ``contribution.skill_dirs`` as ``extra_skill_dirs`` for the
       ``SkillManager`` (plugin-bundled skills auto-discovered);
    2. ``build_skill_tool(mgr)`` + each ``contribution.tools`` → ``ToolSet``;
    3. when ``include_fs`` (default), merge ``build_fs_tools(allowed_roots=[skills_dir,
       *plugin_skill_dirs, *extra_allowed_roots])``;
    4. ``SkillsPromptHook(mgr)`` plus, when there are plugin ``hook_methods``,
       ``CompositeAgentRunHooks(contributions)``, plus, when ``include_planning``, the
       planning ``write_todos`` tool + ``PlanningHook`` — all chained into one via
       ``ChainedAgentRunHooks`` (degenerating to the lone hook when there is only one).

    When ``include_planning`` is set, ``plugin_store`` (defaulting to a fresh
    ``InMemoryPluginStore``) backs the plan state; a host injects a persistent store for
    cross-process recovery.

    ``runner_type`` selects the control-flow paradigm (``"react"`` or ``"plan_execute"``) but
    does NOT change this bundle: the tools/hooks are runner-agnostic by design (the bundle
    carries no runner). It is threaded through for API symmetry with :func:`build_local_agent`,
    where it actually selects the runner.

    The fs, plugins, and planning extensions are imported lazily so a skills-only caller
    never pulls them in. The bundle carries no provider/runner — see
    :func:`build_local_agent`.
    """
    contributions = list(contributions)

    # 1. Aggregate plugin-bundled skill directories for discovery.
    plugin_skill_dirs: list[Path] = []
    for contribution in contributions:
        plugin_skill_dirs.extend(contribution.skill_dirs)

    skill_manager = SkillManager(
        skills_root=str(skills_root) if skills_root is not None else None,
        extra_skill_dirs=plugin_skill_dirs or None,
    )

    # 2. Skill tool + plugin tools.
    tools = ToolSet([build_skill_tool(skill_manager)])
    for contribution in contributions:
        for tool in contribution.tools:
            tools.add_tool(tool)

    # 3. Optional read-only fs tools, fenced to the skills + plugin-skill + extra roots.
    if include_fs:
        from agent_runtime.extensions.fs import build_fs_tools

        allowed_roots: list[str | os.PathLike[str]] = [skill_manager.skills_root]
        allowed_roots.extend(plugin_skill_dirs)
        allowed_roots.extend(extra_allowed_roots)
        tools.merge(ToolSet(build_fs_tools(allowed_roots=allowed_roots)))

    # 4. Assemble hooks: skills inventory + (optional) plugin hooks + (optional) planning.
    #    Collect every live hook, then chain — a lone hook degenerates out of the chain.
    chained: list[BaseAgentRunHooks] = [SkillsPromptHook(skill_manager)]

    has_plugin_hooks = any(contribution.hook_methods for contribution in contributions)
    if has_plugin_hooks:
        from agent_runtime.extensions.plugins import CompositeAgentRunHooks

        chained.append(CompositeAgentRunHooks(contributions))

    planning_hook: "PlanningHook | None" = None
    if include_planning:
        from agent_runtime.extensions.planning import build_planning_extension
        from agent_runtime.extensions.plugins.store import InMemoryPluginStore

        store = plugin_store if plugin_store is not None else InMemoryPluginStore()
        write_todos_tool, planning_hook = build_planning_extension(store)
        tools.add_tool(write_todos_tool)
        chained.append(planning_hook)

    hooks: BaseAgentRunHooks = chained[0] if len(chained) == 1 else ChainedAgentRunHooks(*chained)

    return LocalAgentBasics(
        skill_manager=skill_manager,
        tools=tools,
        hooks=hooks,
        planning_hook=planning_hook,
    )


# --- Tier 3: ready-to-run agent (wraps the runner lifecycle) -----------------


def _build_plan_execute_sub_hook_factory(
    skill_manager: SkillManager,
    include_planning: bool,
    plugin_store: "PluginStore",
) -> Any:
    """Build the isolated child-hook factory injected into ``PlanExecuteRunner``.

    Each EXEC child runner gets its own hook chain bound to its child ``session_id``:
    ``SkillsPromptHook`` (so the executor can use skills), plus — only when
    ``include_planning`` is set — a ``PlanningHook`` over the shared store, which reads/writes
    the child's emerging plan under the child session (isolated from the top-level plan). The
    main runner's hook chain never touches a child, so a main ``PlanningHook`` cannot veto a
    child's completion on the top-level plan.

    Lived here (the composition root) rather than in the runner so the runner stays decoupled
    from the planning/skills hook implementations.
    """

    def factory(sub_session_id: str) -> BaseAgentRunHooks:
        chained: list[BaseAgentRunHooks] = [SkillsPromptHook(skill_manager)]
        if include_planning:
            from agent_runtime.extensions.planning import PlanningHook

            chained.append(PlanningHook(plugin_store))
        return chained[0] if len(chained) == 1 else ChainedAgentRunHooks(*chained)

    return factory



@dataclass
class LocalAgent:
    """A one-shot-ready agent: the Tier-2 bundle plus a reset runner.

    ``build_local_agent`` constructs every piece except the ``provider`` (host
    credentials) and calls ``runner.reset`` for the caller. The methods below delegate
    to the runner. The ``FunctionToolExecutor`` is built by the factory, so its cleanup
    belongs here: ``runner.reset`` stores the executor but never closes it at run end.
    """

    basics: LocalAgentBasics
    runner: BaseAgentRunner
    provider: Provider
    request: ProviderRequest
    run_context: ContextWrapper[SessionContext]

    def step(self) -> AsyncIterator[AgentResponse]:
        """Process one step (delegates to the runner)."""
        return self.runner.step()

    def step_until_done(self, max_step: int) -> AsyncIterator[AgentResponse]:
        """Yield responses until the agent is done (delegates to the runner)."""
        return self.runner.step_until_done(max_step=max_step)

    async def run(self, max_step: int = 20) -> LLMResponse | None:
        """Drain ``step_until_done`` then return ``get_final_llm_resp()``."""
        async for _ in self.runner.step_until_done(max_step=max_step):
            pass
        return self.runner.get_final_llm_resp()

    def get_final_llm_resp(self) -> LLMResponse | None:
        """Return the run's final LLM response (delegates to the runner)."""
        return self.runner.get_final_llm_resp()

    def done(self) -> bool:
        """Whether the agent has finished (delegates to the runner)."""
        return self.runner.done()

    async def aclose(self) -> None:
        """Close the self-built tool executor; no-op when it lacks ``aclose``.

        The ``provider`` is supplied (and possibly reused) by the caller, so it is not
        closed here.
        """
        executor = getattr(self.runner, "tool_executor", None)
        close = getattr(executor, "aclose", None)
        if close is not None:
            await close()


async def build_local_agent(
    provider: Provider,
    *,
    prompt: str | None = None,
    system_prompt: str = "",
    session_id: str | None = None,
    skills_root: str | Path | None = None,
    contributions: Sequence["PluginContribution"] = (),
    include_fs: bool = True,
    include_planning: bool = False,
    plugin_store: "PluginStore | None" = None,
    extra_allowed_roots: Sequence[str | Path] = (),
    runner_type: str = "react",
    max_turns: int = -1,
    streaming: bool = False,
    **runner_kwargs: Any,
) -> LocalAgent:
    """Assemble a ready-to-run :class:`LocalAgent` (async — ``runner.reset`` is a coro).

    Reuses :func:`build_local_agent_basics` (zero duplication), then builds the
    ``ProviderRequest`` / ``ContextWrapper`` / ``FunctionToolExecutor`` / runner and awaits
    ``runner.reset``. ``provider`` is the only piece the caller must supply (host
    credentials/model); the factory never constructs one. ``max_turns`` maps to
    ``enforce_max_turns`` and ``**runner_kwargs`` pass straight through to ``runner.reset``
    for advanced tuning (``fallback_providers``, ``llm_compress_*``, ``truncate_turns``,
    ``custom_compressor``, ...).

    ``runner_type`` selects the control-flow paradigm and is orthogonal to
    ``include_planning``:

    * ``"react"`` (default) — the ReAct :class:`ToolLoopAgentRunner`.
    * ``"plan_execute"`` — :class:`~agent_runtime.core.runners.plan_execute_runner.PlanExecuteRunner`,
      an explicit ``PLAN → EXEC → REPLAN`` state machine. Each EXEC step delegates to an
      isolated child ReAct runner; set ``include_planning=True`` to let those children also
      exhibit emergent per-step replanning (route 1 + route 2).

    Set ``include_planning=True`` to add the ``write_todos`` tool + ``PlanningHook``
    (plan-and-execute over the existing ReAct loop). ``plugin_store`` backs the plan state
    (and, for ``plan_execute``, the phase snapshot); inject a persistent implementation for
    cross-process recovery (the default ``InMemoryPluginStore`` is process-local only).
    """
    if runner_type not in ("react", "plan_execute"):
        raise ValueError(
            f"Unknown runner_type {runner_type!r}; expected 'react' or 'plan_execute'."
        )

    # plan_execute needs a PluginStore for its phase snapshot even without include_planning;
    # create one up front so it is shared with planning when both are on (one store, one plan).
    if (include_planning or runner_type == "plan_execute") and plugin_store is None:
        from agent_runtime.extensions.plugins.store import InMemoryPluginStore

        plugin_store = InMemoryPluginStore()

    basics = build_local_agent_basics(
        skills_root=skills_root,
        contributions=contributions,
        include_fs=include_fs,
        include_planning=include_planning,
        plugin_store=plugin_store,
        extra_allowed_roots=extra_allowed_roots,
        runner_type=runner_type,
    )

    request = ProviderRequest(
        prompt=prompt,
        system_prompt=system_prompt,
        func_tool=basics.tools,
        session_id=session_id or "",
    )
    run_context: ContextWrapper[SessionContext] = ContextWrapper(
        context=SessionContext(session_id=session_id or "default"),
        messages=[],
    )
    tool_executor = FunctionToolExecutor(provider)

    if runner_type == "plan_execute":
        from agent_runtime.core.runners.plan_execute_runner import PlanExecuteRunner

        runner: BaseAgentRunner = PlanExecuteRunner()
        sub_hook_factory = _build_plan_execute_sub_hook_factory(
            skill_manager=basics.skill_manager,
            include_planning=include_planning,
            plugin_store=plugin_store,  # type: ignore[arg-type]
        )
        await runner.reset(
            provider=provider,
            request=request,
            run_context=run_context,
            tool_executor=tool_executor,
            agent_hooks=basics.hooks,
            plugin_store=plugin_store,  # type: ignore[arg-type]
            tool_set=basics.tools,
            sub_hook_factory=sub_hook_factory,
            streaming=streaming,
            enforce_max_turns=max_turns,
            **runner_kwargs,
        )
    else:
        runner = ToolLoopAgentRunner()
        await runner.reset(
            provider=provider,
            request=request,
            run_context=run_context,
            tool_executor=tool_executor,
            agent_hooks=basics.hooks,
            streaming=streaming,
            enforce_max_turns=max_turns,
            **runner_kwargs,
        )

    return LocalAgent(
        basics=basics,
        runner=runner,
        provider=provider,
        request=request,
        run_context=run_context,
    )
