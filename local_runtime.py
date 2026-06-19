"""Composition root: assemble skills + fs + plugins into a runnable local agent.

This package-top module is the *one* place allowed to depend outward on every extension
plus the runner/provider entities (the extensions never import each other). It delivers
the upper two tiers of the assembly stack (design.md §4–5); the fine-grained Tier-1
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
    from agent_runtime.extensions.plugins import PluginContribution

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
    """

    skill_manager: SkillManager
    tools: ToolSet
    hooks: BaseAgentRunHooks


def build_local_agent_basics(
    *,
    skills_root: str | Path | None = None,
    contributions: Sequence["PluginContribution"] = (),
    include_fs: bool = True,
    extra_allowed_roots: Sequence[str | Path] = (),
) -> LocalAgentBasics:
    """Wire skills (+ optional fs + optional plugin contributions) into one bundle.

    Only connects existing builders — no skills/fs/plugins logic is reimplemented:

    1. aggregate each ``contribution.skill_dirs`` as ``extra_skill_dirs`` for the
       ``SkillManager`` (plugin-bundled skills auto-discovered);
    2. ``build_skill_tool(mgr)`` + each ``contribution.tools`` → ``ToolSet``;
    3. when ``include_fs`` (default), merge ``build_fs_tools(allowed_roots=[skills_dir,
       *plugin_skill_dirs, *extra_allowed_roots])``;
    4. ``SkillsPromptHook(mgr)`` plus, when there are plugin ``hook_methods``,
       ``CompositeAgentRunHooks(contributions)`` — chained into one via
       ``ChainedAgentRunHooks`` (degenerating to the lone hook when there is only one).

    The fs and plugins extensions are imported lazily so a skills-only caller never
    pulls them in. The bundle carries no provider/runner — see :func:`build_local_agent`.
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

    # 4. Skills inventory hook, plus plugin hooks when any contribution declares them.
    hooks: BaseAgentRunHooks = SkillsPromptHook(skill_manager)
    has_plugin_hooks = any(contribution.hook_methods for contribution in contributions)
    if has_plugin_hooks:
        from agent_runtime.extensions.plugins import CompositeAgentRunHooks

        hooks = ChainedAgentRunHooks(
            SkillsPromptHook(skill_manager),
            CompositeAgentRunHooks(contributions),
        )

    return LocalAgentBasics(skill_manager=skill_manager, tools=tools, hooks=hooks)


# --- Tier 3: ready-to-run agent (wraps the runner lifecycle) -----------------


@dataclass
class LocalAgent:
    """A one-shot-ready agent: the Tier-2 bundle plus a reset runner.

    ``build_local_agent`` constructs every piece except the ``provider`` (host
    credentials) and calls ``runner.reset`` for the caller. The methods below delegate
    to the runner. The ``FunctionToolExecutor`` is built by the factory, so its cleanup
    belongs here: ``runner.reset`` stores the executor but never closes it at run end.
    """

    basics: LocalAgentBasics
    runner: "ToolLoopAgentRunner"
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
    extra_allowed_roots: Sequence[str | Path] = (),
    max_turns: int = -1,
    streaming: bool = False,
    **runner_kwargs: Any,
) -> LocalAgent:
    """Assemble a ready-to-run :class:`LocalAgent` (async — ``runner.reset`` is a coro).

    Reuses :func:`build_local_agent_basics` (zero duplication), then builds the
    ``ProviderRequest`` / ``ContextWrapper`` / ``FunctionToolExecutor`` /
    ``ToolLoopAgentRunner`` and awaits ``runner.reset``. ``provider`` is the only piece
    the caller must supply (host credentials/model); the factory never constructs one.
    ``max_turns`` maps to ``enforce_max_turns`` and ``**runner_kwargs`` pass straight
    through to ``runner.reset`` for advanced tuning (``fallback_providers``,
    ``llm_compress_*``, ``truncate_turns``, ``custom_compressor``, ...).
    """
    basics = build_local_agent_basics(
        skills_root=skills_root,
        contributions=contributions,
        include_fs=include_fs,
        extra_allowed_roots=extra_allowed_roots,
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
