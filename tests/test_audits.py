"""Programmatic audits for the framework-independence / clean-contract requirements.

* 6.6 — every import resolves to the stdlib, a declared third-party dependency, or the
  package itself; no host-application / framework import leaks in.
* 6.7 — tool contract: ``FunctionTool.handler`` is typed to ``ToolExecResult`` (not
  an event-result type); the runner's parameter filter discriminates on
  ``tool.parameters`` (not ``func_tool.handler``).
* 6.8 — no host-framework ``*.core.utils`` import (utils vendored/rewired per design §7.1).
* 6.9 — ``message_event_result`` defines only ``MessageChain`` (event glue dropped).
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import agent_runtime.message.message_event_result as mer_module
from agent_runtime.core.tool import FunctionTool

PKG_ROOT = Path(__file__).resolve().parent.parent

# Third-party distributions the runtime is allowed to depend on (declared in
# pyproject.toml / requirements.txt), keyed by their top-level import name.
ALLOWED_THIRD_PARTY = {
    "aiohttp",
    "anthropic",
    "anyio",
    "certifi",
    "deprecated",
    "docstring_parser",
    "google",  # google-genai, TYPE_CHECKING-only (Gemini deferred, design.md §11)
    "httpx",
    "jsonschema",
    "mcp",
    "openai",
    "PIL",
    "pydantic",
    "pydantic_core",
    "pydub",
    "pytest",
    "tenacity",
    "typing_extensions",
}
ALLOWED_FIRST_PARTY = {"agent_runtime"}


def _python_files() -> list[Path]:
    return [p for p in PKG_ROOT.rglob("*.py") if "site-packages" not in str(p)]


def _absolute_import_roots(tree: ast.AST) -> set[str]:
    """Top-level package names of every absolute import in the tree."""
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            # level > 0 is a relative import (within the package) — always fine.
            if node.level == 0 and node.module:
                roots.add(node.module.split(".")[0])
    return roots


def test_only_stdlib_declared_deps_or_self_imports():
    """6.6: nothing outside stdlib + declared deps + the package itself is imported."""
    allowed = ALLOWED_THIRD_PARTY | ALLOWED_FIRST_PARTY | set(sys.stdlib_module_names)
    offenders: list[str] = []
    for path in _python_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for root in _absolute_import_roots(tree):
            if root not in allowed:
                offenders.append(f"{path.relative_to(PKG_ROOT)}: imports '{root}'")
    assert not offenders, "unexpected (non-stdlib, undeclared, non-self) imports:\n" + "\n".join(offenders)


def test_handler_typed_to_tool_exec_result():
    """6.7: FunctionTool.handler's resolved type is str|CallToolResult (= ToolExecResult),
    never an event-result type."""
    handler_type = str(FunctionTool.__dataclass_fields__["handler"].type)
    # pydantic resolves the ToolExecResult alias to its underlying types.
    assert "CallToolResult" in handler_type
    assert "MessageEventResult" not in handler_type


def test_runner_discriminates_on_parameters_not_handler():
    """6.7: parameter filter anchored on `tool.parameters`, not `func_tool.handler`."""
    runner_src = (PKG_ROOT / "core" / "runners" / "tool_loop_agent_runner.py").read_text(encoding="utf-8")
    # The new discriminator (design §5.1):
    assert 'func_tool.parameters and func_tool.parameters.get("properties")' in runner_src
    # The old handler-existence discriminator must be gone from the filter block:
    assert "if func_tool.handler:" not in runner_src


def test_message_event_result_only_defines_message_chain():
    """6.9: only MessageChain remains; event glue is gone."""
    assert hasattr(mer_module, "MessageChain")
    for dropped in (
        "MessageEventResult",
        "EventResultType",
        "ResultContentType",
        "CommandResult",
    ):
        assert not hasattr(mer_module, dropped), f"{dropped} should have been removed"


def test_function_tool_has_no_host_specific_fields():
    """6.7 (cont.): host-application-specific FunctionTool fields are stripped."""
    fields = set(FunctionTool.__dataclass_fields__)
    assert "handler_module_path" not in fields
    assert "active" not in fields
    assert "is_background_task" not in fields
    assert "handler" in fields  # the retyped, event-free handler stays
