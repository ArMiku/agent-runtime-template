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
PKG_SRC = PKG_ROOT / "agent_runtime"

# Third-party distributions the runtime is allowed to depend on (declared in
# pyproject.toml / requirements.txt), keyed by their top-level import name.
ALLOWED_THIRD_PARTY = {
    "aiohttp",
    "anthropic",
    "anyio",
    "certifi",
    "deprecated",
    "docstring_parser",
    "google",  # google-genai, TYPE_CHECKING-only (Gemini deferred)
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
    "yaml",
}
ALLOWED_FIRST_PARTY = {"agent_runtime"}


def _python_files() -> list[Path]:
    return [p for p in PKG_SRC.rglob("*.py") if "site-packages" not in str(p)]


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
    runner_src = (PKG_SRC / "core" / "runners" / "tool_loop_agent_runner.py").read_text(encoding="utf-8")
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


# --- skills subsystem: dependency direction + no host/sandbox/neo residue ----

SKILLS_DIR = PKG_SRC / "extensions" / "skills"


def _absolute_import_modules(tree: ast.AST) -> list[str]:
    """Full dotted module names of every absolute (level==0) import in the tree."""
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                modules.append(node.module)
    return modules


def _skills_python_files() -> list[Path]:
    return sorted(SKILLS_DIR.rglob("*.py"))


def test_skills_layer_only_depends_inward():
    """The skills extension may import core/foundation (inward) but never the host
    application, never the plugin layer, and never another extension.

    * no module rooted at ``agent_runtime.extensions.plugins`` (the two extensions are
      wired together only via plain path lists at the host, design §7)
    * no module rooted at any other ``agent_runtime.extensions.*`` subpackage
    * every first-party import resolves under ``agent_runtime`` (core/foundation/self)
    """
    assert SKILLS_DIR.is_dir(), "extensions/skills must exist"
    offenders: list[str] = []
    for path in _skills_python_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for module in _absolute_import_modules(tree):
            top = module.split(".")[0]
            if module.startswith("agent_runtime.extensions.plugins"):
                offenders.append(f"{path.relative_to(PKG_ROOT)}: imports '{module}' (skills must not import plugins)")
                continue
            if module.startswith("agent_runtime.extensions.") and not module.startswith(
                "agent_runtime.extensions.skills"
            ):
                offenders.append(f"{path.relative_to(PKG_ROOT)}: imports '{module}' (no cross-extension deps)")
                continue
            if top == "agent_runtime":
                continue  # inward first-party import — allowed
    assert not offenders, "skills layer dependency violations:\n" + "\n".join(offenders)


def test_skills_layer_has_no_host_or_sandbox_residue():
    """Public symbols and source under extensions/skills must not carry host / sandbox /
    remote-market branding (``sandbox`` / ``neo``)."""
    banned = ("sandbox", "neo")
    offenders: list[str] = []
    for path in _skills_python_files():
        text = path.read_text(encoding="utf-8")
        lowered = text.lower()
        for token in banned:
            if token in lowered:
                offenders.append(f"{path.relative_to(PKG_ROOT)}: contains '{token}'")
    assert not offenders, "host/sandbox/neo residue in skills layer:\n" + "\n".join(offenders)


# --- fs subsystem: dependency direction + no host/sandbox/neo residue --------

FS_DIR = PKG_SRC / "extensions" / "fs"


def _fs_python_files() -> list[Path]:
    return sorted(FS_DIR.rglob("*.py"))


def test_fs_layer_only_depends_inward():
    """The fs extension may import core/foundation (inward) but never the host
    application, never another extension, and never carry host branding.

    * no module rooted at ``agent_runtime.extensions.plugins`` or
      ``agent_runtime.extensions.skills`` (fs is wired to skills only via plain path
      lists handed in by the host — design §7; no direct import)
    * no module rooted at any other ``agent_runtime.extensions.*`` subpackage
    * every first-party import resolves under ``agent_runtime`` (core/foundation/self)
    """
    assert FS_DIR.is_dir(), "extensions/fs must exist"
    offenders: list[str] = []
    for path in _fs_python_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for module in _absolute_import_modules(tree):
            top = module.split(".")[0]
            if module.startswith("agent_runtime.extensions.skills"):
                offenders.append(f"{path.relative_to(PKG_ROOT)}: imports '{module}' (fs must not import skills)")
                continue
            if module.startswith("agent_runtime.extensions.plugins"):
                offenders.append(f"{path.relative_to(PKG_ROOT)}: imports '{module}' (fs must not import plugins)")
                continue
            if module.startswith("agent_runtime.extensions.") and not module.startswith(
                "agent_runtime.extensions.fs"
            ):
                offenders.append(f"{path.relative_to(PKG_ROOT)}: imports '{module}' (no cross-extension deps)")
                continue
            if top == "agent_runtime":
                continue  # inward first-party import — allowed
    assert not offenders, "fs layer dependency violations:\n" + "\n".join(offenders)


def test_fs_layer_has_no_host_or_sandbox_residue():
    """Public symbols and source under extensions/fs must not carry host / sandbox /
    remote-market branding (``sandbox`` / ``neo``)."""
    banned = ("sandbox", "neo")
    offenders: list[str] = []
    for path in _fs_python_files():
        text = path.read_text(encoding="utf-8")
        lowered = text.lower()
        for token in banned:
            if token in lowered:
                offenders.append(f"{path.relative_to(PKG_ROOT)}: contains '{token}'")
    assert not offenders, "host/sandbox/neo residue in fs layer:\n" + "\n".join(offenders)


# --- composition root: token-neutral branding (import direction NOT constrained) ---

COMPOSITION_ROOT_FILES = (
    PKG_SRC / "local_runtime.py",
    PKG_SRC / "core" / "hooks_chain.py",
)


def test_composition_root_has_no_host_or_sandbox_residue():
    """The composition root (``local_runtime.py``) and the core chain primitive
    (``core/hooks_chain.py``) may depend inward on every extension, but their ``.py``
    source MUST NOT carry host / sandbox / remote-market branding
    (``sandbox`` / ``neo``). Import *direction* is intentionally not
    constrained here — the composition root is the one place allowed to wire everything."""
    banned = ("sandbox", "neo")
    offenders: list[str] = []
    for path in COMPOSITION_ROOT_FILES:
        assert path.is_file(), f"{path} must exist"
        lowered = path.read_text(encoding="utf-8").lower()
        for token in banned:
            if token in lowered:
                offenders.append(f"{path.relative_to(PKG_ROOT)}: contains '{token}'")
    assert not offenders, "host/sandbox/neo residue in composition root:\n" + "\n".join(offenders)


# --- tool-management layer: lives in `tools/`, not `provider/`; depends only inward ----

TOOLS_DIR = PKG_SRC / "tools"
PROVIDER_DIR = PKG_SRC / "provider"


def test_tool_manager_lives_in_tools_layer_not_provider():
    """The tool registry / MCP-lifecycle manager is a tool concern, not a provider
    concern. It MUST live under ``tools/`` and MUST NOT reappear in ``provider/``."""
    assert (TOOLS_DIR / "func_tool_manager.py").is_file(), "tools/func_tool_manager.py must exist"
    assert not (PROVIDER_DIR / "func_tool_manager.py").exists(), (
        "func_tool_manager.py must not live under provider/ (it is a tool concern)"
    )
    for path in sorted(PROVIDER_DIR.rglob("*.py")):
        src = path.read_text(encoding="utf-8")
        assert "class FunctionToolManager" not in src, (
            f"{path.relative_to(PKG_ROOT)}: FunctionToolManager must not be defined in the provider layer"
        )
        assert "FuncCall = FunctionToolManager" not in src, (
            f"{path.relative_to(PKG_ROOT)}: the tool registry alias must not live in the provider layer"
        )


def test_provider_layer_does_not_import_tools_layer():
    """The provider layer is inward of the tool-management layer: ``provider/`` MUST NOT
    import ``agent_runtime.tools`` (tool primitives come from ``core.tool`` instead)."""
    offenders: list[str] = []
    for path in sorted(PROVIDER_DIR.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for module in _absolute_import_modules(tree):
            if module == "agent_runtime.tools" or module.startswith("agent_runtime.tools."):
                offenders.append(f"{path.relative_to(PKG_ROOT)}: imports '{module}' (provider must not import tools)")
    assert not offenders, "provider→tools dependency violations:\n" + "\n".join(offenders)


def test_tools_layer_only_depends_inward():
    """The tools layer may import ``core`` / ``foundation`` (inward) but never the
    provider layer, the extensions, or the host application."""
    assert TOOLS_DIR.is_dir(), "tools/ must exist"
    offenders: list[str] = []
    for path in sorted(TOOLS_DIR.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for module in _absolute_import_modules(tree):
            if module.startswith("agent_runtime.provider"):
                offenders.append(f"{path.relative_to(PKG_ROOT)}: imports '{module}' (tools must not import provider)")
            elif module.startswith("agent_runtime.extensions"):
                offenders.append(f"{path.relative_to(PKG_ROOT)}: imports '{module}' (tools must not import extensions)")
    assert not offenders, "tools layer dependency violations:\n" + "\n".join(offenders)


# --- logging discipline: library etiquette as a copy-this template ----------
#
# agent_runtime is a second-development template: this source is meant to be
# copied verbatim by downstream authors, so a logging anti-pattern here is
# multiplied across every fork. These audits freeze two rules:
#   1. error logging inside an ``except`` block MUST preserve the traceback
#      (``logger.exception(...)`` or ``logger.error(..., exc_info=...)``).
#   2. library source MUST NOT use bare ``print(...)`` (``examples/`` excepted —
#      sample scripts are terminal-facing).

_LIBRARY_DIRS = ("foundation", "core", "provider", "tools", "message", "media", "extensions")


def _library_python_files() -> list[Path]:
    """All package source files except tests/ and examples/."""
    files: list[Path] = []
    for sub in _LIBRARY_DIRS:
        files.extend((PKG_SRC / sub).rglob("*.py"))
    files.extend(p for p in PKG_SRC.glob("*.py"))  # top-level modules
    return sorted(files)


def _call_has_traceback_kwarg(call: ast.Call) -> bool:
    """True if a logging call carries ``exc_info=`` (any truthy form)."""
    return any(kw.arg == "exc_info" for kw in call.keywords)


def _is_logger_error(call: ast.Call) -> bool:
    """Match ``logger.error(...)`` (attribute on the module ``logger`` name).

    Scoped to ``error`` only — an ``except`` block's ``logger.warning`` is often a
    deliberate recoverable-fallback notice (e.g. an optional-dependency import
    failing) where a traceback would be noise. Error-level logs in a handler,
    however, almost always want the stack. This mirrors the spec scenario, which
    targets ``logger.error`` specifically.
    """
    func = call.func
    return (
        isinstance(func, ast.Attribute)
        and func.attr == "error"
        and isinstance(func.value, ast.Name)
        and func.value.id == "logger"
    )


def test_except_blocks_preserve_traceback():
    """Inside an ``except`` handler, ``logger.error`` MUST carry ``exc_info`` (or use
    ``logger.exception``); otherwise the traceback is silently dropped."""
    offenders: list[str] = []
    for path in _library_python_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for handler in (n for n in ast.walk(tree) if isinstance(n, ast.ExceptHandler)):
            for call in (c for c in ast.walk(handler) if isinstance(c, ast.Call)):
                if _is_logger_error(call) and not _call_has_traceback_kwarg(call):
                    offenders.append(
                        f"{path.relative_to(PKG_ROOT)}:{call.lineno}: "
                        f"logger.error(...) in except block drops traceback "
                        f"(use logger.exception or exc_info=...)"
                    )
    assert not offenders, "traceback-dropping error logs in except blocks:\n" + "\n".join(offenders)


def test_library_source_has_no_bare_print():
    """Library source MUST NOT use bare ``print(...)`` (examples/ is exempt)."""
    offenders: list[str] = []
    for path in _library_python_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for call in (c for c in ast.walk(tree) if isinstance(c, ast.Call)):
            func = call.func
            if isinstance(func, ast.Name) and func.id == "print":
                offenders.append(f"{path.relative_to(PKG_ROOT)}:{call.lineno}: bare print(...)")
    assert not offenders, "bare print() in library source:\n" + "\n".join(offenders)
