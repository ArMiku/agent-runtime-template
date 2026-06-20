## Dev environment tips

### Setup & tooling (uv)

Uses [uv](https://docs.astral.sh/uv/) for an isolated, reproducible environment. Never use system/anaconda Python — always go through `uv`.

1. **Setup**: `uv sync --extra dev` creates `.venv` (Python `>=3.12`, auto-selected) with all deps + the `dev` extras (pytest, ruff). Commit `uv.lock`.
2. **Always prefix `uv run`** so commands run in the project venv, not the host:
   - Tests: `uv run pytest -q`
   - Lint/format: `uv run ruff format .` then `uv run ruff check .`
   - Samples: `uv run --env-file .env python -m examples.<name>` (reads secrets from `.env`, which is gitignored)
   - REPL/script: `uv run python ...`
3. **Add a dependency** with `uv add <pkg>` (updates `pyproject.toml` + `uv.lock`). Every `import` in the source must be declared in `pyproject.toml` — the clean `.venv` will not mask missing deps the way a bundled interpreter does.
4. **Layout (flat)**: `agent_runtime/` is the *only* importable package; `examples/` and `tests/` live **outside** it as top-level dirs. Run samples as `python -m examples.<name>`, never `python -m agent_runtime.examples.<name>`. Tests share fakes via relative imports (`from .fakes import ...`).
5. **Dependency direction is enforced** by `tests/test_audits.py` (foundation ← core ← provider/tools ← extensions; no cross-extension or host leaks). Any restructuring must keep this audit green — update its `PKG_SRC` path constants when moving files.
6. **Secrets & proxy**: put `OPENAI_API_KEY` etc. in `.env` (already gitignored) and load with `uv run --env-file .env ...`. If your shell sets a SOCKS proxy (`all_proxy=socks5://`), the `socksio` dependency (already declared) lets httpx use it.

### Basic

1. Do not add any report files such as xxx_SUMMARY.md.
2. After finishing, use `ruff format .` and `ruff check .` to format and check the code.
3. When committing, ensure to use conventional commits messages, such as `feat: add new agent for data analysis` or `fix: resolve bug in provider manager`.
4. Use English for all new comments.
5. For path handling, use `pathlib.Path` instead of string paths.

### No Unnecessary Helpers

Prioritize inline implementation over abstraction. Avoid over-engineering and do not create helper functions unless absolutely necessary.

1. **Inline-First Rule**: If a logic block can be implemented directly within the main function without breaking overall readability, **do not** extract it into a new helper function.
2. **Strict Justification for Helpers**: You may only create a separate helper function if it meets at least one of these criteria:
   - **High Reuse**: The exact same logic is repeated across **3 or more** different locations.
   - **Extreme Complexity**: Inlining the logic makes the main function too long (e.g., >50 lines) or severely derails the main execution flow.
3. **No Fragmentation**: Do not split continuous linear logic (e.g., a single API call, simple form validation, or one-time data formatting) into tiny functions just for the sake of "clean code."
4. **Keep Context Compact**: Handle edge cases, error catching, and logging directly inside the main function block instead of offloading them.
5. **Refactoring Constraint**: When modifying existing code, do not alter the current function structure or extract code into new helpers unless the existing code already violates the complexity or reuse rules above.

### Mandatory Google-Style Docstrings
* **Comment the complex**: Add clear comments to any non-obvious function, method, or parameter.
* **Google Format**: All docstrings must strictly use the Google format (`Args:`, `Returns:`, `Raises:`).

#### Example:

```py
def calculate_metrics(user_id: int, force_refresh: bool = False) -> dict:
    """Brief description of the function.

    Args:
        user_id: Description of the ID.
        force_refresh: Description of the flag.

    Returns:
        Description of the returned dict.

    Raises:
        ValueError: Description of when this occurs.
    """
    # Inline implementation here...
```