"""A safe arithmetic tool exposed to the agent as ``calculate``.

The handler evaluates a math expression via a whitelisted ``ast`` walk — no
``eval``, no name lookups, no calls — so the model can do exact numeric work
(``+ - * / // % **``, parentheses, unary sign) without arbitrary code execution.
"""

from __future__ import annotations

import ast
import operator

from agent_runtime.core.run_context import ContextWrapper
from agent_runtime.core.tool import FunctionTool

# Binary/unary operators allowed in an expression. Anything outside this map
# (calls, names, attribute access, comprehensions, ...) is rejected by the walk.
_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _eval_node(node: ast.AST) -> float:
    """Recursively evaluate a parsed arithmetic node.

    Args:
        node: An ``ast`` node from the parsed expression.

    Returns:
        The numeric value of the sub-expression.

    Raises:
        ValueError: If the node is not part of the allowed arithmetic grammar.
    """
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError(f"Unsupported constant: {node.value!r}")
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        # Guard the most common runtime trap so the model gets a clean message.
        if isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)) and right == 0:
            raise ValueError("division by zero")
        if isinstance(node.op, ast.Pow) and abs(right) > 1000:
            raise ValueError("exponent too large")
        return _BIN_OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_node(node.operand))
    raise ValueError(f"Unsupported expression element: {type(node).__name__}")


async def _calculate(context: ContextWrapper, **kwargs: object) -> str:
    """Evaluate an arithmetic expression and return the result as text.

    Args:
        context: The run context (unused; required by the tool contract).
        **kwargs: Expects ``expression`` — the math expression to evaluate.

    Returns:
        The computed value as a string, or an ``error: ...`` message the model
        can read and recover from.
    """
    expression = str(kwargs.get("expression", "")).strip()
    if not expression:
        return "error: 'expression' is required."
    try:
        tree = ast.parse(expression, mode="eval")
        result = _eval_node(tree.body)
    except ValueError as exc:
        return f"error: {exc}"
    except SyntaxError:
        return f"error: '{expression}' is not a valid arithmetic expression."

    # Render integral floats without a trailing ".0" for a tidy answer.
    if isinstance(result, float) and result.is_integer():
        return str(int(result))
    return str(result)


def build_calculate_tool() -> FunctionTool:
    """Build the ``calculate`` function tool.

    Returns:
        A :class:`FunctionTool` the agent can call to evaluate arithmetic.
    """
    return FunctionTool(
        name="calculate",
        description=(
            "Evaluate an arithmetic expression and return the exact result. "
            "Supports + - * / // % ** and parentheses, e.g. '(12 + 8) * 3 / 4'. "
            "Use this for any numeric computation instead of doing mental math."
        ),
        parameters={
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The arithmetic expression to evaluate, e.g. '2 ** 10 + 24'.",
                },
            },
            "required": ["expression"],
        },
        handler=_calculate,
    )
