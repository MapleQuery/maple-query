"""Exact arithmetic for the agent.

Language models get ratios and compound changes wrong often enough to
matter ("36.1% ÷ 1.297 = 9.5%" for a real change of 4.9%). Every derived
figure in an answer goes through here instead: an expression over
numbers, + - * / ** and parentheses, evaluated by walking the parsed
AST — never `eval`, so nothing but arithmetic can run.
"""

from __future__ import annotations

import ast
import operator
from typing import Any

_BIN = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCS = {"round": round, "abs": abs, "min": min, "max": max}
MAX_LEN = 500


class CalcError(ValueError):
    pass


def evaluate(expression: str) -> float:
    if len(expression) > MAX_LEN:
        raise CalcError(f"expression longer than {MAX_LEN} characters")
    cleaned = expression.replace(",", "").replace("\u00d7", "*").replace("\u00f7", "/")
    try:
        tree = ast.parse(cleaned, mode="eval")
    except SyntaxError as exc:
        raise CalcError(f"not an arithmetic expression: {expression!r}") from exc
    value = _eval(tree.body)
    if isinstance(value, complex):
        raise CalcError("result is not a real number")
    return float(value)


def _eval(node: ast.AST) -> Any:
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise CalcError("exponent too large")
        try:
            return _BIN[type(node.op)](left, right)
        except ZeroDivisionError as exc:
            raise CalcError("division by zero") from exc
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_eval(node.operand))
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in _FUNCS
        and not node.keywords
    ):
        return _FUNCS[node.func.id](*(_eval(a) for a in node.args))
    raise CalcError("only numbers, + - * / ** ( ) and round/abs/min/max are allowed")
