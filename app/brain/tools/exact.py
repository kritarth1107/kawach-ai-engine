"""Exact answers the model must not guess: counting letters or words, arithmetic, days between dates and weekdays.
Models read text in pieces, not letters, so "how many R in petting zoo" came back as 2 (live 20 Oct: it has none).
Every answer here is computed, never estimated."""

from __future__ import annotations

import ast
import operator
from datetime import date, timedelta

from app.brain.tools.base import ToolRefused, TurnCtx, tool

_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow, ast.USub: operator.neg, ast.UAdd: operator.pos}


def arithmetic(expr: str) -> float:
    """+ - * / // % ** and brackets on numbers only; nothing else is evaluated."""
    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) and not isinstance(n.value, bool):
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
            right = ev(n.right)
            if isinstance(n.op, ast.Pow) and abs(right) > 100:
                raise ValueError("exponent too large")
            return _OPS[type(n.op)](ev(n.left), right)
        if isinstance(n, ast.UnaryOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.operand))
        raise ValueError("only numbers and + - * / % ** ( )")
    text = expr.replace("×", "*").replace("÷", "/").replace(",", "")
    if len(text) > 200:
        raise ValueError("too long")
    return ev(ast.parse(text, mode="eval"))


def count(text: str, of: str) -> int:
    """How many times `of` appears in text, ignoring case (a letter, or a word or phrase)."""
    if not of:
        raise ValueError("say what to count")
    return text.lower().count(of.lower())


@tool(
    "compute",
    "Exact answers you must never guess: how many of a letter or word are in some text, its length or spelling, "
    "arithmetic, days between dates, a date plus days, or a weekday. Say the result exactly as returned.",
    {
        "kind": {"type": "string", "enum": ["count", "length", "spell", "math", "days_between", "add_days", "weekday"]},
        "text": {"type": "string", "description": "The text, the sum (e.g. '12*7+5'), or a date YYYY-MM-DD"},
        "of": {"type": "string", "description": "count: the letter or word; days_between: second date; add_days: days"},
    },
    ["kind", "text"],
)
async def compute(ctx: TurnCtx, a: dict) -> dict:
    kind, text, of = a["kind"], str(a.get("text") or ""), str(a.get("of") or "")
    try:
        if kind == "count":
            return {"result": count(text, of), "counted": of, "in": text}
        if kind == "length":
            return {"letters": sum(ch.isalpha() for ch in text), "characters": len(text), "words": len(text.split())}
        if kind == "spell":
            return {"spelled": "-".join(ch for ch in text if not ch.isspace())}
        if kind == "math":
            value = arithmetic(text)
            return {"result": int(value) if float(value).is_integer() else round(value, 6)}
        if kind == "days_between":
            return {"days": abs((date.fromisoformat(of) - date.fromisoformat(text)).days)}
        if kind == "add_days":
            day = date.fromisoformat(text) + timedelta(days=int(of))
            return {"date": day.isoformat(), "weekday": day.strftime("%A")}
        if kind == "weekday":
            return {"weekday": date.fromisoformat(text).strftime("%A")}
    except (ValueError, SyntaxError, ZeroDivisionError, TypeError) as exc:
        raise ToolRefused(f"Could not compute that: {exc}") from exc
    raise ToolRefused("kind is count, length, spell, math, days_between, add_days or weekday")
