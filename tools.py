"""Tools the agent can call. Every tool takes ONE string and returns a string."""
from __future__ import annotations

import ast
import datetime as dt
import html
import json
import math
import operator
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Callable


class ToolError(Exception):
    """Raised by a tool when the input is bad. The agent turns it into an observation."""


@dataclass
class Tool:
    name: str
    description: str
    func: Callable[[str], str]

    def run(self, arg: str) -> str:
        return self.func(arg)


# ---------------------------------------------------------------- calculator
_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv, ast.USub: operator.neg, ast.UAdd: operator.pos,
}
_FUNCS = {
    "sqrt": math.sqrt, "log": math.log, "log10": math.log10, "sin": math.sin,
    "cos": math.cos, "tan": math.tan, "abs": abs, "round": round,
    "floor": math.floor, "ceil": math.ceil,
}
_CONSTS = {"pi": math.pi, "e": math.e}


def _eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.Name) and node.id in _CONSTS:
        return _CONSTS[node.id]
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 1000:
            raise ToolError("Exponent too large (max 1000).")
        return _OPS[type(node.op)](left, right)
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id in _FUNCS and not node.keywords):
        return _FUNCS[node.func.id](*[_eval(a) for a in node.args])
    raise ToolError("Unsupported expression. Use numbers, + - * / ** %, parentheses, "
                    f"and functions: {', '.join(_FUNCS)}.")


def calculator(expression: str) -> str:
    try:
        result = _eval(ast.parse(expression.strip(), mode="eval"))
    except ToolError:
        raise
    except ZeroDivisionError:
        raise ToolError("Division by zero.")
    except (SyntaxError, ValueError, TypeError) as exc:
        raise ToolError(f"Could not evaluate '{expression}': {exc}")
    if isinstance(result, float) and result.is_integer():
        result = int(result)
    elif isinstance(result, float):
        result = float(f"{result:.10g}")
    return str(result)


# ------------------------------------------------------------------ datetime
def current_datetime(_: str = "") -> str:
    now = dt.datetime.now()
    return f"{now:%Y-%m-%d %H:%M} ({now:%A}). Year={now.year}, month={now.month}, day={now.day}."


# ---------------------------------------------------------------- web search
def _wikipedia_search(query: str) -> str:
    url = ("https://en.wikipedia.org/w/api.php?action=query&list=search&format=json"
           f"&srlimit=3&srsearch={urllib.parse.quote(query)}")
    req = urllib.request.Request(url, headers={"User-Agent": "react-agent-demo/1.0"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        data = json.load(resp)
    hits = data.get("query", {}).get("search", [])
    return "\n".join(
        f"- {h['title']}: {html.unescape(re.sub(r'<[^>]+>', '', h['snippet']))}" for h in hits
    )


def web_search(query: str) -> str:
    query = query.strip()
    if not query:
        raise ToolError("Empty search query.")
    text = ""
    try:  # preferred: DuckDuckGo
        from ddgs import DDGS
        results = DDGS().text(query, max_results=3)
        text = "\n".join(f"- {r['title']}: {r['body']}" for r in results)
    except Exception:
        pass
    if not text:  # fallback: Wikipedia API (no key needed)
        try:
            text = _wikipedia_search(query)
        except Exception as exc:
            raise ToolError(f"Search failed: {exc}")
    return text or "No results found. Try a shorter or different query."


def default_tools() -> list[Tool]:
    return [
        Tool("calculator",
             "Evaluate a math expression, e.g. '(15 * 3) / 2' or 'sqrt(1889) * 3'. "
             "Input: the expression only.", calculator),
        Tool("web_search",
             "Search the web for facts (dates, people, places, numbers). "
             "Input: a short search query.", web_search),
        Tool("current_datetime",
             "Get today's date and weekday. Input: any string (ignored).", current_datetime),
    ]
