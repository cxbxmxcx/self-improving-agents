"""Code skills: procedures kept in procedural memory as functions the agent calls.

A text skill tells the model how to do a task, and the model still does the
work. A code skill does the work: the harness turns each stored function
into a tool, so the model only decides when to call it and with what. This
is how learned memory can remove reasoning rather than describe it.

Skills may be written by a learner model, so they run under a guard. The
source is parsed first and refused if it imports anything, names anything
beginning with an underscore, or defines more than functions and constants.
It then runs with a short list of builtins, with only the objects the
harness binds (a read-only calendar, say), and under a time limit.

This guard keeps honest mistakes contained in a teaching harness. It is
not a security sandbox: code from an untrusted source belongs in a separate
process or container with no access to anything that matters.
"""

from __future__ import annotations

import ast
import asyncio
import builtins
import inspect
import sys
import time
from collections.abc import Callable
from typing import Any, get_type_hints

from pydantic import create_model

from helix.tools import Tool

MAX_RANGE = 1_000_000


class SkillTimeout(Exception):
    """A skill ran past its time limit and was stopped inside its own code."""


def _bounded_range(*args: int) -> range:
    # A huge range loops inside C code, where the deadline cannot reach it.
    r = range(*args)
    if len(r) > MAX_RANGE:
        raise ValueError(f"range of {len(r)} exceeds the skill limit of {MAX_RANGE}")
    return r


# Enough Python to compute with data, nothing that reaches outside it.
SAFE_BUILTINS = {name: getattr(builtins, name) for name in (
    "abs", "all", "any", "bool", "dict", "divmod", "enumerate", "filter", "float", "int",
    "isinstance", "len", "list", "map", "max", "min", "reversed", "round", "set",
    "sorted", "str", "sum", "tuple", "zip", "Exception", "KeyError", "ValueError", "TypeError",
)} | {"range": _bounded_range}
# Statements allowed at the top of a skill's source.
_TOP_LEVEL = (ast.FunctionDef, ast.Assign, ast.AnnAssign, ast.Expr)
DEFAULT_TIMEOUT = 5.0


class UnsafeSkill(ValueError):
    """A skill's source failed the guard and was not run."""


def check_source(source: str, entry: str) -> None:
    """Refuse source the guard does not allow; return quietly when it passes."""
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise UnsafeSkill(f"the skill does not parse: {exc.msg} (line {exc.lineno})") from exc
    for node in tree.body:
        if not isinstance(node, _TOP_LEVEL):
            raise UnsafeSkill(f"only functions and constants may sit at the top level, not {type(node).__name__}")
        # Top-level code runs when the skill loads, outside the time limit,
        # so it may hold literals (constants, a docstring) and nothing else.
        value = node.value if isinstance(node, (ast.Assign, ast.AnnAssign, ast.Expr)) else None
        if value is not None:
            try:
                ast.literal_eval(value)
            except ValueError as exc:
                raise UnsafeSkill("top-level values must be literals") from exc
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise UnsafeSkill("skills may not import")
        if isinstance(node, (ast.Global, ast.Nonlocal, ast.AsyncFunctionDef, ast.Lambda)):
            raise UnsafeSkill(f"skills may not use {type(node).__name__.lower()}")
        name = (node.id if isinstance(node, ast.Name) else node.attr if isinstance(node, ast.Attribute)
                else node.name if isinstance(node, ast.FunctionDef) else node.arg if isinstance(node, ast.arg)
                else None)
        if name and name.startswith("_"):
            raise UnsafeSkill(f"skills may not name {name!r}")
    if not any(isinstance(n, ast.FunctionDef) and n.name == entry for n in tree.body):
        raise UnsafeSkill(f"the skill does not define {entry}()")


def compile_skill(source: str, entry: str) -> Callable[..., Any]:
    """The skill's entry function, compiled under the guard."""
    check_source(source, entry)
    namespace: dict[str, Any] = {"__builtins__": SAFE_BUILTINS}
    exec(compile(source, f"<skill:{entry}>", "exec"), namespace)
    return namespace[entry]


def skill_tool(name: str, description: str, source: str, *, bind: dict[str, Any] | None = None,
               timeout: float = DEFAULT_TIMEOUT) -> Tool:
    """A tool that runs a code skill, with `bind` supplied by the harness.

    The model sees only the parameters the harness does not bind. A skill
    that fails returns its error as the tool result, so the run, and any
    learner reading it later, sees what went wrong.
    """
    bind = dict(bind or {})
    fn = compile_skill(source, name)
    hints = get_type_hints(fn)
    fields: dict[str, Any] = {}
    for param, spec in inspect.signature(fn).parameters.items():
        if param in bind:
            continue
        default = ... if spec.default is inspect.Parameter.empty else spec.default
        # A learned function may leave parameters unannotated; accept anything
        # there rather than guess a type and refuse the call.
        fields[param] = (hints.get(param, Any), default)
    args_model = create_model(f"{name}_Args", **fields)

    def run(**kwargs: Any) -> Any:
        # A thread cannot be killed, so the skill stops itself: a trace
        # function checks the deadline on every line of skill code and
        # raises there, and the thread ends instead of spinning forever.
        deadline = time.monotonic() + timeout

        def tracer(frame: Any, event: str, arg: Any) -> Any:
            if time.monotonic() > deadline:
                raise SkillTimeout(f"{name} took longer than {timeout:g} seconds")
            return tracer

        sys.settrace(tracer)
        try:
            return fn(**bind, **kwargs)
        finally:
            sys.settrace(None)

    async def call(**kwargs: Any) -> Any:
        try:
            return await asyncio.to_thread(run, **kwargs)
        except SkillTimeout as exc:
            return {"error": str(exc)}
        except Exception as exc:  # the skill's own bug, reported to the model
            return {"error": f"{name} failed: {type(exc).__name__}: {exc}"}

    return Tool(name=name, description=description, args_model=args_model, fn=call)


def tools_from_entries(entries: list[Any], *, bind: dict[str, Any] | None = None,
                       timeout: float = DEFAULT_TIMEOUT) -> list[Tool]:
    """Tools for the recalled entries that hold a function; others are context only."""
    return [skill_tool(e.content["skill_name"], e.content.get("description", ""), e.content["body"],
                       bind=bind, timeout=timeout)
            for e in entries if e.content.get("body_kind") == "python"]


async def recalled_skill_tools(tier: Any, task: str, context: Any, *, k: int,
                               bind: dict[str, Any] | None = None, timeout: float = DEFAULT_TIMEOUT) -> list[Tool]:
    """Tools for exactly the functions memory recalls for this request.

    It uses the memory hooks' own retrieval with the same request, context,
    and k, so the function the agent can call is the one its context names.
    Tools are fixed when the agent is built, so the harness calls this
    before construction; the agent loop never changes.
    """
    from helix.memory.hooks import recall_skills

    return tools_from_entries(await recall_skills(tier, task, context, k), bind=bind, timeout=timeout)
