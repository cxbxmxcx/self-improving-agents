"""Decision programs: code that serves a request without a language model, asking Jev only for judgments.

Chapter 6's procedures took the reasoning out of a request; the agent still read
the request, called the procedure and acted on its answer. A decision program
takes the agent out too. It is source with one entry point:

    async def decide(request, kit):
        ...
        return {"action": "file_claim", "arguments": {...}, "reply": "..."}   # served without a model
        return {"handoff": "why"}                                            # the agent (an LLM) serves it
        return {"handoff": "why", "route": "low", "memory": "what applies"}  # the LLM, at that reasoning, reading that

and it may use only what `kit` offers:

- `kit.find(pattern, text)`: the distinct regex matches in `text`, in order (programs can't import `re`);
- `kit.read(name, **arguments)`: a read-only tool of the domain, on this request's world;
- `await kit.procedure(name, **arguments)`: a trusted procedure's answer, under the code-skill guard;
- `await kit.ask(state, questions)`: Jev, TypeSafe's System One model, for typed judgments code can't make:
  `{"type": "choice", "instructions": ..., "criteria": {...}}`, `{"type": "noul", ...}`, `{"type": "score", ...}`.
  Answers come back as plain dicts (`choice`/`confidence`/`probabilities`, `noul`, `score`).

The guard is the learning lab's: no imports, no names starting with `_`, literal top-level constants, and only
the code-skill builtins, so a decision program a model writes runs under the same rules as a mechanism.
Every judgment goes through helix's `Decider` interface; a recorded decision model replays without a key.
"""

from __future__ import annotations

import ast
import asyncio
import json
import re
import time
from collections import Counter
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from helix.deciders.base import (
    Answer,
    Choice,
    Decider,
    DecisionUsage,
    MissingAnswer,
    Probability,
    Question,
    Score,
    ask_measured,
)
from helix.eval.effort import EFFORTS
from helix.memory.mechanism_runtime import MECHANISM_BUILTINS

ENTRY = "decide"
JEV_MODEL = "jev-1.14.0"
_TOP_LEVEL = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Assign, ast.AnnAssign, ast.Expr)
# What a program may use, as a learner reads it in the program's contract.
KIT_GUIDE = """`request` is the user's message, as text. `kit` is the only way to reach anything else:
- `kit.find(pattern, text)`: the distinct regex matches in `text`, in order. The program can't import re.
- `kit.read(name, **arguments)`: one of the read tools below, on this request's data.
- `await kit.procedure(name, **arguments)`: a trusted procedure's answer.
- `await kit.ask(state, questions)`: Jev, a model that returns typed judgments and never writes text. `state` is
  a JSON object; instructions name its fields in backticks, as in `request`. `questions` maps names you choose to
  question objects, all answered in one call. The reply maps the same names to answers:
  {"type": "choice", "instructions": "...", "criteria": {"option": "what this option means", ...}}
      answers {"choice": "the likeliest option", "confidence": 0.0-1.0, "probabilities": {"option": p, ...}}
  {"type": "noul", "instructions": "a yes-or-no question"} answers {"noul": the probability of yes}
  {"type": "score", "instructions": "...", "criteria": ["level 0", "level 1", ...]}
      answers {"score": a number, "confidence": 0.0-1.0}
A choice always picks something, so when nothing may fit, include an option for that, or ask a separate noul
whether anything fits. Treat a noul above 0.7 as yes and below 0.3 as no. To pull a value out of a request, find
the candidates with a broad regex and ask a choice whose options are the candidates themselves, plus "none".
The program must be `async def decide(request, kit)`, with top-level constants that are plain literals, and it
may not use str.format (build strings with + and str())."""
RETURNS = ('Return {"action": name, "arguments": {...}, "reply": "a short message to the user"} to act, or '
           '{"handoff": "why"} to give the request to the agent.')


RETURNS_ROUTED = RETURNS + (' When the agent should serve a request, you can also say how: add "route", the reasoning it '
                            'should use (none, low, medium or high), and "memory", text it reads first with what you '
                            'have learned that applies to this request. Keep what you learn as constants in the '
                            'program, and choose for each request whether to act, or which memory and reasoning to '
                            'hand over. The cheapest way that gets the request right is the best one.')


class UnsafeProgram(ValueError):
    """A decision program's source failed the guard and was not run."""


def check_program(source: str) -> None:
    """The learning lab's guard, with `decide(request, kit)` as the entry point."""
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise UnsafeProgram(f"the program does not parse: {exc.msg} (line {exc.lineno})") from exc
    for node in tree.body:
        if not isinstance(node, _TOP_LEVEL):
            raise UnsafeProgram(f"only functions and constants may sit at the top level, not {type(node).__name__}")
        value = node.value if isinstance(node, (ast.Assign, ast.AnnAssign, ast.Expr)) else None
        if value is not None:
            try:
                ast.literal_eval(value)
            except ValueError as exc:
                raise UnsafeProgram("top-level values must be literals") from exc
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise UnsafeProgram("programs may not import")
        if isinstance(node, (ast.Global, ast.Nonlocal)):
            raise UnsafeProgram(f"programs may not use {type(node).__name__.lower()}")
        if isinstance(node, ast.ExceptHandler) and node.type is None:
            raise UnsafeProgram("programs may not use a bare except; name the exception")
        if isinstance(node, ast.Attribute) and node.attr in ("format", "format_map"):
            raise UnsafeProgram(f"programs may not call .{node.attr}()")
        name = (node.id if isinstance(node, ast.Name) else node.attr if isinstance(node, ast.Attribute)
                else node.name if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                else node.arg if isinstance(node, ast.arg) else None)
        if name and name.startswith("_"):
            raise UnsafeProgram(f"programs may not name {name!r}")
    entry = next((n for n in tree.body if isinstance(n, ast.AsyncFunctionDef) and n.name == ENTRY), None)
    if entry is None or len(entry.args.args) != 2:
        raise UnsafeProgram(f"the program does not define async def {ENTRY}(request, kit)")


def _first_missing(kept: list[ast.AST], now: list[ast.AST]) -> ast.AST | None:
    left = Counter(ast.dump(node) for node in now)
    for node in kept:
        key = ast.dump(node)
        if not left[key]:
            return node
        left[key] -= 1
    return None


def _actions(tree: ast.AST) -> list[ast.AST]:
    return [n for n in ast.walk(tree) if isinstance(n, ast.Dict)
            and any(isinstance(k, ast.Constant) and k.value == "action" for k in n.keys)]


def frozen_change(seed: str, candidate: str) -> str | None:
    """What `candidate` changed that an edit of `seed` must keep, or None if it kept everything.

    A search editing a working program may rewrite how the program judges a request, which is the body of `decide`:
    its questions for Jev, the thresholds on the answers, and its hand-offs. It may add constants and helpers. The rest
    stays as `seed` has it: every constant and helper, every call to `kit.find`, `kit.read`, `kit.procedure` or a
    helper, and every action, with no new one added. Both sources must already have passed `check_program`.
    """
    before, after = ast.parse(seed), ast.parse(candidate)

    def top(tree: ast.Module) -> list[ast.AST]:
        return [n for n in tree.body if not isinstance(n, ast.Expr) and getattr(n, "name", None) != ENTRY]

    helpers = {getattr(n, "name", None) for n in top(before)} - {None}

    def calls(tree: ast.AST) -> list[ast.AST]:
        def frozen(function: ast.AST) -> bool:
            if isinstance(function, ast.Name):
                return function.id in helpers
            return (isinstance(function, ast.Attribute) and isinstance(function.value, ast.Name)
                    and function.value.id == "kit" and function.attr in ("find", "read", "procedure"))

        return [n for n in ast.walk(tree) if isinstance(n, ast.Call) and frozen(n.func)]

    def kept(node: ast.AST | None) -> str | None:
        return None if node is None else f"it changed a frozen part; keep this exactly: {ast.unparse(node)[:200]}"

    added = _first_missing(_actions(after), _actions(before))
    return (kept(_first_missing(top(before), top(after))) or kept(_first_missing(_actions(before), _actions(after)))
            or kept(_first_missing(calls(before), calls(after)))
            or (None if added is None else f"it added an action; an edit may not: {ast.unparse(added)[:200]}"))


def to_question(question: dict[str, Any]) -> Question:
    """A question as a program writes it, in the Decider interface's terms."""
    kind = question.get("type")
    if kind == "choice":
        criteria = question["criteria"]
        return Choice(question["instructions"],
                      dict(criteria) if isinstance(criteria, dict) else {str(option): None for option in criteria})
    if kind == "noul":
        return Probability(question["instructions"], question.get("criteria"))
    if kind == "score":
        return Score(question["instructions"], list(question["criteria"]))
    raise ValueError(f"a question's type is choice, noul or score, not {kind!r}")


def from_answer(answer: Answer) -> dict[str, Any]:
    """An answer as a program reads it: plain values it can compare in code."""
    if answer.kind == "probability":
        return {"type": "noul", "noul": answer.value}
    if answer.kind == "choice":
        return {"type": "choice", "choice": answer.value, "confidence": answer.confidence,
                "probabilities": dict(answer.probabilities)}
    return {"type": "score", "score": answer.value, "confidence": answer.confidence,
            "probabilities": dict(answer.probabilities)}


@dataclass
class Kit:
    """What a decision program may use on one request, metered."""

    reads: dict[str, Callable[..., Any]]
    procedures: dict[str, Callable[..., Awaitable[Any]]]
    decider: Decider | None = None
    usage: DecisionUsage = field(default_factory=DecisionUsage)
    waited: float = 0.0  # wall-clock seconds spent inside the decision model on this request
    log: list[str] = field(default_factory=list)
    answers: dict[str, dict[str, Any]] = field(default_factory=dict)  # what Jev said on this request, by question

    def find(self, pattern: str, text: str) -> list[str]:
        found: list[str] = []
        for match in re.finditer(pattern, text or ""):
            if match.group(0) not in found:
                found.append(match.group(0))
        return found

    def read(self, name: str, **arguments: Any) -> Any:
        if name not in self.reads:
            raise KeyError(f"no read tool {name!r}; the kit offers {sorted(self.reads)}")
        return self.reads[name](**arguments)

    async def procedure(self, name: str, **arguments: Any) -> Any:
        if name not in self.procedures:
            raise KeyError(f"no procedure {name!r}; the kit offers {sorted(self.procedures)}")
        return await self.procedures[name](**arguments)

    async def ask(self, state: Any, questions: dict[str, Any]) -> dict[str, Any]:
        if self.decider is None:
            raise RuntimeError("this kit has no decision model")
        asked = {name: to_question(question) for name, question in questions.items()}
        started = time.perf_counter()
        answers, call = await ask_measured(self.decider, state, asked)
        self.waited += time.perf_counter() - started
        self.usage.add(call)
        replies = {name: from_answer(answer) for name, answer in answers.items()}
        kept = ("choice", "confidence", "noul", "score")
        self.answers.update({name: {key: value for key, value in reply.items() if key in kept}
                             for name, reply in replies.items()})
        return replies


async def run_program(source: str, request: str, kit: Kit, *, timeout: float = 60.0) -> dict[str, Any]:
    """A program's decision on one request: an action with arguments, or a hand-off with its reason."""
    check_program(source)
    namespace: dict[str, Any] = {"__builtins__": MECHANISM_BUILTINS}
    exec(compile(source, "<decision program>", "exec"), namespace)  # top level is definitions and literals only
    decision = await asyncio.wait_for(namespace[ENTRY](request, kit), timeout)
    if not isinstance(decision, dict) or not ("handoff" in decision or "action" in decision):
        return {"handoff": f"the program returned {decision!r}, neither an action nor a hand-off"}
    if "route" in decision and decision["route"] not in EFFORTS:
        decision = {key: value for key, value in decision.items() if key not in ("route", "memory")}
    return decision


@dataclass
class Desk:
    """A domain's side of serving a request with a program: its world, tools, trusted procedures and verdict."""

    world: Callable[[dict[str, Any]], Any]
    reads: Callable[[Any], dict[str, Callable[..., Any]]]
    procedures: Callable[[Any], dict[str, Callable[..., Awaitable[Any]]]]
    actions: dict[str, Callable[..., Any]]  # name -> act(world, **arguments)
    verdict: Callable[[Any, str], dict[str, Any]]  # (world, reply) -> {"success": bool, "reason": str}


@dataclass
class Served:
    """One request served by a program: handled or handed off, right or not, and what it cost."""

    case_id: str
    handled: bool
    right: bool | None
    reason: str
    decider_calls: int
    decider_input_tokens: int
    dollars: float
    seconds: float
    route: str | None = None  # the reasoning the agent should use, when a hand-off says
    memory: str | None = None  # what the agent should read first, when a hand-off says
    answers: dict[str, dict[str, Any]] = field(default_factory=dict)  # what Jev said, for a search to read

    def to_dict(self) -> dict[str, Any]:
        return dict(vars(self))


async def serve(source: str, task: dict[str, Any], desk: Desk, decider: Decider | None) -> Served:
    """Serve one request with a program and judge the result as the agent's would be judged.

    A task with `expect: "handoff"` is one the program should give to the agent: handing it off is right, and
    acting on it is wrong whatever the action was.
    """
    world = desk.world(task)
    kit = Kit(desk.reads(world), desk.procedures(world), decider)
    started = time.perf_counter()
    try:
        decision = await run_program(source, task["text"], kit)
    except (UnsafeProgram, MissingAnswer):
        raise
    except Exception as exc:  # noqa: BLE001 - a program's own failure hands the request off
        decision = {"handoff": f"the program failed: {type(exc).__name__}: {exc}"}
    local = time.perf_counter() - started - kit.waited
    seconds = round(kit.usage.seconds + max(local, 0.0), 4)
    base = {"case_id": task["id"], "decider_calls": kit.usage.calls, "decider_input_tokens": kit.usage.input_tokens,
            "dollars": kit.usage.dollars, "seconds": seconds, "answers": kit.answers}
    expected_handoff = task.get("expect") == "handoff"
    if "handoff" in decision:
        return Served(handled=False, right=True if expected_handoff and not decision.get("route") else None,
                      reason=str(decision["handoff"]), route=decision.get("route"), memory=decision.get("memory"), **base)
    act = desk.actions.get(decision["action"])
    if act is None:
        return Served(handled=False, right=None, reason=f"no action {decision['action']!r}", **base)
    if expected_handoff:
        return Served(handled=True, right=False, reason=f"it took {decision['action']}, but this request should have "
                      "been handed off", **base)
    try:
        act(world, **(decision.get("arguments") or {}))
    except Exception as exc:  # noqa: BLE001 - arguments the action can't take are the program's mistake
        return Served(handled=True, right=False, reason=f"the action failed: {type(exc).__name__}: {exc}", **base)
    verdict = desk.verdict(world, str(decision.get("reply") or ""))
    right, reason = bool(verdict.get("success")), str(verdict.get("reason", ""))
    if not right:  # what it did, so a learner can see the mistake
        reason = f"{reason} (it took {decision['action']} with {json.dumps(decision.get('arguments'), default=str)})"
    return Served(handled=True, right=right, reason=reason, **base)
