"""Mechanism runtime: learning mechanisms held as source, run through a metered lab.

Chapter 6's learning mechanisms were classes a person wrote. Chapter 7 puts
them under search, so a mechanism becomes source text that a designer model
can rewrite: top-level constants such as its writer prompts, helper
functions, and one entry point, `async def learn(lab)`, returning the
learned function's source (or a dict with "source" and optional "values").

The lab is the mechanism's only capability. It offers the experience, the
contract, the writer, and practice on observed requests, and it meters each:
writer calls, practice calls, and wall-clock time all have limits, and a
design that exceeds one stops with the reason recorded. The checker's verdict
reaches the mechanism through practice and cannot be changed by it.

Mechanism source runs under a guard in the spirit of the code-skill guard:
parsed first and refused for imports, private names, bare `except`, and
`str.format` (whose field syntax can reach attributes; templates go through
`lab.fill`). Like that guard, it contains honest mistakes in a teaching
harness and is not a security sandbox; source from an untrusted designer
belongs in a separate process or container.
"""

from __future__ import annotations

import ast
import string
import sys
import time
from dataclasses import dataclass
from typing import Any

from helix.memory.code_skills import SAFE_BUILTINS, UnsafeSkill
from helix.memory.learning_mechanisms import (Act, Contract, Episode, LearnedMemory, Writer, contract_text,
                                              render_episode)

ENTRY = "learn"
# Decision 3 of the chapter 7 plan: writer effort is searchable, high is not on offer.
EFFORTS = ("none", "low", "medium")
MECHANISM_BUILTINS = dict(SAFE_BUILTINS)
_TOP_LEVEL = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Assign, ast.AnnAssign, ast.Expr)
# str.format and format_map resolve "{x.attr}" and "{x[key]}" fields, a known way out of a namespace.
_REFUSED_ATTRIBUTES = {"format", "format_map"}


class UnsafeMechanism(ValueError):
    """A mechanism's source failed the guard and was not run."""


class MechanismStopped(Exception):
    """A mechanism ran and stopped without a learned function: over budget, out of time, or failed."""

    def __init__(self, reason: str, writer_calls: int, usage: list[dict[str, int]] | None = None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.writer_calls = writer_calls
        # A design that stopped still spent; its calls count toward the search's cost.
        self.usage = list(usage or [])


class _Stop(BaseException):
    # BaseException so a design's `except Exception` cannot swallow its own budget stop.
    pass


class _HarnessFault(BaseException):
    # A failure inside the lab (a missing recording, a network error) is the harness's,
    # not the design's: it must reach the caller rather than score the design zero.
    def __init__(self, original: Exception) -> None:
        super().__init__(str(original))
        self.original = original


@dataclass(frozen=True)
class MechanismBudget:
    """Limits a design runs within; None leaves that resource unlimited."""

    writer_calls: int | None = None
    practice_calls: int | None = None
    seconds: float = 3600.0


def check_mechanism(source: str) -> None:
    """Refuse mechanism source the guard does not allow; return quietly when it passes."""
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise UnsafeMechanism(f"the mechanism does not parse: {exc.msg} (line {exc.lineno})") from exc
    for node in tree.body:
        if not isinstance(node, _TOP_LEVEL):
            raise UnsafeMechanism(f"only functions and constants may sit at the top level, not "
                                  f"{type(node).__name__}")
        value = node.value if isinstance(node, (ast.Assign, ast.AnnAssign, ast.Expr)) else None
        if value is not None:
            try:
                ast.literal_eval(value)
            except ValueError as exc:
                raise UnsafeMechanism("top-level values must be literals") from exc
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise UnsafeMechanism("mechanisms may not import")
        if isinstance(node, (ast.Global, ast.Nonlocal)):
            raise UnsafeMechanism(f"mechanisms may not use {type(node).__name__.lower()}")
        if isinstance(node, ast.ExceptHandler) and node.type is None:
            raise UnsafeMechanism("mechanisms may not use a bare except; name the exception")
        if isinstance(node, ast.Attribute) and node.attr in _REFUSED_ATTRIBUTES:
            raise UnsafeMechanism(f"mechanisms may not call .{node.attr}(); fill templates with lab.fill")
        name = (node.id if isinstance(node, ast.Name) else node.attr if isinstance(node, ast.Attribute)
                else node.name if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                else node.arg if isinstance(node, ast.arg) else None)
        if name and name.startswith("_"):
            raise UnsafeMechanism(f"mechanisms may not name {name!r}")
    entry = next((n for n in tree.body if isinstance(n, ast.AsyncFunctionDef) and n.name == ENTRY), None)
    if entry is None:
        raise UnsafeMechanism(f"the mechanism does not define async def {ENTRY}(lab)")
    if len(entry.args.args) != 1:
        raise UnsafeMechanism(f"{ENTRY}() takes exactly one argument, the lab")


@dataclass
class _MeteredWriter(Writer):
    # Writer.write calls ask once or twice; metering ask covers both.
    lab: Any = None

    async def ask(self, prompt: str, *, effort: str | None = None) -> str:
        self.lab._spend("writer")
        if effort is not None and effort not in EFFORTS:
            raise ValueError(f"writer effort must be one of {EFFORTS}, not {effort!r}")
        try:
            return await super().ask(prompt, effort=effort)
        except Exception as exc:
            raise _HarnessFault(exc) from exc


class Lab:
    """Everything a mechanism may use, metered. Private attributes are out of a design's reach."""

    Refused = UnsafeSkill

    def __init__(self, episodes: list[Episode], act: Act, contract: Contract, writer: Writer,
                 budget: MechanismBudget | None = None) -> None:
        self.episodes = list(episodes)
        self.contract = contract_text(contract)
        self.entry = contract.name
        self._contract = contract
        self._act = act
        self._budget = budget or MechanismBudget()
        self._writer = _MeteredWriter(model=writer.model, params=dict(writer.params), timeout=writer.timeout,
                                      lab=self)
        self._practiced = 0
        self._deadline = float("inf")
        self._stopped: str | None = None
        self._notes: list[str] = []

    @property
    def writer_calls(self) -> int:
        return self._writer.calls

    @property
    def writer_usage(self) -> list[dict[str, int]]:
        return list(self._writer.usage)

    # ------------------------------------------------------------ the writer

    async def ask(self, prompt: str, effort: str | None = None) -> str:
        """The writer's reply, as text."""
        return await self._writer.ask(prompt, effort=effort)

    async def write(self, prompt: str, effort: str | None = None) -> str:
        """A function from the writer that passes the memory guard; raises lab.Refused if it never does."""
        return await self._writer.write(prompt, self._contract, effort=effort)

    # ------------------------------------------------------------ practice

    async def practice(self, source: str, task_id: str) -> tuple[float, str, str]:
        """Try a function on one observed request: (reward, the user's reason, the calendars it read)."""
        self._spend("practice")
        try:
            return await self._act(source, task_id)
        except Exception as exc:
            raise _HarnessFault(exc) from exc

    # ------------------------------------------------------------ plain helpers

    def stream(self) -> list[str]:
        """The distinct requests, in the order they were experienced."""
        return list(dict.fromkeys(e.task_id for e in self.episodes))

    def request(self, task_id: str) -> str:
        """The request text of the first episode for a task."""
        return next(e.request for e in self.episodes if e.task_id == task_id)

    def render(self, episode: Episode) -> str:
        """An episode as chapter 6's writers read it; a design may build its own view instead."""
        return render_episode(episode)

    def fill(self, template: str, **values: Any) -> str:
        """str.format for plain names only, so a template cannot reach attributes or items."""
        for _, name, _, _ in string.Formatter().parse(template):
            if name is not None and not name.isidentifier():
                raise ValueError(f"lab.fill takes plain names only, not {{{name}}}")
        return template.format(**values)

    def note(self, text: Any) -> None:
        """Something the design wants a person, or the designer, to read later."""
        self._notes.append(str(text))

    # ------------------------------------------------------------ metering

    def _spend(self, resource: str) -> None:
        self._check_time()
        if self._stopped:
            raise _Stop(self._stopped)
        if resource == "writer":
            limit, used = self._budget.writer_calls, self._writer.calls
        else:
            limit, used = self._budget.practice_calls, self._practiced
            self._practiced += 1
        if limit is not None and used >= limit:
            self._stopped = f"the mechanism used its {limit} {resource} calls"
            raise _Stop(self._stopped)

    def _check_time(self) -> None:
        if time.monotonic() > self._deadline:
            self._stopped = self._stopped or f"the mechanism ran past {self._budget.seconds:g} seconds"
            raise _Stop(self._stopped)


async def run_mechanism(source: str, lab: Lab, *, name: str = "mechanism") -> LearnedMemory:
    """The function a mechanism learns, run once under the guard, the lab's budget, and a deadline.

    Raises UnsafeMechanism before running refused source, and MechanismStopped
    when the design runs out of budget or time, or fails, so a search can
    score it zero and keep the reason. One mechanism runs at a time per thread.
    """
    check_mechanism(source)
    filename = f"<mechanism:{name}>"
    namespace: dict[str, Any] = {"__builtins__": MECHANISM_BUILTINS}
    # Top-level code is literals and definitions only, so loading runs nothing.
    exec(compile(source, filename, "exec"), namespace)
    learn = namespace[ENTRY]

    def line_tracer(frame: Any, event: str, arg: Any) -> Any:
        lab._check_time()
        return line_tracer

    def tracer(frame: Any, event: str, arg: Any) -> Any:
        # Trace only the design's own frames; a loop with no lab call still meets the deadline.
        return line_tracer if frame.f_code.co_filename == filename else None

    lab._deadline = time.monotonic() + lab._budget.seconds
    previous = sys.gettrace()
    sys.settrace(tracer)
    try:
        result = await learn(lab)
    except _Stop as exc:
        raise MechanismStopped(str(exc), lab.writer_calls, lab.writer_usage) from None
    except _HarnessFault as fault:
        raise fault.original from None
    except Exception as exc:  # the design's own bug, kept as the reason it scored nothing
        raise MechanismStopped(f"the mechanism failed: {type(exc).__name__}: {exc}", lab.writer_calls, lab.writer_usage) from exc
    finally:
        sys.settrace(previous)
    values: dict[str, float] = {}
    if isinstance(result, dict):
        values = dict(result.get("values") or {})
        result = result.get("source")
    if not isinstance(result, str) or not result.strip():
        raise MechanismStopped("the mechanism returned no function source", lab.writer_calls, lab.writer_usage)
    return LearnedMemory(name, result, lab.writer_calls, list(lab._notes), values, lab.writer_usage)
