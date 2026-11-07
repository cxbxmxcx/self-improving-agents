"""Procedures: the functions metacognition learns, declared as typed stubs.

    @procedure(api=[read_calendar])
    def find_shared_starts(attendees: list[str], days: list[str], minutes: int,
                           earliest: str = "", latest_end: str = "") -> dict:
        \"\"\"Find every start time ... when everyone is free.

        Returns: {"starts": {day: ["HH:MM", ...]}}, earliest first.
        \"\"\"
        ...

The stub's name, signature, and docstring become the contract a learned
function must meet (helix's Contract). The learned function takes one more
parameter first, `api`, which the harness binds to the read-only tools listed
in `api=[...]`; the model never sees it. A body of `...` means "learn this".
A real body is a hand-written memory the agent trusts from the start; it must
take `api` first and pass the code-skill guard.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from helix.memory.code_skills import UnsafeSkill, check_source
from helix.memory.learning_mechanisms import Contract

from metacog.exceptions import UserError
from metacog.tool import FunctionTool


@dataclass
class Procedure:
    name: str
    contract: Contract
    # What the model reads about the procedure's tool (and the one line of context).
    description: str
    api: list[FunctionTool] = field(default_factory=list)
    # A hand-written body the agent trusts from the start, if any.
    reference: str | None = None

    @classmethod
    def from_contract(cls, contract: Contract, *, api: Sequence[FunctionTool] = (), reference: str | None = None,
                      description: str | None = None) -> Procedure:
        tools = _check_api(api)
        if reference is not None:
            _check_source(reference, contract.name)
        return cls(contract.name, contract, description or contract.description, tools, reference)


def _check_api(api: Sequence[Any]) -> list[FunctionTool]:
    tools = list(api)
    for tool in tools:
        if not isinstance(tool, FunctionTool) or not tool.read_only:
            raise UserError(f"procedure api entries must be @function_tool(read_only=True) tools, not {tool!r}")
    return tools


def _check_source(source: str, name: str) -> None:
    try:
        check_source(source, name)
    except UnsafeSkill as exc:
        raise UserError(f"procedure {name}'s body fails the code-skill guard: {exc}") from exc


def _annotation(annotation: Any) -> str | None:
    if annotation is inspect.Parameter.empty:
        return None
    return annotation if isinstance(annotation, str) else inspect.formatannotation(annotation)


def _signature(fn: Callable[..., Any], name: str) -> tuple[str, bool]:
    """The contract's def line, with `api` first, and whether the stub already takes `api`."""
    sig = inspect.signature(fn)
    params = list(sig.parameters.values())
    has_api = bool(params) and params[0].name == "api"
    parts = [] if has_api else ["api"]
    for param in params:
        text = param.name
        annotation = _annotation(param.annotation)
        if annotation and param.name != "api":
            text += f": {annotation}"
        if param.default is not inspect.Parameter.empty:
            text += f" = {param.default!r}"
        parts.append(text)
    returns = _annotation(sig.return_annotation)
    return f"def {name}({', '.join(parts)})" + (f" -> {returns}" if returns else "") + ":", has_api


def _split_doc(doc: str) -> tuple[str, str]:
    """(description, returns) from a docstring with an optional `Returns:` section."""
    lines = inspect.cleandoc(doc or "").splitlines()
    for i, line in enumerate(lines):
        if line.strip().lower().startswith("returns:"):
            head = " ".join(part.strip() for part in lines[:i] if part.strip())
            rest = [line.strip()[len("returns:"):].strip(), *(part.strip() for part in lines[i + 1:] if part.strip())]
            return head, " ".join(part for part in rest if part)
    return " ".join(part.strip() for part in lines if part.strip()), ""


def _reference(fn: Callable[..., Any]) -> str | None:
    """The function's source without decorators, or None when its body is only a docstring and `...`."""
    try:
        source = textwrap.dedent(inspect.getsource(fn))
    except (OSError, TypeError):
        return None
    func = ast.parse(source).body[0]
    body = list(func.body)  # type: ignore[attr-defined]
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
            and isinstance(body[0].value.value, str):
        body = body[1:]
    if all(isinstance(s, ast.Pass) or (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant)
                                      and s.value.value is Ellipsis) for s in body):
        return None
    lines = source.splitlines()
    return "\n".join(lines[func.lineno - 1:]) + "\n"


def _api_text(tools: list[FunctionTool]) -> str:
    if not tools:
        return "The `api` argument offers nothing; compute from the other arguments alone."
    return " ".join(f"`api.{tool.name}({', '.join(tool.params_model.model_fields)})` calls the {tool.name} tool"
                    f" and returns exactly what it returns: {tool.description.strip()}" for tool in tools)


def procedure(func: Callable[..., Any] | None = None, *, api: Sequence[FunctionTool] = (), name: str | None = None,
              description: str | None = None, returns: str | None = None) -> Any:
    """Declare a procedure from a typed stub. Usable bare or with arguments."""

    def build(fn: Callable[..., Any]) -> Procedure:
        proc_name = name or fn.__name__
        doc_description, doc_returns = _split_doc(fn.__doc__ or "")
        signature, has_api = _signature(fn, proc_name)
        reference = _reference(fn)
        if reference is not None:
            if not has_api:
                raise UserError(f"procedure {proc_name} has a body, so it must take `api` as its first parameter")
            if proc_name != fn.__name__:
                raise UserError("a procedure with a body keeps its function's name")
        if not (description or doc_description):
            raise UserError(f"procedure {proc_name} needs a docstring or description: the model reads it")
        returned = _annotation(inspect.signature(fn).return_annotation)
        contract = Contract(name=proc_name, signature=signature, description=description or doc_description,
                            api=_api_text(_check_api(api)),
                            returns=returns or doc_returns or (f"Return a {returned}." if returned else ""))
        return Procedure.from_contract(contract, api=api, reference=reference)

    return build(func) if func is not None else build
