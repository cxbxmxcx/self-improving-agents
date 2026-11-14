"""Input and output guardrails, shaped like the OpenAI Agents SDK's.

An input guardrail runs before the first model call and an output guardrail
on the final output. A tripped guardrail becomes a helix Refusal, which ends
the run at once (SPEC §6.2), and the Runner raises the matching tripwire
exception. Input guardrails run before the model rather than beside it, so a
tripped one never spends a model call.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass
class GuardrailFunctionOutput:
    output_info: Any
    tripwire_triggered: bool


async def _call(fn: Callable[..., Any], *args: Any) -> GuardrailFunctionOutput:
    result = fn(*args)
    return await result if inspect.isawaitable(result) else result


@dataclass
class InputGuardrailResult:
    guardrail: InputGuardrail
    output: GuardrailFunctionOutput


@dataclass
class OutputGuardrailResult:
    guardrail: OutputGuardrail
    agent_output: Any
    output: GuardrailFunctionOutput


@dataclass
class InputGuardrail:
    """`guardrail_function(context, agent, input)` returning a GuardrailFunctionOutput."""

    guardrail_function: Callable[..., Any]
    name: str | None = None

    def get_name(self) -> str:
        return self.name or self.guardrail_function.__name__

    async def run(self, context: Any, agent: Any, input: Any) -> InputGuardrailResult:
        return InputGuardrailResult(self, await _call(self.guardrail_function, context, agent, input))


@dataclass
class OutputGuardrail:
    """`guardrail_function(context, agent, output)` returning a GuardrailFunctionOutput."""

    guardrail_function: Callable[..., Any]
    name: str | None = None

    def get_name(self) -> str:
        return self.name or self.guardrail_function.__name__

    async def run(self, context: Any, agent: Any, output: Any) -> OutputGuardrailResult:
        return OutputGuardrailResult(self, output, await _call(self.guardrail_function, context, agent, output))


def input_guardrail(func: Callable[..., Any] | None = None, *, name: str | None = None) -> Any:
    def wrap(fn: Callable[..., Any]) -> InputGuardrail:
        return InputGuardrail(fn, name=name)

    return wrap(func) if func is not None else wrap


def output_guardrail(func: Callable[..., Any] | None = None, *, name: str | None = None) -> Any:
    def wrap(fn: Callable[..., Any]) -> OutputGuardrail:
        return OutputGuardrail(fn, name=name)

    return wrap(func) if func is not None else wrap
