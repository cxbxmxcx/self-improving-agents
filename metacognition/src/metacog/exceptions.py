"""Errors a run can raise, named as the OpenAI Agents SDK names them."""

from __future__ import annotations

from typing import Any


class AgentsException(Exception):
    """Base class for errors metacog raises during a run."""


class UserError(AgentsException):
    """The SDK was used in a way it does not support."""


class ModelBehaviorError(AgentsException):
    """The model produced something the run cannot use, such as output that fails `output_type`."""


class MaxTurnsExceeded(AgentsException):
    """The run used every turn without reaching a final output."""

    def __init__(self, message: str, trajectory: Any = None) -> None:
        super().__init__(message)
        self.trajectory = trajectory


class InputGuardrailTripwireTriggered(AgentsException):
    """An input guardrail tripped; the model was never called."""

    def __init__(self, guardrail_result: Any) -> None:
        self.guardrail_result = guardrail_result
        super().__init__(f"Input guardrail {guardrail_result.guardrail.get_name()} triggered tripwire")


class OutputGuardrailTripwireTriggered(AgentsException):
    """An output guardrail tripped on the agent's final output."""

    def __init__(self, guardrail_result: Any) -> None:
        self.guardrail_result = guardrail_result
        super().__init__(f"Output guardrail {guardrail_result.guardrail.get_name()} triggered tripwire")
