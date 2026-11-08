"""The result of a run, shaped like the OpenAI Agents SDK's RunResult.

Beyond the Agents SDK fields it carries the helix trajectory (the full record
memory learns from and signals score) and the run's cost in dollars and the
seconds its model calls took.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from helix.metacognition.items import ModelResponse, RunItem, messages_from
from helix.metacognition.run_context import RunContextWrapper, Usage


@dataclass
class RunResult:
    input: str | list[dict[str, Any]]
    new_items: list[RunItem]
    raw_responses: list[ModelResponse]
    final_output: Any
    last_agent: Any
    context_wrapper: RunContextWrapper[Any]
    trajectory: Any  # helix.trajectory.Trajectory
    cost: Any  # helix.signals.cost.RunCost: dollars, seconds, tokens
    input_guardrail_results: list[Any] = field(default_factory=list)
    output_guardrail_results: list[Any] = field(default_factory=list)
    trace_id: str | None = None
    # What metacognition did for this run (a helix.metacognition.MetaReport), when the agent has it.
    metacognition: Any = None

    @property
    def usage(self) -> Usage:
        return self.context_wrapper.usage

    def final_output_as(self, cls: type, raise_if_incorrect_type: bool = False) -> Any:
        if raise_if_incorrect_type and not isinstance(self.final_output, cls):
            raise TypeError(f"final output is {type(self.final_output).__name__}, not {cls.__name__}")
        return self.final_output

    def to_input_list(self) -> list[dict[str, Any]]:
        """This run's input followed by what it added, ready to pass as the next run's input."""
        base = [{"role": "user", "content": self.input}] if isinstance(self.input, str) else list(self.input)
        return base + messages_from(self.trajectory)
