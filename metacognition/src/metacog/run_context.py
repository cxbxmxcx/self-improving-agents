"""The context object every tool, guardrail, and hook receives."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

TContext = TypeVar("TContext")


@dataclass
class Usage:
    """Token counts over a run's model calls. Reasoning and cached tokens are kept apart, since
    reasoning is the work metacognition removes and cached input is priced lower."""

    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    reasoning_tokens: int = 0
    cached_tokens: int = 0

    def add_call(self, usage: dict[str, Any] | None) -> None:
        usage = usage or {}
        self.requests += 1
        self.input_tokens += int(usage.get("prompt_tokens") or 0)
        self.output_tokens += int(usage.get("completion_tokens") or 0)
        self.total_tokens += int(usage.get("total_tokens") or 0)
        self.reasoning_tokens += int(usage.get("reasoning_tokens") or 0)
        self.cached_tokens += int(usage.get("cached_tokens") or 0)


@dataclass
class RunContextWrapper(Generic[TContext]):
    """Your context object for this run, plus what the run has used so far."""

    context: TContext
    usage: Usage = field(default_factory=Usage)
