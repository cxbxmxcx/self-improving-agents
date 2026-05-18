"""EvalSource: how an Improver gets its questions for a round.

Two concrete implementations:

- FixedEvalSet: wraps a static EvalSet. Reproducible, controllable, the Ch 2
  default. Same N questions every round (or a stratified sample if you
  configure it that way).

- RecentTrajectorySource: subscribes to TrajectoryCompleted events on the
  event bus, buffers the last N completed trajectories, and serves them as
  questions for live-traffic improvement. The "candidate vs reference" judge
  compares fresh agent runs against the stored ones. Used in Ch 3+ when the
  Improver learns from production traffic.

Both conform to the EvalSource Protocol so the Improver does not care which
mode is active.
"""

from __future__ import annotations

from collections import deque
from typing import Protocol, runtime_checkable

from helix.eval.dataset import EvalQuestion, EvalSet
from helix.observability.bus import EventBus, get_bus
from helix.observability.events import TrajectoryCompleted


@runtime_checkable
class EvalSource(Protocol):
    """Anything that can produce EvalQuestion lists for an Improver round."""

    async def get_questions(self, n: int | None = None) -> list[EvalQuestion]: ...

    @property
    def label(self) -> str: ...


class FixedEvalSet:
    """EvalSource backed by a fixed, in-memory EvalSet."""

    def __init__(self, eval_set: EvalSet, stratified: bool = True, seed: int | None = None) -> None:
        self.eval_set = eval_set
        self.stratified = stratified
        self.seed = seed

    @property
    def label(self) -> str:
        return f"fixed_eval_set({len(self.eval_set)} questions)"

    async def get_questions(self, n: int | None = None) -> list[EvalQuestion]:
        if n is None or n >= len(self.eval_set):
            return list(self.eval_set.questions)
        return self.eval_set.sample(n=n, stratified=self.stratified, seed=self.seed)


class RecentTrajectorySource:
    """EvalSource backed by a rolling buffer of TrajectoryCompleted events.

    Subscribes to the event bus at construction. Each new TrajectoryCompleted
    pushes onto a bounded deque. get_questions() converts the most recent N
    into EvalQuestion objects using the trajectory's task as the question and
    final_output as a reference proxy.

    This is a thin v1: the reference answer is the past trajectory's own
    output. That is intentionally weak: the Improver using this source is
    asking "would a different prompt have done this same task differently
    and better?" rather than "is this answer correct against ground truth."
    Ch 8 introduces stronger ground-truth sourcing for live traffic.
    """

    def __init__(self, buffer_size: int = 100, bus: EventBus | None = None) -> None:
        self._buffer: deque[EvalQuestion] = deque(maxlen=buffer_size)
        self._bus = bus or get_bus()
        self._bus.subscribe("trajectory_completed", self._on_trajectory)

    @property
    def label(self) -> str:
        return f"recent_trajectory_source(buffer={self._buffer.maxlen}, current={len(self._buffer)})"

    @property
    def size(self) -> int:
        return len(self._buffer)

    def _on_trajectory(self, event: TrajectoryCompleted) -> None:
        if not isinstance(event, TrajectoryCompleted):
            return
        if event.outcome != "completed":
            return
        question = EvalQuestion(
            id=f"live::{event.trajectory_id[:8]}",
            band=0,
            question=event.task,
            reference_answer=event.final_output or "",
            tags=("live", "trajectory"),
        )
        self._buffer.append(question)

    async def get_questions(self, n: int | None = None) -> list[EvalQuestion]:
        items = list(self._buffer)
        if n is None or n >= len(items):
            return items
        return items[-n:]
