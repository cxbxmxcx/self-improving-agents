"""The reasoning floor: the lowest reasoning setting that solves a task class.

Chapter 6 measures metacognition by how much reasoning a task class still
needs. An effort sweep runs the same tasks at each reasoning setting and
reports, per setting, how many were solved and what the runs spent. The
floor is the lowest setting whose success meets a threshold fixed before the
runs; a learned memory works when it lowers the floor.

Two thresholds are reported together. A setting is adequate when it solves
at least seven tasks in eight and solid when it solves every task, the
chapter's goal. Resource averages include failed runs, because a cheap
failure is not a saving.

The setting reaches the model through a PRE_MODEL RequestPatch, so the agent
loop is unchanged and the harness, not the agent, decides how hard to think.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from helix.hooks import RequestPatch
from helix.signal import Cost, GapMeasurement, SignalKind, derive_signal_id
from helix.signals.cost import run_cost
from helix.trajectory import StepKind, Trajectory

# The settings in order of increasing reasoning, as gpt-5.4-mini names them.
EFFORTS: tuple[str, ...] = ("none", "low", "medium", "high")
# Seven of eight: one miss is tolerated, two are not.
ADEQUATE = 7 / 8
SOLID = 1.0


class EffortSetting:
    """A PRE_MODEL hook that holds every model call of a run at one setting."""

    def __init__(self, effort: str, *, param: str = "reasoning_effort") -> None:
        self.effort = effort
        self.param = param

    async def __call__(self, **_: Any) -> RequestPatch:
        return RequestPatch(params={self.param: self.effort}, source=f"effort:{self.effort}")


def reasoning_tokens(trajectory: Trajectory) -> int:
    """Reasoning tokens over every model call in a run."""
    return sum(int((step.payload.get("usage") or {}).get("reasoning_tokens") or 0)
               for step in trajectory.steps if step.kind == StepKind.MODEL_CALL)


@dataclass
class RunRecord:
    """One task at one setting: whether it was solved, why, and what it spent."""

    task_id: str
    effort: str
    solved: bool
    reason: str = ""
    reasoning_tokens: int = 0
    tokens: int = 0
    dollars: float = 0.0
    seconds: float = 0.0
    model_calls: int = 0

    @classmethod
    def of(cls, task_id: str, effort: str, solved: bool, trajectory: Trajectory | None, *,
           reason: str = "", prices: dict[str, tuple[float, ...]] | None = None) -> RunRecord:
        if trajectory is None:
            return cls(task_id, effort, solved, reason)
        cost = run_cost(trajectory, prices)
        calls = sum(1 for s in trajectory.steps if s.kind == StepKind.MODEL_CALL)
        return cls(task_id, effort, solved, reason, reasoning_tokens(trajectory), cost.tokens,
                   round(cost.dollars, 6), round(cost.seconds, 4), calls)


@dataclass
class EffortCell:
    """Every task at one setting."""

    effort: str
    runs: list[RunRecord] = field(default_factory=list)

    @property
    def solved(self) -> int:
        return sum(r.solved for r in self.runs)

    @property
    def total(self) -> int:
        return len(self.runs)

    @property
    def rate(self) -> float:
        return self.solved / self.total if self.total else 0.0

    def mean(self, attribute: str) -> float:
        return sum(getattr(r, attribute) for r in self.runs) / self.total if self.total else 0.0


@dataclass
class EffortSweep:
    """One arm (a memory, or none) measured at every setting."""

    arm: str
    cells: dict[str, EffortCell] = field(default_factory=dict)

    def floor(self, threshold: float = ADEQUATE) -> str | None:
        """The lowest setting meeting the threshold, or None when none does."""
        for effort in EFFORTS:
            cell = self.cells.get(effort)
            if cell is not None and cell.total and cell.rate >= threshold - 1e-9:
                return effort
        return None

    def rows(self) -> list[dict[str, Any]]:
        return [{"arm": self.arm, "effort": e, "solved": c.solved, "total": c.total,
                 "reasoning_tokens": round(c.mean("reasoning_tokens"), 1),
                 "dollars": round(c.mean("dollars"), 6), "seconds": round(c.mean("seconds"), 3)}
                for e, c in ((e, self.cells[e]) for e in EFFORTS if e in self.cells)]

    def to_dict(self) -> dict[str, Any]:
        return {"arm": self.arm, "adequate_floor": self.floor(ADEQUATE), "solid_floor": self.floor(SOLID),
                "cells": {e: [r.__dict__ for r in c.runs] for e, c in self.cells.items()}}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EffortSweep:
        sweep = cls(data["arm"])
        for effort, runs in data["cells"].items():
            sweep.cells[effort] = EffortCell(effort, [RunRecord(**r) for r in runs])
        return sweep


# One task at one setting -> (solved, reason, trajectory).
RunTask = Callable[[Any, str], Awaitable[tuple[bool, str, Trajectory | None]]]


async def sweep(arm: str, tasks: Sequence[Any], run: RunTask, *, efforts: Sequence[str] = EFFORTS,
                task_id: Callable[[Any], str] = lambda t: str(t["id"]), concurrency: int = 4,
                prices: dict[str, tuple[float, ...]] | None = None) -> EffortSweep:
    """Run every task at every setting and collect the cells.

    Tasks run concurrently up to `concurrency`; each run gets its own
    environment from `run`, so order never changes a result.
    """
    unknown = [e for e in efforts if e not in EFFORTS]
    if unknown:
        raise ValueError(f"unknown reasoning settings: {unknown}")
    gate = asyncio.Semaphore(max(1, concurrency))
    result = EffortSweep(arm, {e: EffortCell(e) for e in efforts})

    async def one(task: Any, effort: str) -> RunRecord:
        async with gate:
            solved, reason, trajectory = await run(task, effort)
        return RunRecord.of(task_id(task), effort, solved, trajectory, reason=reason, prices=prices)

    records = await asyncio.gather(*(one(t, e) for e in efforts for t in tasks))
    for record in records:
        result.cells[record.effort].runs.append(record)
    for cell in result.cells.values():
        cell.runs.sort(key=lambda r: r.task_id)
    return result


class ReasoningFloorSignal:
    """The floor a memory reaches on a task class, as a measurement. SPEC §3.

    The candidate is the memory (a skill artifact, or the bare prompt for the
    no-memory arm) and the ground truth is its effort sweep on the class. The
    gap is how many settings the floor sits above the target: zero at or
    below the target, one per setting above it, and the full range when no
    setting meets the threshold. This is the signal that decides whether a
    skill applies to a class.
    """

    def __init__(self, *, target: str = "low", threshold: float = ADEQUATE, version: int = 1) -> None:
        if target not in EFFORTS:
            raise ValueError(f"target must be one of {EFFORTS}")
        self.target = target
        self.threshold = threshold
        self._version = version

    @property
    def kind(self) -> SignalKind:
        return SignalKind.METRIC

    @property
    def cost_estimate(self) -> Cost:
        return Cost(wall_clock_sec=0.001)

    @property
    def signal_id(self) -> str:
        return derive_signal_id("ReasoningFloorSignal", {"target": self.target, "threshold": self.threshold})

    @property
    def signal_version(self) -> int:
        return self._version

    async def measure(self, candidate: Any, trajectory: Trajectory | None = None,
                      reference: Any | None = None, ground_truth: Any | None = None) -> GapMeasurement:
        if not isinstance(ground_truth, EffortSweep):
            return GapMeasurement(confidence=0.0, signal_id=self.signal_id, signal_version=self.signal_version,
                                  feedback="reasoning floor: no effort sweep to measure", metadata={"empty": True})
        return self.measure_sweep(ground_truth)

    def measure_sweep(self, result: EffortSweep) -> GapMeasurement:
        floor = result.floor(self.threshold)
        span = len(EFFORTS) - EFFORTS.index(self.target)
        steps = span if floor is None else max(0, EFFORTS.index(floor) - EFFORTS.index(self.target))
        return GapMeasurement(
            score=1.0 - steps / span,
            signal_id=self.signal_id,
            signal_version=self.signal_version,
            triggered=steps > 0,
            raw_value=float(steps),
            feedback=(f"{result.arm}: floor {floor or 'not reached'} "
                      f"(target {self.target}, threshold {self.threshold:.3f})"),
            metadata={"arm": result.arm, "floor": floor, "target": self.target,
                      "threshold": self.threshold, "solid_floor": result.floor(SOLID),
                      "rows": result.rows()},
        )
