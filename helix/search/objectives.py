"""Aggregation and Pareto selection over measured objectives.

A candidate is on the front when no other feasible candidate is at least as
good on every objective and better on one. Chapter 6 ranks on outcome,
dollars, and seconds, and treats safety as feasibility: an unsafe candidate
is dropped before any comparison, so it can never trade its way back in.

`aggregate` folds a candidate's per-request measurements into one, keeping
the mean of each objective, so selection compares candidates, not requests.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, TypeVar

from helix.signal import Cost, GapMeasurement

T = TypeVar("T")
Directions = Mapping[str, str]  # objective -> "maximize" | "minimize"


def objectives_of(item: Any) -> dict[str, Any] | None:
    """The objectives a Variant's measurement (or a measurement) carries."""
    measurement = getattr(item, "measurement", item)
    metadata = getattr(measurement, "metadata", None) or {}
    found = metadata.get("objectives")
    return found if isinstance(found, dict) else None


def dominates(a: Mapping[str, float], b: Mapping[str, float], directions: Directions) -> bool:
    """True when `a` is no worse than `b` on every objective and better on one."""
    better = False
    for name, direction in directions.items():
        x, y = a[name], b[name]
        if direction == "minimize":
            x, y = -x, -y
        if x < y:
            return False
        if x > y:
            better = True
    return better


def pareto_front(items: list[T], directions: Directions, *,
                 objectives: Callable[[T], Mapping[str, Any] | None] = objectives_of,
                 feasible: Callable[[Mapping[str, Any]], bool] = lambda o: o.get("safe", True) is not False) -> list[T]:
    """The feasible, non-dominated items, in their original order. Items with
    a missing objective are left out, since they cannot be compared."""
    scored: list[tuple[T, Mapping[str, Any]]] = []
    for item in items:
        found = objectives(item)
        if found is None or not feasible(found) or any(found.get(n) is None for n in directions):
            continue
        scored.append((item, found))
    return [item for i, (item, mine) in enumerate(scored)
            if not any(dominates(other, mine, directions) for j, (_, other) in enumerate(scored) if j != i)]


def pick(front: list[T], directions: Directions, *,
         objectives: Callable[[T], Mapping[str, Any] | None] = objectives_of) -> T | None:
    """One item from a front, by the objectives in the order given: best on the
    first, then the second breaks ties, and so on."""
    def key(item: T) -> tuple[float, ...]:
        found = objectives(item) or {}
        return tuple(found[n] if d == "maximize" else -found[n] for n, d in directions.items())

    return max(front, key=key) if front else None


def aggregate(rows: list[tuple[str, GapMeasurement]], *, signal_id: str | None = None,
              worst: int = 5) -> GapMeasurement:
    """One measurement from per-request absolute measurements.

    The score is the mean over requests that have one. Objectives average
    across requests, except safety, which holds only if every request was
    safe. The feedback quotes the worst-scoring requests, which is what a
    reflective mutation needs to read.
    """
    scored = [(rid, m) for rid, m in rows if m.score is not None]
    mean = sum(m.score for _, m in scored) / len(scored) if scored else None
    objectives: dict[str, Any] = {}
    found = [o for o in (objectives_of(m) for _, m in rows) if o]
    if found:
        for name in sorted({k for o in found for k in o}):
            values = [o[name] for o in found if o.get(name) is not None]
            if name == "safe":
                objectives[name] = all(values)
            elif values and all(isinstance(v, (int, float)) for v in values):
                objectives[name] = sum(values) / len(values)
    low = sorted(scored, key=lambda row: row[1].score)[:worst]
    cost = Cost()
    for _, m in rows:
        cost = cost + m.cost
    return GapMeasurement(
        score=mean,
        feedback="\n".join(f"[{rid}] {m.feedback}" for rid, m in low if m.feedback) or None,
        confidence=len(scored) / len(rows) if rows else 0.0,
        signal_id=signal_id,
        triggered=any(m.triggered for _, m in rows) or objectives.get("safe") is False,
        raw_value=mean,
        cost=cost,
        metadata={"n_questions": len(rows), "objectives": objectives or None,
                  "per_question": [{"question_id": rid, "score": m.score, "triggered": m.triggered}
                                   for rid, m in rows]},
    )
