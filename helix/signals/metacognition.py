"""MetacognitionObjective: outcome, safety, dollars, and seconds in one measurement.

Metacognition is worth keeping only if it holds the outcome up, never acts
unsafely, and costs clearly less than deliberating everywhere. This signal
folds the four into one score for searches that need a number, and keeps
them apart in the metadata for searches that rank on several objectives at
once (chapter 6's Pareto selection).

Safety is a filter, not a trade: an unsafe run scores 0 whatever its outcome
and triggers the measurement. Cost enters the score only at the price the
weights set, which is a judgment about what a dollar and a second are worth.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from helix.artifact import Artifact
from helix.signal import Cost, GapMeasurement, Preference, Signal, SignalKind, derive_signal_id
from helix.signals.cost import CostWeights, RunCost, run_cost
from helix.trajectory import Trajectory

# How a multi-objective selection reads `metadata["objectives"]`; "safe" filters.
OBJECTIVES = {"outcome": "maximize", "dollars": "minimize", "seconds": "minimize"}


def objective_score(objectives: dict, weights: CostWeights) -> float | None:
    """The one-number score of a run's objectives: 0 when unsafe, nothing while
    the reward is pending, else the outcome less the priced cost."""
    if objectives.get("safe") is False:
        return 0.0
    if objectives.get("outcome") is None:
        return None
    penalty = weights.penalty(objectives.get("dollars", 0.0), objectives.get("seconds", 0.0))
    return max(0.0, min(1.0, objectives["outcome"] - penalty))


class MetacognitionObjective:
    """One measurement per run over outcome, safety, dollars, and seconds. SPEC §3.5."""

    def __init__(self, outcome: Signal, safety: Signal, *, weights: CostWeights | None = None,
                 prices: dict[str, tuple[float, float]] | None = None, version: int = 1) -> None:
        self.outcome = outcome
        self.safety = safety
        self.weights = weights or CostWeights()
        self.prices = dict(prices or {})
        self._version = version

    @property
    def kind(self) -> SignalKind:
        return SignalKind.COMPOSITE

    @property
    def cost_estimate(self) -> Cost:
        return self.outcome.cost_estimate + self.safety.cost_estimate

    @property
    def signal_id(self) -> str:
        return derive_signal_id("MetacognitionObjective", {
            "outcome": self.outcome.signal_id, "safety": self.safety.signal_id,
            "weights": asdict(self.weights), "prices": sorted(self.prices.items()),
        })

    @property
    def signal_version(self) -> int:
        return self._version

    async def measure(self, candidate: Artifact, trajectory: Trajectory | None = None,
                      reference: Artifact | None = None, ground_truth: Any | None = None) -> GapMeasurement:
        outcome = await self.outcome.measure(candidate, trajectory, reference, ground_truth)
        safety = await self.safety.measure(candidate, trajectory, reference, ground_truth)
        cost = run_cost(trajectory, self.prices) if trajectory is not None else RunCost()
        unsafe = safety.triggered
        objectives = {"outcome": outcome.score, "safe": not unsafe, "dollars": cost.dollars, "seconds": cost.seconds}
        score = objective_score(objectives, self.weights)
        # No reward yet: report the costs, but do not guess a score.
        confidence = 1.0 if unsafe else (0.0 if score is None else outcome.confidence)
        reward = "no reward yet" if outcome.score is None else f"outcome {outcome.score:.2f}"
        return GapMeasurement(
            score=score,
            preference=Preference.NONE,
            feedback=f"{reward}, {'UNSAFE' if unsafe else 'safe'}, ${cost.dollars:.6f}, {cost.seconds:.2f} s"
                     + (f"; {safety.feedback}" if unsafe else ""),
            confidence=confidence,
            signal_id=self.signal_id,
            signal_version=self.signal_version,
            triggered=unsafe,
            raw_value=score,
            cost=outcome.cost + safety.cost,
            metadata={"objectives": objectives, "weights": asdict(self.weights), "run_cost": cost.to_dict(),
                      "components": [outcome.to_dict(), safety.to_dict()]},
        )
