"""ImplicitFeedbackSignal: turn user reactions into a measurement.

Reads from the FeedbackStore and produces a GapMeasurement for a trajectory
based on what users actually did with the agent's answer. Combines:

  - Explicit thumbs (most direct, when available)
  - Outcome at session end (resolved / escalated / abandoned)
  - Follow-up timing (short follow-ups indicate the answer didn't resolve)
  - Copy / regenerate actions
  - Free-text comments (signal that the user cared enough to say something)

Each kind contributes a per-kind mean to the final score; weights default
to a sensible "thumbs matter most, follow-ups matter least" ordering.

Failure mode: Skalse 2022 reward hacking. The agent may learn to produce
satisfying-feeling answers rather than correct ones. Mitigation is to
combine this with GoldenSetCalibrator and LiveTrajectoryJudge in a
CompositeSignal so satisfaction signal doesn't dominate alone.

When no feedback exists for a trajectory, the signal returns confidence=0.0
so a CompositeSignal can weight it appropriately. This is correct behavior:
silence isn't a measurement.
"""

from __future__ import annotations

from typing import Any

from helix.artifact import Artifact
from helix.feedback import FeedbackKind, FeedbackStore, get_feedback_store
from helix.signal import (
    Cost,
    GapMeasurement,
    Preference,
    SignalKind,
    derive_signal_id,
)
from helix.trajectory import Trajectory


# Per-kind weights. Sum doesn't need to be 1; the signal normalizes.
DEFAULT_KIND_WEIGHTS: dict[str, float] = {
    FeedbackKind.THUMBS.value: 1.0,
    FeedbackKind.OUTCOME.value: 0.8,
    FeedbackKind.COPY.value: 0.5,
    FeedbackKind.REGENERATE.value: 0.7,
    FeedbackKind.FOLLOWUP.value: 0.3,
    FeedbackKind.COMMENT.value: 0.4,
    FeedbackKind.EDIT.value: 0.5,
}


def _map_value_to_score(value: float) -> float:
    """Feedback values live in [-1, 1]; scores live in [0, 1]."""
    return max(0.0, min(1.0, 0.5 + value / 2.0))


class ImplicitFeedbackSignal:
    """Aggregates user reactions for one trajectory into a GapMeasurement.

    Usage in a CompositeSignal:
        signal = CompositeSignal(
            signals=[
                LiveTrajectoryJudge(weight=0.5),
                ImplicitFeedbackSignal(weight=0.3),
                GoldenSetCalibrator(weight=0.2),
            ],
            aggregator="weighted_mean",
        )
    """

    def __init__(
        self,
        store: FeedbackStore | None = None,
        kind_weights: dict[str, float] | None = None,
        min_records_for_confidence: int = 3,
        version: int = 1,
    ) -> None:
        self.store = store or get_feedback_store()
        self.kind_weights = kind_weights or dict(DEFAULT_KIND_WEIGHTS)
        # Below this count, confidence is scaled down linearly so a single
        # thumbs-up doesn't get treated as a strong signal.
        self.min_records_for_confidence = min_records_for_confidence
        self._version = version

    @property
    def kind(self) -> SignalKind:
        # Implicit feedback is environment-derived reward, not ground truth.
        # Mislabeling it GROUND_TRUTH inflated its verifiability ceiling in the
        # eval stack composition (SPEC §3.4); ENVIRONMENT_REWARD is accurate.
        return SignalKind.ENVIRONMENT_REWARD

    @property
    def cost_estimate(self) -> Cost:
        # Pure SQL read; negligible cost.
        return Cost(tokens=0, dollars=0.0, wall_clock_sec=0.05)

    @property
    def signal_id(self) -> str:
        return derive_signal_id(
            "ImplicitFeedbackSignal",
            {
                "kind_weights": self.kind_weights,
                "min_records_for_confidence": self.min_records_for_confidence,
            },
        )

    @property
    def signal_version(self) -> int:
        return self._version

    async def measure(
        self,
        candidate: Artifact,
        trajectory: Trajectory | None = None,
        reference: Artifact | None = None,
        ground_truth: Any | None = None,
    ) -> GapMeasurement:
        if trajectory is None:
            return _empty_measurement("trajectory not provided")

        agg = await self.store.aggregate_for_trajectory(trajectory.id)
        n = agg["n"]
        if n == 0:
            return _empty_measurement(
                f"no feedback recorded for trajectory {trajectory.id[:8]}"
            )

        # Per-kind weighted mean (normalized weights)
        by_kind = agg["by_kind"]
        weighted_sum = 0.0
        weight_total = 0.0
        contributing_kinds: list[str] = []
        for kind_value, mean_value in by_kind.items():
            w = self.kind_weights.get(kind_value, 0.0)
            if w <= 0:
                continue
            weighted_sum += w * _map_value_to_score(mean_value)
            weight_total += w
            contributing_kinds.append(kind_value)

        score = weighted_sum / weight_total if weight_total > 0 else 0.5

        # Confidence ramps from 0 to 1 as record count climbs to the
        # min_records_for_confidence threshold.
        confidence = min(1.0, n / max(1, self.min_records_for_confidence))

        feedback_str = (
            f"implicit feedback aggregate over {n} record(s): "
            f"score={score:.3f}; kinds contributing: {sorted(contributing_kinds)}"
        )

        return GapMeasurement(
            score=score,
            preference=Preference.NONE,
            feedback=feedback_str,
            confidence=confidence,
            signal_id=self.signal_id,
            signal_version=self.signal_version,
            cost=self.cost_estimate,
            metadata={
                "signal_kind": "implicit_feedback",
                "n_records": n,
                "by_kind": by_kind,
                "trajectory_id": trajectory.id,
            },
        )


def _empty_measurement(reason: str) -> GapMeasurement:
    return GapMeasurement(
        score=None,
        preference=Preference.NONE,
        feedback=reason,
        confidence=0.0,
        cost=Cost(),
        metadata={"signal_kind": "implicit_feedback", "empty": True, "reason": reason},
    )
