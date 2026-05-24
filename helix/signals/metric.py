"""MetricSignal: deterministic observations over a Trajectory.

Reads counters off a Trajectory and emits a GapMeasurement. Four metrics are
supported out of the box:

  - "tokens": sum of step.payload["usage"]["total_tokens"] across all steps
  - "latency_sec": (trajectory.ended_at - trajectory.started_at).total_seconds()
  - "tool_calls": count of steps where kind == TOOL_CALL
  - "model_calls": count of steps where kind == MODEL_CALL

The raw observation goes into GapMeasurement.raw_value; SignalThreshold
normalizes it into score in [0,1] and decides whether `triggered` fires.
This is what lets a CompositeSignal mix a judge (already in [0,1]) with a
cost metric without either dominating by raw magnitude. SPEC §3.3, §3.6.
"""

from __future__ import annotations

from typing import Any, Literal

from helix.artifact import Artifact
from helix.signal import (
    Cost,
    GapMeasurement,
    Preference,
    SignalKind,
    SignalThreshold,
    derive_signal_id,
)
from helix.trajectory import StepKind, Trajectory


MetricName = Literal["tokens", "latency_sec", "tool_calls", "model_calls"]


def _extract_tokens(trajectory: Trajectory) -> float:
    total = 0
    for step in trajectory.steps:
        usage = (step.payload or {}).get("usage") or {}
        if isinstance(usage, dict):
            total += int(usage.get("total_tokens", 0) or 0)
    return float(total)


def _extract_latency(trajectory: Trajectory) -> float:
    if trajectory.ended_at is None:
        return 0.0
    delta = trajectory.ended_at - trajectory.started_at
    return float(delta.total_seconds())


def _extract_tool_calls(trajectory: Trajectory) -> float:
    return float(sum(1 for s in trajectory.steps if s.kind == StepKind.TOOL_CALL))


def _extract_model_calls(trajectory: Trajectory) -> float:
    return float(sum(1 for s in trajectory.steps if s.kind == StepKind.MODEL_CALL))


_EXTRACTORS = {
    "tokens": _extract_tokens,
    "latency_sec": _extract_latency,
    "tool_calls": _extract_tool_calls,
    "model_calls": _extract_model_calls,
}


class MetricSignal:
    """Signal that observes a single Trajectory counter and normalizes it.

    Usage:
        token_signal = MetricSignal(
            metric="tokens",
            threshold=SignalThreshold(
                baseline=2000,
                threshold=500,
                direction="minimize",
                normalizer="ratio",
            ),
        )
        m = await token_signal.measure(candidate, trajectory=t)
        # m.raw_value is the token count; m.score is in [0,1] (higher better);
        # m.triggered is True if tokens > 2500.

    Combine with a judge in a CompositeSignal to balance quality and cost.
    """

    def __init__(
        self,
        metric: MetricName,
        threshold: SignalThreshold,
        version: int = 1,
    ) -> None:
        if metric not in _EXTRACTORS:
            raise ValueError(
                f"unknown metric {metric!r}; expected one of {list(_EXTRACTORS)}"
            )
        self.metric = metric
        self.threshold = threshold
        self._version = version

    @property
    def kind(self) -> SignalKind:
        return SignalKind.METRIC

    @property
    def cost_estimate(self) -> Cost:
        # Pure in-memory scan over trajectory steps; effectively free.
        return Cost(tokens=0, dollars=0.0, wall_clock_sec=0.001)

    @property
    def signal_id(self) -> str:
        return derive_signal_id(
            "MetricSignal",
            {
                "metric": self.metric,
                "baseline": self.threshold.baseline if not callable(self.threshold.baseline) else "callable",
                "threshold": self.threshold.threshold,
                "direction": self.threshold.direction,
                "normalizer": self.threshold.normalizer,
            },
        )

    @property
    def signal_version(self) -> int:
        return self._version

    async def measure(
        self,
        candidate: Artifact,
        trajectory: Trajectory | None = None,
        reference: Artifact | None = None,  # unused
        ground_truth: Any | None = None,    # unused
    ) -> GapMeasurement:
        if trajectory is None:
            return GapMeasurement(
                score=None,
                preference=Preference.NONE,
                feedback=f"metric {self.metric}: no trajectory provided",
                confidence=0.0,
                signal_id=self.signal_id,
                signal_version=self.signal_version,
                triggered=False,
                raw_value=None,
                cost=Cost(),
                metadata={"signal_kind": "metric", "metric": self.metric, "empty": True},
            )

        raw = _EXTRACTORS[self.metric](trajectory)
        score = self.threshold.normalize(raw)
        triggered = self.threshold.is_triggered(raw)

        return GapMeasurement(
            score=score,
            preference=Preference.NONE,
            feedback=(
                f"metric {self.metric}={raw:.3f} (score={score:.3f}, "
                f"triggered={triggered})"
            ),
            confidence=1.0,
            signal_id=self.signal_id,
            signal_version=self.signal_version,
            triggered=triggered,
            raw_value=raw,
            cost=self.cost_estimate,
            metadata={
                "signal_kind": "metric",
                "metric": self.metric,
                "trajectory_id": trajectory.id,
                "baseline": self.threshold.resolve_baseline(),
                "threshold": self.threshold.threshold,
                "direction": self.threshold.direction,
                "normalizer": self.threshold.normalizer,
            },
        )
