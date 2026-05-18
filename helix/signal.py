"""The Signal primitive.

A Signal is anything that returns a GapMeasurement. Every form of measurement
(ground-truth eval, LLM-as-Judge absolute or pairwise, ContrastiveJudge,
Process Reward Models, reflection, formal proof) is a Signal. SPEC section 3.

The GapMeasurement is intentionally a union shape rather than a polymorphic
hierarchy: different signal families fill different fields. A pairwise signal
fills `preference`; an absolute signal fills `score`; a reflection signal fills
`feedback`. This keeps the consumption side (Search methods) uniform.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable

from helix.artifact import Artifact, ParentRef
from helix.trajectory import Trajectory


class SignalKind(str, Enum):
    """Signal families per SPEC section 3.3."""
    GROUND_TRUTH = "ground_truth"
    LLM_JUDGE_ABSOLUTE = "llm_judge_absolute"
    LLM_JUDGE_PAIRWISE = "llm_judge_pairwise"
    CONTRASTIVE = "contrastive"
    PROCESS_REWARD_MODEL = "process_reward_model"
    REFLECTION = "reflection"
    FORMAL_PROOF = "formal_proof"
    COMPOSITE = "composite"


class Preference(str, Enum):
    """Pairwise verdict. NONE means the signal is not a pairwise signal."""
    LEFT = "left"
    RIGHT = "right"
    TIE = "tie"
    NONE = "none"


@dataclass
class Cost:
    """Cost estimate per call. Used by SearchBudget to enforce ceilings."""
    tokens: int = 0
    dollars: float = 0.0
    wall_clock_sec: float = 0.0

    def __add__(self, other: Cost) -> Cost:
        return Cost(
            tokens=self.tokens + other.tokens,
            dollars=self.dollars + other.dollars,
            wall_clock_sec=self.wall_clock_sec + other.wall_clock_sec,
        )


@dataclass
class GapMeasurement:
    """The union return type of every Signal. SPEC section 3.1.

    Different signal families fill different fields. The consumption side
    branches on which fields are populated, not on the runtime type.

    - Absolute signals fill `score` in [0, 1].
    - Pairwise signals fill `preference`.
    - Reflective signals fill `feedback`.
    - Contrastive signals fill both `preference` and a differential `feedback`.
    - PRMs fill `score` (aggregate) and a per-step array in `metadata`.
    - Formal-proof signals fill `score` as 1.0 or 0.0.
    """

    score: float | None = None
    preference: Preference = Preference.NONE
    feedback: str | None = None
    confidence: float = 1.0
    rubric_id: ParentRef = None
    cost: Cost = field(default_factory=Cost)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "preference": self.preference.value,
            "feedback": self.feedback,
            "confidence": self.confidence,
            "rubric_id": list(self.rubric_id) if self.rubric_id else None,
            "cost": {
                "tokens": self.cost.tokens,
                "dollars": self.cost.dollars,
                "wall_clock_sec": self.cost.wall_clock_sec,
            },
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GapMeasurement:
        cost_data = data.get("cost", {})
        return cls(
            score=data.get("score"),
            preference=Preference(data.get("preference", "none")),
            feedback=data.get("feedback"),
            confidence=data.get("confidence", 1.0),
            rubric_id=tuple(data["rubric_id"]) if data.get("rubric_id") else None,
            cost=Cost(
                tokens=cost_data.get("tokens", 0),
                dollars=cost_data.get("dollars", 0.0),
                wall_clock_sec=cost_data.get("wall_clock_sec", 0.0),
            ),
            metadata=data.get("metadata", {}),
        )


@runtime_checkable
class Signal(Protocol):
    """The Signal protocol. SPEC section 3.2.

    Implementations accept a candidate Artifact and three optional pieces of
    context. Different signal kinds use different subsets:

      - Ground-truth: candidate, ground_truth.
      - Absolute LLM judges: candidate, trajectory.
      - Pairwise LLM judges: candidate, reference, trajectory.
      - Environment reward: trajectory only.
      - PRM: trajectory only; fills per-step metadata.
      - Reflection: trajectory only; fills `feedback`.
      - ContrastiveJudge: candidate, reference, trajectory; differential feedback.
    """

    @property
    def kind(self) -> SignalKind: ...

    @property
    def cost_estimate(self) -> Cost: ...

    async def measure(
        self,
        candidate: Artifact,
        trajectory: Trajectory | None = None,
        reference: Artifact | None = None,
        ground_truth: Any | None = None,
    ) -> GapMeasurement: ...


# ---------------------------------------------------------------------------
# CompositeSignal: aggregate multiple Signals into one. SPEC section 3.5.
# ---------------------------------------------------------------------------

AggregatorName = str  # "mean" | "weighted_mean" | "conservative_min" | "judge_of_judges"


class CompositeSignal:
    """Combine multiple Signals into a single GapMeasurement.

    Aggregator strategies:
      - "mean": arithmetic mean of scores. Ties broken by majority preference.
      - "weighted_mean": same, with per-signal weights.
      - "conservative_min": min of scores (worst-case). Used when any failure
        should veto a candidate. SPEC section 3.5.
      - "judge_of_judges": route to a meta-Signal that scores the others'
        outputs. v1 implementation: pick the highest-confidence one.

    Aggregating preferences uses majority vote (with confidence as tiebreaker).
    """

    def __init__(
        self,
        signals: list[Signal],
        aggregator: AggregatorName = "mean",
        weights: list[float] | None = None,
    ) -> None:
        if not signals:
            raise ValueError("CompositeSignal requires at least one signal")
        if weights is not None and len(weights) != len(signals):
            raise ValueError("weights must match signals length")
        self.signals = signals
        self.aggregator = aggregator
        self.weights = weights or [1.0] * len(signals)

    @property
    def kind(self) -> SignalKind:
        return SignalKind.COMPOSITE

    @property
    def cost_estimate(self) -> Cost:
        total = Cost()
        for s in self.signals:
            total = total + s.cost_estimate
        return total

    async def measure(
        self,
        candidate: Artifact,
        trajectory: Trajectory | None = None,
        reference: Artifact | None = None,
        ground_truth: Any | None = None,
    ) -> GapMeasurement:
        measurements = []
        total_cost = Cost()
        for s in self.signals:
            m = await s.measure(candidate, trajectory, reference, ground_truth)
            measurements.append(m)
            total_cost = total_cost + m.cost

        score = self._aggregate_score(measurements)
        preference = self._aggregate_preference(measurements)
        feedback = self._aggregate_feedback(measurements)
        confidence = self._aggregate_confidence(measurements)

        return GapMeasurement(
            score=score,
            preference=preference,
            feedback=feedback,
            confidence=confidence,
            cost=total_cost,
            metadata={"components": [m.to_dict() for m in measurements]},
        )

    def _aggregate_score(self, ms: list[GapMeasurement]) -> float | None:
        scored = [(m.score, w) for m, w in zip(ms, self.weights) if m.score is not None]
        if not scored:
            return None
        if self.aggregator == "conservative_min":
            return min(s for s, _ in scored)
        if self.aggregator == "judge_of_judges":
            # pick highest-confidence component's score
            picks = [(m.score, m.confidence) for m in ms if m.score is not None]
            return max(picks, key=lambda p: p[1])[0] if picks else None
        total_w = sum(w for _, w in scored)
        return sum(s * w for s, w in scored) / total_w if total_w else None

    def _aggregate_preference(self, ms: list[GapMeasurement]) -> Preference:
        votes = [m.preference for m in ms if m.preference != Preference.NONE]
        if not votes:
            return Preference.NONE
        tally: dict[Preference, int] = {}
        for v in votes:
            tally[v] = tally.get(v, 0) + 1
        ranked = sorted(tally.items(), key=lambda kv: kv[1], reverse=True)
        if len(ranked) == 1 or ranked[0][1] > ranked[1][1]:
            return ranked[0][0]
        return Preference.TIE

    def _aggregate_feedback(self, ms: list[GapMeasurement]) -> str | None:
        parts = [m.feedback for m in ms if m.feedback]
        if not parts:
            return None
        return "\n\n---\n\n".join(parts)

    def _aggregate_confidence(self, ms: list[GapMeasurement]) -> float:
        if not ms:
            return 0.0
        return sum(m.confidence for m in ms) / len(ms)
