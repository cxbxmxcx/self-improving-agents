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
    METRIC = "metric"
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


# ---------------------------------------------------------------------------
# SignalThreshold: normalize unbounded observations into [0,1] and fire a
# triggered flag when the gap crosses threshold. SPEC §3.6.
# ---------------------------------------------------------------------------

from collections.abc import Callable as _Callable  # noqa: E402
from typing import Literal as _Literal  # noqa: E402


@dataclass
class SignalThreshold:
    """Embedded by signals that observe an unbounded quantity (tokens, latency,
    error rate) so the resulting score is in [0,1] and composes with judge
    signals in a CompositeSignal.

    - `baseline` is the expected value. A float fixes it; a zero-arg callable
      resolves it at measure time (e.g. rolling p50 of recent trajectories).
      None means no baseline; normalize() falls back to identity-bounded.
    - `threshold` is the magnitude of |raw - baseline| that fires `triggered`.
      None means the signal never triggers.
    - `direction` is "minimize" (lower raw is better; latency, tokens) or
      "maximize" (higher raw is better; judge scores, hit rates).
    - `normalizer` picks the squashing function: `minmax` against a configured
      range, `zscore` against baseline plus scale, `ratio` (raw / baseline),
      `identity` (clip raw into [0,1] as-is).
    """

    baseline: float | _Callable[[], float] | None = None
    threshold: float | None = None
    direction: _Literal["minimize", "maximize"] = "minimize"
    normalizer: _Literal["minmax", "zscore", "ratio", "identity"] = "ratio"
    # minmax bounds
    min_value: float | None = None
    max_value: float | None = None
    # zscore scale (one standard deviation)
    scale: float | None = None

    def resolve_baseline(self) -> float | None:
        b = self.baseline
        if callable(b):
            return float(b())
        if b is None:
            return None
        return float(b)

    def normalize(self, raw: float) -> float:
        """Map raw → [0,1] under `direction` and `normalizer`. Higher score is
        always better. Out-of-range values clip; NaN-like inputs return 0.5.

        Anchor: when raw == baseline, score is 0.5. Direction shapes the ramp.
        """
        try:
            r = float(raw)
        except (TypeError, ValueError):
            return 0.5

        if self.normalizer == "minmax":
            lo = self.min_value if self.min_value is not None else r
            hi = self.max_value if self.max_value is not None else r
            if hi <= lo:
                v = 0.5
            else:
                v = (r - lo) / (hi - lo)
            # Flip for minimize: low raw = high score
            if self.direction == "minimize":
                v = 1.0 - v
        elif self.normalizer == "zscore":
            b = self.resolve_baseline()
            s = self.scale if (self.scale is not None and self.scale > 0) else 1.0
            if b is None:
                v = 0.5
            else:
                # Map z = (r - b) / s into [0,1] via clipped linear sigmoid
                z = (r - b) / s
                v = 0.5 + 0.5 * max(-1.0, min(1.0, z / 3.0))
            if self.direction == "minimize":
                v = 1.0 - v
        elif self.normalizer == "ratio":
            b = self.resolve_baseline()
            if b is None or b == 0:
                v = 0.5
            else:
                # Anchor: raw == baseline → 0.5. For minimize, low raw is good,
                # so use baseline/raw to invert. For maximize, raw/baseline.
                if self.direction == "minimize":
                    if r == 0:
                        v = 1.0  # zero cost is perfect under minimize
                    else:
                        v = (b / r) / 2.0  # raw=baseline → 0.5; raw<baseline → >0.5
                else:
                    v = (r / b) / 2.0  # raw=baseline → 0.5; raw>baseline → >0.5
        else:  # identity
            v = r
            if self.direction == "minimize":
                v = 1.0 - v

        return max(0.0, min(1.0, v))

    def is_triggered(self, raw: float) -> bool:
        """True when |raw - baseline| crosses threshold (in the unfavorable
        direction). With no baseline or no threshold, returns False."""
        if self.threshold is None:
            return False
        b = self.resolve_baseline()
        if b is None:
            return False
        try:
            r = float(raw)
        except (TypeError, ValueError):
            return False
        if self.direction == "minimize":
            return (r - b) > self.threshold
        return (b - r) > self.threshold


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
    - Metric signals fill `raw_value` with the observation and `score` with
      its normalized form; `triggered` is True when the threshold fired.
    """

    score: float | None = None
    preference: Preference = Preference.NONE
    feedback: str | None = None
    confidence: float = 1.0
    rubric_id: ParentRef = None
    signal_id: str | None = None
    signal_version: int | None = None
    triggered: bool = False
    raw_value: float | None = None
    cost: Cost = field(default_factory=Cost)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "preference": self.preference.value,
            "feedback": self.feedback,
            "confidence": self.confidence,
            "rubric_id": list(self.rubric_id) if self.rubric_id else None,
            "signal_id": self.signal_id,
            "signal_version": self.signal_version,
            "triggered": self.triggered,
            "raw_value": self.raw_value,
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
            signal_id=data.get("signal_id"),
            signal_version=data.get("signal_version"),
            triggered=bool(data.get("triggered", False)),
            raw_value=data.get("raw_value"),
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

    `signal_id` is a stable identifier for this signal's configuration. Two
    signals of the same kind with different weights, different child sets, or
    different threshold settings have different ids. `signal_version` is bumped
    when the signal's measurement semantics change. The archive uses both to
    keep measurements attributable across signal evolution.
    """

    @property
    def kind(self) -> SignalKind: ...

    @property
    def cost_estimate(self) -> Cost: ...

    @property
    def signal_id(self) -> str: ...

    @property
    def signal_version(self) -> int: ...

    async def measure(
        self,
        candidate: Artifact,
        trajectory: Trajectory | None = None,
        reference: Artifact | None = None,
        ground_truth: Any | None = None,
    ) -> GapMeasurement: ...


def derive_signal_id(cls_name: str, config: dict[str, Any]) -> str:
    """Default signal_id derivation: class name plus a content hash of key config.

    Signals call this from their `signal_id` property with whatever fields define
    their configuration. Composite signals pass `{"weights": [...], "child_ids": [...]}`;
    judges pass `{"model": "...", "rubric_id": "..."}`. The hash is short and
    deterministic so two signals with identical configs share an id.
    """
    import hashlib
    import json
    payload = json.dumps(config, sort_keys=True, default=str).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()[:12]
    return f"{cls_name}:{digest}"


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
        version: int = 1,
    ) -> None:
        if not signals:
            raise ValueError("CompositeSignal requires at least one signal")
        if weights is not None and len(weights) != len(signals):
            raise ValueError("weights must match signals length")
        self.signals = signals
        self.aggregator = aggregator
        self.weights = weights or [1.0] * len(signals)
        self._version = version

    @property
    def kind(self) -> SignalKind:
        return SignalKind.COMPOSITE

    @property
    def cost_estimate(self) -> Cost:
        total = Cost()
        for s in self.signals:
            total = total + s.cost_estimate
        return total

    @property
    def signal_id(self) -> str:
        # SPEC §3.5: composite id derives from aggregator + weights + child ids
        # so two composites with the same shape share an id and two composites
        # with different weights or different children are distinct.
        child_ids = [getattr(s, "signal_id", type(s).__name__) for s in self.signals]
        return derive_signal_id(
            "CompositeSignal",
            {"aggregator": self.aggregator, "weights": self.weights, "child_ids": child_ids},
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
        triggered = self._aggregate_triggered(measurements)

        component_triggers = [
            {"signal_id": m.signal_id, "triggered": m.triggered} for m in measurements
        ]

        return GapMeasurement(
            score=score,
            preference=preference,
            feedback=feedback,
            confidence=confidence,
            signal_id=self.signal_id,
            signal_version=self.signal_version,
            triggered=triggered,
            cost=total_cost,
            metadata={
                "components": [m.to_dict() for m in measurements],
                "component_triggers": component_triggers,
            },
        )

    def _aggregate_triggered(self, ms: list[GapMeasurement]) -> bool:
        # SPEC §3.5: conservative_min triggers if any child triggered;
        # other aggregators trigger if a weighted majority triggered.
        flags = [m.triggered for m in ms]
        if not any(flags):
            return False
        if self.aggregator == "conservative_min":
            return True
        total_w = sum(self.weights)
        triggered_w = sum(w for f, w in zip(flags, self.weights) if f)
        return triggered_w > total_w / 2.0

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
