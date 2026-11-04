"""What a run cost: dollars, the seconds the user waited, and tokens.

A run's cost comes from three places in its trajectory: the agent's own
model calls (MODEL_CALL steps), the scaffold's System 2 calls (revision and
reflection), and the decision model's calls, which the scaffold records
with their price. `run_cost` sums them, and `CostSignal` turns one of the
totals into a measurement.

Seconds count only the user's wait. Reflection runs after the reply has
gone out, so it adds dollars but not seconds. Replayed runs keep the
seconds recorded live, which is what lets a replay reproduce latency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from helix.artifact import Artifact
from helix.scaffold.common import META_KEY
from helix.signal import Cost, GapMeasurement, Preference, SignalKind, SignalThreshold, derive_signal_id
from helix.trajectory import StepKind, Trajectory

# Work done after the reply is sent; it costs dollars but nobody waits for it.
AFTER_REPLY = frozenset({"reflect"})


@dataclass(frozen=True)
class CostWeights:
    """What a dollar and a second are worth in outcome points, so cost can be
    folded into one score: value = outcome - dollars * a - seconds * b."""

    dollars: float = 0.0
    seconds: float = 0.0

    def penalty(self, dollars: float, seconds: float) -> float:
        return dollars * self.dollars + seconds * self.seconds


@dataclass
class RunCost:
    dollars: float = 0.0
    seconds: float = 0.0
    tokens: int = 0
    wall_clock_sec: float | None = None
    # calls, dollars, seconds, and tokens for "agent", "system2", and "decider"
    by_source: dict[str, dict[str, float]] = field(default_factory=dict)
    # Models whose calls could not be priced; their dollars count as zero.
    unpriced: list[str] = field(default_factory=list)

    def add(self, source: str, *, dollars: float = 0.0, seconds: float = 0.0, tokens: int = 0,
            waited: bool = True) -> None:
        part = self.by_source.setdefault(source, {"calls": 0, "dollars": 0.0, "seconds": 0.0, "tokens": 0})
        part["calls"] += 1
        part["dollars"] += dollars
        part["seconds"] += seconds
        part["tokens"] += tokens
        self.dollars += dollars
        self.tokens += tokens
        if waited:
            self.seconds += seconds

    def to_dict(self) -> dict[str, Any]:
        return {"dollars": self.dollars, "seconds": round(self.seconds, 4), "tokens": self.tokens,
                "wall_clock_sec": self.wall_clock_sec, "by_source": self.by_source, "unpriced": self.unpriced}


def _tokens(usage: Any) -> tuple[int, int]:
    usage = usage if isinstance(usage, dict) else {}
    return int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0)


def call_dollars(model: str, usage: Any, prices: dict[str, tuple[float, ...]] | None = None) -> float | None:
    """The price of one model call, or None when it cannot be priced.

    `prices` maps a model to dollars per million input and output tokens (and,
    optionally, cached input tokens), as
    providers publish them; pinning them keeps a book's numbers stable across
    LiteLLM releases. Other models are priced from LiteLLM's table.
    """
    if not isinstance(usage, dict) or not usage:
        return None
    prompt, completion = _tokens(usage)
    if prices and model in prices:
        # An optional third price is for cached input, which providers bill
        # at a fraction of fresh input; without it every input token is fresh.
        per_input, per_output, *rest = prices[model]
        cached = min(int(usage.get("cached_tokens") or 0), prompt) if rest else 0
        per_cached = rest[0] if rest else per_input
        return ((prompt - cached) * per_input + cached * per_cached + completion * per_output) / 1_000_000
    import litellm

    # Checking the table first keeps unknown names (test doubles, scripted
    # workers) from raising inside LiteLLM.
    if model not in litellm.model_cost and model.split("/", 1)[-1] not in litellm.model_cost:
        return None
    try:
        input_cost, output_cost = litellm.cost_per_token(model=model, prompt_tokens=prompt,
                                                         completion_tokens=completion)
    except Exception:
        return None
    return float(input_cost) + float(output_cost)


def run_cost(trajectory: Trajectory, prices: dict[str, tuple[float, float]] | None = None) -> RunCost:
    """Everything one run cost, by source."""
    cost = RunCost()
    if trajectory.ended_at is not None:
        cost.wall_clock_sec = (trajectory.ended_at - trajectory.started_at).total_seconds()

    def price(source: str, model: str, usage: Any, seconds: float, waited: bool) -> None:
        dollars = call_dollars(model, usage, prices)
        if dollars is None and model not in cost.unpriced:
            cost.unpriced.append(model)
        cost.add(source, dollars=dollars or 0.0, seconds=seconds, tokens=sum(_tokens(usage)), waited=waited)

    for step in trajectory.steps:
        if step.kind == StepKind.MODEL_CALL and "model" in step.payload:
            price("agent", str(step.payload["model"]), step.payload.get("usage"),
                  float(step.payload.get("seconds") or 0.0), True)
    meta = trajectory.metadata.get(META_KEY) or {}
    for call in meta.get("llm_calls", []):
        price("system2", str(call.get("model")), call.get("usage"), float(call.get("seconds") or 0.0),
              call.get("purpose") not in AFTER_REPLY)
    for call in meta.get("decider_calls", []):
        cost.add("decider", dollars=float(call.get("dollars") or 0.0), seconds=float(call.get("seconds") or 0.0),
                 tokens=int(call.get("input_tokens") or 0), waited=call.get("hook") not in AFTER_REPLY)
    return cost


class CostSignal:
    """Dollars or seconds per run, normalized against a baseline. SPEC §3.6.

    `raw_value` holds the observation and `score` its normalized form, higher
    being cheaper or faster; `triggered` fires when the threshold says so.
    """

    def __init__(self, metric: Literal["dollars", "seconds"], threshold: SignalThreshold, *,
                 prices: dict[str, tuple[float, float]] | None = None, version: int = 1) -> None:
        if metric not in ("dollars", "seconds"):
            raise ValueError(f"metric must be 'dollars' or 'seconds', got {metric!r}")
        self.metric = metric
        self.threshold = threshold
        self.prices = dict(prices or {})
        self._version = version

    @property
    def kind(self) -> SignalKind:
        return SignalKind.METRIC

    @property
    def cost_estimate(self) -> Cost:
        return Cost(wall_clock_sec=0.001)

    @property
    def signal_id(self) -> str:
        baseline = self.threshold.baseline
        return derive_signal_id("CostSignal", {
            "metric": self.metric, "baseline": "callable" if callable(baseline) else baseline,
            "threshold": self.threshold.threshold, "normalizer": self.threshold.normalizer,
            "prices": sorted(self.prices.items()),
        })

    @property
    def signal_version(self) -> int:
        return self._version

    async def measure(self, candidate: Artifact, trajectory: Trajectory | None = None,
                      reference: Artifact | None = None, ground_truth: Any | None = None) -> GapMeasurement:
        if trajectory is None:
            return GapMeasurement(confidence=0.0, signal_id=self.signal_id, signal_version=self.signal_version,
                                  feedback=f"{self.metric}: no trajectory to measure", metadata={"empty": True})
        cost = run_cost(trajectory, self.prices)
        raw = cost.dollars if self.metric == "dollars" else cost.seconds
        score = self.threshold.normalize(raw)
        triggered = self.threshold.is_triggered(raw)
        return GapMeasurement(
            score=score,
            preference=Preference.NONE,
            feedback=f"{self.metric}={raw:.6g} (score={score:.3f}, triggered={triggered})",
            # A run with unpriced calls understates its dollars.
            confidence=0.5 if (self.metric == "dollars" and cost.unpriced) else 1.0,
            signal_id=self.signal_id,
            signal_version=self.signal_version,
            triggered=triggered,
            raw_value=raw,
            cost=self.cost_estimate,
            metadata={"metric": self.metric, "trajectory_id": trajectory.id, "run_cost": cost.to_dict()},
        )
