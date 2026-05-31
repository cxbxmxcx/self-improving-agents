"""TaskSuccessSignal: a deterministic ground-truth signal over task scenarios.

Where an LLM-as-judge asks a model "which answer is better," a task agent can be
graded against the world: did it book a flight matching the request, a hotel
under budget, the activities asked for? This signal runs a candidate artifact
against a fixed set of scenarios and averages a deterministic per-scenario score.
It is the ground-truth signal Chapter 3 uses to drive evolutionary search over
tool descriptions, optionally composited with a judge via CompositeSignal (§3.5).

The signal is domain-agnostic. The caller supplies a `score_scenario(candidate,
scenario)` coroutine that knows how to run the candidate on one scenario and
return a [0, 1] score plus a detail dict. The travel agent provides that closure;
the signal just orchestrates, caches per (id, version), and reports.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from helix.artifact import Artifact
from helix.signal import (
    Cost,
    GapMeasurement,
    Preference,
    SignalKind,
    derive_signal_id,
)
from helix.trajectory import Trajectory

# (candidate artifact, scenario) -> (score in [0, 1], detail dict)
ScenarioScorer = Callable[[Artifact, Any], Awaitable[tuple[float, dict]]]


@dataclass
class _CacheEntry:
    score: float
    per_scenario: list[dict[str, Any]]
    measured_at: float


class TaskSuccessSignal:
    """Average deterministic task success over a scenario set. SPEC §3.3, §3.4.

    Construction:
        signal = TaskSuccessSignal(
            scenarios=load_travel_scenarios(...),
            score_scenario=travel_score_scenario,   # builds + runs the agent, checks state
            max_scenarios=8,
            cache_ttl_sec=600,
        )

    measure(candidate) builds nothing itself: it calls score_scenario for each
    scenario, averages the scores, and returns an absolute GapMeasurement with
    kind=GROUND_TRUTH. Results cache per (candidate.id, candidate.version).
    """

    def __init__(
        self,
        scenarios: list[Any],
        score_scenario: ScenarioScorer,
        *,
        max_scenarios: int | None = None,
        cache_ttl_sec: float = 600.0,
        signal_label: str = "task_success",
        version: int = 1,
    ) -> None:
        if not scenarios:
            raise ValueError("TaskSuccessSignal requires at least one scenario")
        self.scenarios = scenarios
        self.score_scenario = score_scenario
        self.max_scenarios = max_scenarios
        self.cache_ttl_sec = cache_ttl_sec
        self.signal_label = signal_label
        self._version = version
        self._cache: dict[tuple[str, int], _CacheEntry] = {}

    @property
    def kind(self) -> SignalKind:
        return SignalKind.GROUND_TRUTH

    @property
    def cost_estimate(self) -> Cost:
        n = len(self._active_scenarios())
        # Each scenario is one multi-tool agent run.
        return Cost(tokens=6000 * n, wall_clock_sec=12.0 * n)

    @property
    def signal_id(self) -> str:
        ids = [getattr(s, "id", str(i)) for i, s in enumerate(self.scenarios)]
        return derive_signal_id(
            "TaskSuccessSignal",
            {"label": self.signal_label, "scenario_ids": ids, "max": self.max_scenarios},
        )

    @property
    def signal_version(self) -> int:
        return self._version

    def _active_scenarios(self) -> list[Any]:
        if self.max_scenarios and self.max_scenarios < len(self.scenarios):
            return self.scenarios[: self.max_scenarios]
        return self.scenarios

    async def measure(
        self,
        candidate: Artifact,
        trajectory: Trajectory | None = None,  # unused: this signal runs its own
        reference: Artifact | None = None,
        ground_truth: Any | None = None,
    ) -> GapMeasurement:
        key = (candidate.id, candidate.version)
        now = time.monotonic()
        cached = self._cache.get(key)
        if cached is not None and (now - cached.measured_at) < self.cache_ttl_sec:
            return self._to_measurement(cached.score, cached.per_scenario, cached=True)

        per_scenario: list[dict[str, Any]] = []
        total = 0.0
        scenarios = self._active_scenarios()
        for sc in scenarios:
            try:
                score, detail = await self.score_scenario(candidate, sc)
            except Exception as exc:  # a failed run scores zero, not crashes the round
                score, detail = 0.0, {"error": f"{type(exc).__name__}: {exc}"}
            score = max(0.0, min(1.0, float(score)))
            total += score
            per_scenario.append({"scenario": getattr(sc, "id", "?"), "score": score, **detail})

        mean = total / len(scenarios) if scenarios else 0.0
        self._cache[key] = _CacheEntry(score=mean, per_scenario=per_scenario, measured_at=now)
        return self._to_measurement(mean, per_scenario, cached=False)

    def _to_measurement(
        self, score: float, per_scenario: list[dict[str, Any]], *, cached: bool
    ) -> GapMeasurement:
        n = len(per_scenario)
        passed = sum(1 for p in per_scenario if p["score"] >= 0.999)
        return GapMeasurement(
            score=score,
            preference=Preference.NONE,
            feedback=f"task success {score:.2f} ({passed}/{n} scenarios fully satisfied)"
            + (" (cached)" if cached else ""),
            confidence=1.0,
            signal_id=self.signal_id,
            signal_version=self.signal_version,
            cost=Cost(),
            metadata={
                "signal_kind": "task_success",
                "cached": cached,
                "n_scenarios": n,
                "n_full_pass": passed,
                "per_scenario": per_scenario,
            },
        )

    def clear_cache(self) -> None:
        self._cache.clear()
