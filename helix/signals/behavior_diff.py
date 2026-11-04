"""BehaviorDiffSignal: how far one run's behavior moved from another's.

A lesson that changes nothing is reflection theater. The test is to run the
same request before and after the lesson and compare what the agent did: the
route it took, the tool calls it made and whether they ran, and the outcome
it reached. A reworded reply with the same actions is not a change in
behavior, so wording carries no weight by default.

The score is the divergence itself, so a flat diff triggers the signal.
Whether the change was for the better is the outcome signal's question.
Chapter 7 reuses the same comparison on a proposal card (DESIGN_NOTES).
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from difflib import SequenceMatcher
from typing import Any

from helix._async import resolve
from helix.artifact import Artifact
from helix.scaffold.common import META_KEY
from helix.signal import Cost, GapMeasurement, Preference, SignalKind, derive_signal_id
from helix.signals.safety import tool_calls
from helix.trajectory import Trajectory

DEFAULT_WEIGHTS = {"route": 1.0, "tools": 1.0, "outcome": 1.0, "reply": 0.0}

# Finds the run to compare against, when it is not passed in.
BaselineLookup = Callable[[Trajectory], "Trajectory | None | Awaitable[Any]"]
RewardFn = Callable[[Trajectory], "float | None | Awaitable[Any]"]


def _signatures(trajectory: Trajectory) -> list[str]:
    """Each call as name(arguments), marked when it did not run."""
    out = []
    for call in tool_calls(trajectory):
        args = json.dumps(call["arguments"], sort_keys=True, default=str)
        out.append(f"{call['tool']}({args})" + ("" if call["executed"] else " [not run]"))
    return out


def _distance(a: list[str] | str, b: list[str] | str) -> float:
    if not a and not b:
        return 0.0
    return 1.0 - SequenceMatcher(None, a, b).ratio()


class BehaviorDiffSignal:
    """Divergence of a run from a baseline run of the same request. SPEC §3.3."""

    def __init__(self, *, weights: dict[str, float] | None = None, min_divergence: float = 0.05,
                 reward_fn: RewardFn | None = None, baseline: BaselineLookup | None = None,
                 version: int = 1) -> None:
        self.weights = {**DEFAULT_WEIGHTS, **(weights or {})}
        unknown = set(self.weights) - set(DEFAULT_WEIGHTS)
        if unknown:
            raise ValueError(f"unknown behavior components: {sorted(unknown)}")
        self.min_divergence = min_divergence
        self.reward_fn = reward_fn
        self.baseline = baseline
        self._version = version

    @property
    def kind(self) -> SignalKind:
        return SignalKind.METRIC

    @property
    def cost_estimate(self) -> Cost:
        return Cost(wall_clock_sec=0.001)

    @property
    def signal_id(self) -> str:
        return derive_signal_id("BehaviorDiffSignal", {"weights": self.weights, "min": self.min_divergence,
                                                       "reward": self.reward_fn is not None})

    @property
    def signal_version(self) -> int:
        return self._version

    async def measure(self, candidate: Artifact, trajectory: Trajectory | None = None,
                      reference: Artifact | None = None, ground_truth: Any | None = None) -> GapMeasurement:
        """Compare `trajectory` with its baseline: a Trajectory passed as
        `ground_truth` (or under its "baseline" key), else the lookup."""
        before = ground_truth.get("baseline") if isinstance(ground_truth, dict) else ground_truth
        if not isinstance(before, Trajectory) and self.baseline is not None and trajectory is not None:
            before = await resolve(self.baseline(trajectory))
        if trajectory is None or not isinstance(before, Trajectory):
            return GapMeasurement(confidence=0.0, signal_id=self.signal_id, signal_version=self.signal_version,
                                  feedback="behavior diff: no baseline run to compare with",
                                  metadata={"empty": True})
        return await self.compare(before, trajectory)

    async def compare(self, before: Trajectory, after: Trajectory) -> GapMeasurement:
        components: dict[str, float] = {}
        notes: list[str] = []
        routes = [(t.metadata.get(META_KEY) or {}).get("route") for t in (before, after)]
        if any(routes):
            components["route"] = float(routes[0] != routes[1])
            if routes[0] != routes[1]:
                notes.append(f"route {routes[0]} -> {routes[1]}")
        calls = [_signatures(before), _signatures(after)]
        components["tools"] = _distance(*calls)
        if calls[0] != calls[1]:
            notes.append(f"tool calls {len(calls[0])} -> {len(calls[1])}, {components['tools']:.2f} apart")
        rewards = [await resolve(self.reward_fn(t)) if self.reward_fn else None for t in (before, after)]
        if None not in rewards:
            components["outcome"] = abs(float(rewards[0]) - float(rewards[1]))
        else:
            components["outcome"] = float(before.outcome != after.outcome)
        if components["outcome"]:
            shown = rewards if None not in rewards else [before.outcome.value, after.outcome.value]
            notes.append(f"outcome {shown[0]} -> {shown[1]}")
        replies = [" ".join(str(t.final_output or "").lower().split()) for t in (before, after)]
        components["reply"] = _distance(*replies)
        weighted = [(v, self.weights[k]) for k, v in components.items() if self.weights[k] > 0]
        total = sum(w for _, w in weighted)
        divergence = sum(v * w for v, w in weighted) / total if total else 0.0
        flat = divergence < self.min_divergence
        return GapMeasurement(
            score=divergence,
            preference=Preference.NONE,
            feedback="; ".join(notes) if notes else "no change in route, tool calls, or outcome",
            confidence=1.0,
            signal_id=self.signal_id,
            signal_version=self.signal_version,
            triggered=flat,
            raw_value=divergence,
            cost=self.cost_estimate,
            metadata={"before": before.id, "after": after.id, "components": components, "routes": routes,
                      "tool_calls": {"before": calls[0], "after": calls[1]}, "rewards": rewards,
                      "flat": flat},
        )
