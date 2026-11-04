"""OutcomeSignal: the environment's verdict on one run.

Chapter 5's calendar gave a reward for each run: 1 when the booking was
valid and accepted, 0 when it was invalid or rejected, nothing while
feedback was pending. This signal measures that reward as a GapMeasurement,
so search and the online improver can read it like any other signal. The
`reward_fn` knows the task; the signal only records it.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from helix._async import resolve
from helix.artifact import Artifact
from helix.signal import Cost, GapMeasurement, Preference, SignalKind, derive_signal_id
from helix.trajectory import Trajectory

# A reward in [0, 1], or (reward, detail), or None when there is none yet.
RewardFn = Callable[[Trajectory], "float | tuple[float, dict] | None | Awaitable[Any]"]


class OutcomeSignal:
    """The reward a run earned, as an absolute measurement. SPEC §3.3.

    A checker that compares a run with known truth is a ground-truth signal,
    the default. A reward read from a real user's reaction is an environment
    reward; pass `kind=SignalKind.ENVIRONMENT_REWARD` to say so.
    """

    def __init__(self, reward_fn: RewardFn, *, label: str = "outcome",
                 kind: SignalKind = SignalKind.GROUND_TRUTH, version: int = 1) -> None:
        self.reward_fn = reward_fn
        self.label = label
        self._kind = kind
        self._version = version

    @property
    def kind(self) -> SignalKind:
        return self._kind

    @property
    def cost_estimate(self) -> Cost:
        return Cost(wall_clock_sec=0.001)

    @property
    def signal_id(self) -> str:
        return derive_signal_id("OutcomeSignal", {"label": self.label, "kind": self._kind.value})

    @property
    def signal_version(self) -> int:
        return self._version

    async def measure(self, candidate: Artifact, trajectory: Trajectory | None = None,
                      reference: Artifact | None = None, ground_truth: Any | None = None) -> GapMeasurement:
        result = await resolve(self.reward_fn(trajectory)) if trajectory is not None else None
        if result is None:
            # No reward yet is not a zero: pending feedback teaches nothing.
            return GapMeasurement(confidence=0.0, signal_id=self.signal_id, signal_version=self.signal_version,
                                  feedback=f"{self.label}: no reward for this run",
                                  metadata={"label": self.label, "empty": True})
        reward, detail = result if isinstance(result, tuple) else (result, {})
        reward = max(0.0, min(1.0, float(reward)))
        return GapMeasurement(
            score=reward,
            preference=Preference.NONE,
            feedback=detail.get("feedback") or f"{self.label}={reward:.3f}",
            confidence=1.0,
            signal_id=self.signal_id,
            signal_version=self.signal_version,
            raw_value=reward,
            cost=self.cost_estimate,
            metadata={"label": self.label, "trajectory_id": trajectory.id, **detail},
        )
