"""Attach fixed value-based memory learning to the agent harness.

SESSION_END supplies a completed trajectory to MemoryLearner. The learner
updates values in the same live memory tier that MemoryHooks reads, while
the recall formula, context rendering, and injection timing remain fixed.

The L2 layer rule permits automatic entry updates; it does not make their
contents inherently safe or establish that recall caused the outcome.
The prompt improver remains a separate driver with a different round shape.
"""

from __future__ import annotations

import inspect
import uuid
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from helix.artifact import ArtifactKind, layer_of
from helix.hooks import HookPoint
from helix.observability.bus import EventBus, get_bus
from helix.memory.learning import MemoryLearner
from helix.signal import GapMeasurement, Signal
from helix.trajectory import Trajectory

RewardFn = Callable[[Trajectory], float | None | Awaitable[float | None]]

# The one layer this improver may auto-apply on. Memory entries are L2 by
# the layer rule (SPEC §1.2); anything higher needs a gate this class lacks.
ONLINE_MAX_LAYER = 2


@dataclass
class OnlineMemoryImproverStatus:
    improver_id: str
    target_artifact_id: str
    running: bool
    trajectories_seen: int
    updates_applied: int
    entries_touched: int
    skipped_no_reward: int
    last_reward: float | None
    last_error: str | None


class OnlineMemoryImprover:
    """Harness hook that runs one fixed value update per trajectory.

    Reward resolution, in order: `reward_fn(trajectory)` when given, else
    `outcome.measure(candidate=agent.system_prompt, trajectory=trajectory)`
    whose score is the reward and whose measurement carries feedback and
    confidence through to the value update. A run with no resolvable reward is
    skipped, never guessed.
    """

    def __init__(
        self,
        agent: Any,
        learner: MemoryLearner | None = None,
        outcome: Signal | None = None,
        reward_fn: RewardFn | None = None,
        improver_id: str | None = None,
        bus: EventBus | None = None,
        *,
        search: MemoryLearner | None = None,
    ) -> None:
        if outcome is None and reward_fn is None:
            raise ValueError("OnlineMemoryImprover needs an outcome signal or a reward_fn")
        if learner is not None and search is not None and learner is not search:
            raise ValueError("provide learner or the legacy search argument, not different objects")
        learner = learner if learner is not None else search
        if learner is None:
            raise ValueError("OnlineMemoryImprover needs a memory learner")
        tier_name = learner.tier.tier_name
        if getattr(agent, "memory", {}).get(tier_name) is not learner.tier:
            raise ValueError("memory learner must update the tier installed in the agent harness")
        if layer_of(ArtifactKind.MEMORY_ENTRY) > ONLINE_MAX_LAYER:
            raise ValueError("memory entries are not online-safe under the current layer table")
        self.agent = agent
        self.learner = learner
        self.outcome = outcome
        self.reward_fn = reward_fn
        self.improver_id = improver_id or f"online-memory-{uuid.uuid4().hex[:8]}"
        # The target is a family of artifacts, not one id; the wildcard is
        # what the chat UI and Observatory display for this improver.
        self.target_artifact_id = f"memory.{tier_name}.*"
        self.bus = bus or get_bus()
        self._running = False
        self._subscribed = False
        self._trajectories_seen = 0
        self._updates_applied = 0
        self._entries_touched = 0
        self._skipped_no_reward = 0
        self._last_reward: float | None = None
        self._last_error: str | None = None

    # ---------------- wiring (SPEC §17.1) ----------------

    @property
    def search(self) -> MemoryLearner:
        """Compatibility name for callers of the previous memory-improver API."""
        return self.learner

    def on_attached(self) -> None:
        self._ensure_subscribed()

    def _ensure_subscribed(self) -> None:
        if self._subscribed:
            return
        self.agent.hooks.register(HookPoint.SESSION_END, self._on_session_end)
        self._subscribed = True

    async def start(self) -> None:
        self._ensure_subscribed()
        self._running = True

    async def stop(self) -> None:
        self._running = False

    def status(self) -> OnlineMemoryImproverStatus:
        return OnlineMemoryImproverStatus(
            improver_id=self.improver_id,
            target_artifact_id=self.target_artifact_id,
            running=self._running,
            trajectories_seen=self._trajectories_seen,
            updates_applied=self._updates_applied,
            entries_touched=self._entries_touched,
            skipped_no_reward=self._skipped_no_reward,
            last_reward=self._last_reward,
            last_error=self._last_error,
        )

    # ---------------- the per-run step ----------------

    async def _resolve_reward(self, trajectory: Trajectory) -> tuple[float | None, GapMeasurement | None]:
        if self.reward_fn is not None:
            value = self.reward_fn(trajectory)
            if inspect.isawaitable(value):
                value = await value
            return (None if value is None else float(value)), None
        assert self.outcome is not None
        measurement = await self.outcome.measure(
            candidate=self.agent.system_prompt, trajectory=trajectory
        )
        if measurement.score is None:
            return None, measurement
        return float(measurement.score), measurement

    async def _on_session_end(self, trajectory: Trajectory, **_: Any) -> None:
        if not self._running:
            return None
        self._trajectories_seen += 1
        try:
            reward, measurement = await self._resolve_reward(trajectory)
            if reward is None:
                self._skipped_no_reward += 1
                return None
            self._last_reward = reward
            touched = await self.learner.update(
                trajectory, reward=reward, outcome_measurement=measurement
            )
            self._updates_applied += 1
            self._entries_touched += len(touched)
        except Exception as exc:  # the hot path never pays for a learning bug
            self._last_error = f"{type(exc).__name__}: {exc}"
        return None

    async def step(self, trajectory: Trajectory) -> list:
        """Apply one update outside the hook path (replay, tests, scripts)."""
        reward, measurement = await self._resolve_reward(trajectory)
        if reward is None:
            self._skipped_no_reward += 1
            return []
        self._last_reward = reward
        touched = await self.learner.update(trajectory, reward=reward, outcome_measurement=measurement)
        self._updates_applied += 1
        self._entries_touched += len(touched)
        return touched
