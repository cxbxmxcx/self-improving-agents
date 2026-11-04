"""UtilitySignal: credit an outcome back to the memory entries recalled on the way to it.

A memory entry is an artifact (SPEC §7.3), and an artifact needs a measurement
before any search can act on it. The outcome of a run says nothing about one
entry directly; it says something about the whole set the PRE_MODEL hook
injected. This signal turns that run-level reward into per-entry credit so the
archive can rank memory by how often recalling it led somewhere good.

The reward can come from another Signal (a ground-truth checker, a judge, an
environment reward) or be handed in directly. Attribution is deliberately
simple: every recalled entry shares the reward (Memento-style equal credit),
or earlier-recalled entries get more of it because the tiers rank by relevance
and the top hit shaped the answer most. Confidence follows the outcome signal,
so a low-confidence judge never produces high-confidence memory credit.
"""

from __future__ import annotations

from typing import Any, Literal

from helix.artifact import Artifact
from helix.search.base import Variant
from helix.signal import (
    Cost,
    GapMeasurement,
    Preference,
    Signal,
    SignalKind,
)
from helix.trajectory import Trajectory

Attribution = Literal["equal", "position"]
# Matches helix.memory.hooks.RECALLED_KEY; duplicated so a signal import never
# drags the memory tiers (and their embedder) into the process.
RECALLED_KEY = "memory_recalled"
SEARCH_METHOD = "utility"


def _clip(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def position_weights(n: int) -> list[float]:
    """Halving weights by recall rank, normalized so the first entry gets 1.0."""
    return [0.5 ** i for i in range(n)]


class UtilitySignal:
    """Per-entry credit from a run outcome. SPEC §3.3 (UTILITY), v0.3.

    `measure()` keeps the Signal protocol for one memory artifact; `attribute()`
    is the batch form most callers want because a run credits every entry it
    recalled at once. `record_all()` writes those credits to the archive against
    the exact recalled version, which is what MemRL-style search reads back.
    """

    def __init__(
        self,
        outcome: Signal | None = None,
        attribution: Attribution = "equal",
        signal_id: str = "utility",
        signal_version: int = 1,
    ) -> None:
        if attribution not in ("equal", "position"):
            raise ValueError(f"unknown attribution {attribution!r}; use 'equal' or 'position'")
        self.outcome = outcome
        self.attribution: Attribution = attribution
        self._signal_id = signal_id
        self._version = signal_version

    @property
    def kind(self) -> SignalKind:
        return SignalKind.UTILITY

    @property
    def cost_estimate(self) -> Cost:
        # The only cost is whatever the wrapped outcome signal charges.
        return self.outcome.cost_estimate if self.outcome is not None else Cost()

    @property
    def signal_id(self) -> str:
        return self._signal_id

    @property
    def signal_version(self) -> int:
        return self._version

    # ---------------- reward resolution ----------------

    async def _resolve_reward(
        self,
        trajectory: Trajectory,
        candidate: Artifact | None,
        reward: float | None,
        outcome_measurement: GapMeasurement | None,
    ) -> tuple[float | None, float, str | None]:
        """Return (reward, confidence, outcome_signal_id) from whichever source was given.

        An explicit reward is ground truth as far as this signal knows, so it
        carries full confidence; a measurement or a wrapped signal carries its own.
        """
        if reward is not None:
            return _clip(reward), 1.0, None
        if outcome_measurement is None and self.outcome is not None:
            outcome_measurement = await self.outcome.measure(candidate, trajectory)
        if outcome_measurement is None or outcome_measurement.score is None:
            return None, 0.0, getattr(outcome_measurement, "signal_id", None)
        return (
            _clip(outcome_measurement.score),
            _clip(outcome_measurement.confidence),
            outcome_measurement.signal_id,
        )

    # ---------------- batch attribution ----------------

    async def attribute(
        self,
        trajectory: Trajectory,
        reward: float | None = None,
        outcome_measurement: GapMeasurement | None = None,
    ) -> dict[str, GapMeasurement]:
        """Credit every recalled entry, keyed by memory artifact id.

        Entries without an artifact ref cannot be measured in the archive and
        are skipped. A run that recalled nothing has nothing to credit.
        """
        recalled = [
            r for r in (trajectory.metadata.get(RECALLED_KEY) or []) if r.get("artifact_ref")
        ]
        if not recalled:
            return {}

        resolved_reward, confidence, outcome_id = await self._resolve_reward(
            trajectory, None, reward, outcome_measurement
        )
        if resolved_reward is None:
            return {}

        weights = (
            position_weights(len(recalled))
            if self.attribution == "position"
            else [1.0] * len(recalled)
        )
        out: dict[str, GapMeasurement] = {}
        for rank, (row, w) in enumerate(zip(recalled, weights)):
            artifact_id = str(row["artifact_ref"][0])
            out[artifact_id] = GapMeasurement(
                score=_clip(resolved_reward * w),
                preference=Preference.NONE,
                feedback=(
                    f"utility credit {resolved_reward * w:.2f} from reward "
                    f"{resolved_reward:.2f} ({self.attribution}, rank {rank})"
                ),
                confidence=confidence,
                signal_id=self.signal_id,
                signal_version=self.signal_version,
                raw_value=resolved_reward,
                cost=Cost(),
                metadata={
                    "trajectory_id": trajectory.id,
                    "n_recalled": len(recalled),
                    "attribution": self.attribution,
                    "rank": rank,
                    "weight": w,
                    "entry_id": row.get("entry_id"),
                    "tier": row.get("tier"),
                    "artifact_ref": list(row["artifact_ref"]),
                    "outcome_signal_id": outcome_id,
                },
            )
        return out

    # ---------------- Signal protocol ----------------

    async def measure(
        self,
        candidate: Artifact,
        trajectory: Trajectory | None = None,
        reference: Artifact | None = None,
        ground_truth: Any | None = None,
    ) -> GapMeasurement:
        """Credit for one memory artifact. `ground_truth` may carry a float reward."""
        if trajectory is None:
            return self._not_recalled("trajectory not provided", None)
        reward = ground_truth if isinstance(ground_truth, (int, float)) and not isinstance(ground_truth, bool) else None
        credits = await self.attribute(trajectory, reward=reward)
        hit = credits.get(candidate.id)
        if hit is None:
            return self._not_recalled("not recalled", trajectory.id)
        return hit

    def _not_recalled(self, reason: str, trajectory_id: str | None) -> GapMeasurement:
        return GapMeasurement(
            score=None,
            preference=Preference.NONE,
            feedback=reason,
            confidence=0.0,
            signal_id=self.signal_id,
            signal_version=self.signal_version,
            cost=Cost(),
            metadata={"reason": reason, "trajectory_id": trajectory_id},
        )

    # ---------------- archive write ----------------

    async def record_all(
        self,
        archive: Any,
        trajectory: Trajectory,
        reward: float | None = None,
        outcome_measurement: GapMeasurement | None = None,
    ) -> list[tuple[str, GapMeasurement]]:
        """Write each credit to the archive against the recalled version.

        The version matters: a later mutation of the same entry must not
        inherit credit earned by the content the run actually saw.
        """
        credits = await self.attribute(trajectory, reward=reward, outcome_measurement=outcome_measurement)
        written: list[tuple[str, GapMeasurement]] = []
        for artifact_id, m in credits.items():
            ref = m.metadata["artifact_ref"]
            variant = await archive.by_id(artifact_id, int(ref[1]))
            if variant is None:
                continue
            art = variant.artifact
            # The artifact is already stored; passing it as its own parent keeps
            # record() from re-inserting a parent row it does not need.
            await archive.record(
                Variant(artifact=art, parent=art, search_method=SEARCH_METHOD), m
            )
            written.append((artifact_id, m))
        return written
