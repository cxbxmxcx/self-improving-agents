"""Value-based reinforcement learning for the harness's memory entries.

The fixed one-step rule is q_new = q + alpha * (credit - q). Feedback changes
stored utility values; the learner does not tune retrieval settings, rewrite
memory hooks, or learn a different update algorithm. This is an observed-return
value update, not general next-state Q-learning or a model-weight update.

The live tier supplies recall and the archive retains immutable versions. The
optional content editor is an explicit extension; Chapter 5 does not enable it.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from helix.artifact import Artifact, ArtifactKind
from helix.memory import operators as ops
from helix.memory.archive_bridge import entry_payload, entry_utility, latest_artifact
from helix.memory.base import MemoryEntry, memory_artifact_id
from helix.observability.bus import EventBus, get_bus
from helix.signal import GapMeasurement
from helix.trajectory import Trajectory

if TYPE_CHECKING:
    from helix.archive import Archive
    from helix.search.base import Variant
    from helix.signals.utility import UtilitySignal

# Keep the archive label stable for existing records and the Search adapter.
SEARCH_METHOD = "memrl"

# An editor rewrites an entry payload in the light of outcome feedback; it is
# injectable so tests and offline runs never need a model.
Editor = Callable[[dict[str, Any], str], Awaitable[dict[str, Any]]]

# Which payload keys stand in for the intent and the experience halves of the
# triplet, in preference order. Episodic entries use user_message/final_output,
# insight entries from the group-relative sampler use insight.
_INTENT_KEYS = ("intent", "user_message", "task", "question", "key")
_EXPERIENCE_KEYS = ("experience", "final_output", "answer", "insight", "summary", "value", "description")


def _first(payload: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in payload and payload[key] not in (None, ""):
            return payload[key]
    return None


def ieu(artifact: Artifact) -> dict[str, Any]:
    """The Intent-Experience-Utility view of a memory-entry artifact.

    The payload keeps whatever keys the tier wrote; this helper only names
    which of them play the intent and experience roles so the chapter can
    talk about triplets without changing the stored shape.
    """
    payload = entry_payload(artifact)
    return {
        "intent": _first(payload, _INTENT_KEYS),
        "experience": _first(payload, _EXPERIENCE_KEYS),
        "utility": entry_utility(artifact),
        "artifact_ref": list(artifact.ref),
    }


class MemoryLearner:
    """Update recalled values with a fixed, configured learning rule.

    Credit outside the ambiguous band supplies the target; alpha controls the
    move toward it. UPVOTE and DOWNVOTE record that move as child versions in
    the same memory tier read by the agent harness.
    """

    algorithm = "one_step_value_update"

    def __init__(
        self,
        tier: Any,
        archive: Archive,
        utility: UtilitySignal,
        alpha: float = 0.3,
        upvote_threshold: float = 0.5,
        downvote_threshold: float = 0.2,
        editor: Editor | None = None,
        created_by: str = "memrl",
        bus: EventBus | None = None,
    ) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        if downvote_threshold > upvote_threshold:
            raise ValueError("downvote_threshold must not exceed upvote_threshold")
        self.tier = tier
        self.archive = archive
        # The tier must share the archive, otherwise the operators would mint
        # lineage in one store and the search would read utility from another.
        if getattr(tier, "archive", None) is None:
            tier.archive = archive
        elif tier.archive is not archive:
            raise ValueError("MemoryLearner must use the memory tier's archive")
        self.utility = utility
        self.alpha = alpha
        self.upvote_threshold = upvote_threshold
        self.downvote_threshold = downvote_threshold
        self.editor = editor
        self.created_by = created_by
        self.bus = bus or get_bus()
        # The variants the most recent update produced. They are a convenience
        # for propose(); every one of them is also in the archive, which stays
        # the source of truth.
        self._last_variants: list[Variant] = []

    # ---------------- the learning step ----------------

    async def update(
        self,
        trajectory: Trajectory,
        reward: float | None = None,
        outcome_measurement: GapMeasurement | None = None,
    ) -> list[MemoryEntry]:
        """Apply the same one-step value update to each credited entry.

        Returns the entries whose utility moved, each already pointing at its
        new version. Entries the tier no longer indexes, or whose credit fell
        in the ambiguous band, are left alone.
        """
        credits = await self.utility.attribute(
            trajectory, reward=reward, outcome_measurement=outcome_measurement
        )
        feedback = self._feedback_for(trajectory, outcome_measurement)
        self._last_variants = []
        updated: list[MemoryEntry] = []
        for artifact_id, credit_m in credits.items():
            credit = float(credit_m.score or 0.0)
            if self.downvote_threshold < credit < self.upvote_threshold:
                continue
            entry = await self._entry_for(artifact_id, credit_m)
            if entry is None:
                continue
            current = await latest_artifact(self.archive, artifact_id)
            if current is None:
                continue
            q = entry_utility(current)
            q_new = q + self.alpha * (credit - q)
            delta = abs(q_new - q)
            if delta < 1e-9:
                continue
            if q_new > q:
                operator = ops.Operator.UPVOTE
                entry = await ops.upvote(
                    self.tier, entry, delta=delta, created_by=self.created_by,
                    reason=self._reason(credit, q, q_new), bus=None,
                )
            else:
                operator = ops.Operator.DOWNVOTE
                entry = await ops.downvote(
                    self.tier, entry, delta=delta, created_by=self.created_by,
                    reason=self._reason(credit, q, q_new), bus=None,
                )
            variant = await self._record(entry, current, credit_m, operator, q, credit)
            self._last_variants.append(variant)

            # ExpeL's EDIT: a failure with feedback is a chance to rewrite the
            # experience rather than only devalue it.
            if (
                operator is ops.Operator.DOWNVOTE
                and self.editor is not None
                and feedback
            ):
                before_edit = await latest_artifact(self.archive, artifact_id)
                new_payload = await self.editor(dict(entry.content), feedback)
                if new_payload and new_payload != entry.content:
                    entry = await ops.edit(
                        self.tier, entry, new_payload, created_by=self.created_by,
                        reason=feedback[:200], bus=None,
                    )
                    edited = await self._record(
                        entry, before_edit, credit_m, ops.Operator.EDIT, entry.score, credit
                    )
                    self._last_variants.append(edited)
            updated.append(entry)
        return updated

    async def _entry_for(self, artifact_id: str, credit_m: GapMeasurement) -> MemoryEntry | None:
        entry_id = credit_m.metadata.get("entry_id")
        entry = await self.tier.get(entry_id) if entry_id else None
        if entry is None and hasattr(self.tier, "by_artifact_id"):
            entry = await self.tier.by_artifact_id(artifact_id)
        if entry is None:
            return None
        # Only entries this tier owns can be voted through this tier.
        if memory_artifact_id(self.tier.tier_name, entry.id) != artifact_id:
            return None
        return entry

    async def _record(
        self,
        entry: MemoryEntry,
        parent: Artifact,
        credit_m: GapMeasurement,
        operator: ops.Operator,
        q_before: float,
        credit: float,
    ) -> Variant:
        """Store the credit against the version the operator just minted."""
        from helix.search.base import Variant

        assert entry.artifact_ref is not None
        stored = await self.archive.by_id(entry.artifact_ref[0], entry.artifact_ref[1])
        assert stored is not None, "operator minted a version the archive does not hold"
        new_art = stored.artifact
        q_after = entry_utility(new_art)
        meta = {
            "operator": operator.value,
            "q_before": q_before,
            "q_after": q_after,
            "credit": credit,
            "alpha": self.alpha,
            "learning_algorithm": self.algorithm,
        }
        measurement = dataclasses.replace(credit_m, metadata={**credit_m.metadata, **meta})
        variant = Variant(
            artifact=new_art,
            parent=parent,
            search_method=SEARCH_METHOD,
            measurement=measurement,
            metadata=meta,
        )
        await self.archive.record(variant, measurement)
        return variant

    @staticmethod
    def _reason(credit: float, q: float, q_new: float) -> str:
        return f"memrl credit {credit:.2f}: q {q:.3f} -> {q_new:.3f}"

    @staticmethod
    def _feedback_for(trajectory: Trajectory, outcome_measurement: GapMeasurement | None) -> str | None:
        if outcome_measurement is not None and outcome_measurement.feedback:
            return outcome_measurement.feedback
        fb = trajectory.metadata.get("feedback")
        return str(fb) if fb else None

    # ---------------- reading the table ----------------

    async def best_entries(self, k: int = 5) -> list[Variant]:
        """The top-k measured memory entries, never a prompt or a skill."""
        return await self.archive.best(k=k, kind=ArtifactKind.MEMORY_ENTRY)

    async def prune(self, min_utility: float | None = None) -> int:
        """Forget live rows whose utility sank below the floor; the archive keeps them."""
        floor = self.downvote_threshold if min_utility is None else min_utility
        return await ops.evict_below(self.tier, floor, bus=self.bus)
