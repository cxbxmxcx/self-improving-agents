"""ExpeL's four memory operators as archive mutations. SPEC §7.3, §11.5.

ExpeL (Zhao et al., 2023) maintains its insight pool with ADD, UPVOTE,
DOWNVOTE, and EDIT. Here each operator is a mutation over a MEMORY_ENTRY
artifact: ADD is a genesis, the other three mint a child version with a
parent pointer, and the tier's live index row is refreshed to point at the
newest version.

"Reinforcement" in this chapter means moving the utility number carried in
artifact content, never a weight update. Utility lives in content rather
than metadata because the archive de-duplicates by content hash within an
id, so a metadata-only change would collapse onto its parent.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from helix.artifact import Artifact
from helix.memory.archive_bridge import (
    entry_utility,
    latest_artifact,
    memory_artifact_content,
    record_version,
)
from helix.memory.base import EntryId, EvictionPolicy, MemoryEntry, memory_artifact_id
from helix.observability.events import Consolidated, MemoryWritten


class Operator(str, Enum):
    ADD = "add"
    UPVOTE = "upvote"
    DOWNVOTE = "downvote"
    EDIT = "edit"


@dataclass
class OperatorResult:
    """What one operator did, for callers that log or render lineage."""

    operator: Operator
    entry: MemoryEntry
    parent_ref: tuple[str, int] | None
    new_ref: tuple[str, int] | None
    utility: float


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _require_archive(tier: Any) -> Any:
    archive = getattr(tier, "archive", None)
    if archive is None:
        raise ValueError(
            f"memory tier {getattr(tier, 'tier_name', tier)!r} has no archive; "
            "ExpeL operators need lineage, build the tier with archive=..."
        )
    return archive


async def _resolve(tier: Any, entry_or_id: MemoryEntry | EntryId) -> tuple[MemoryEntry, Artifact]:
    """Fetch the live index row and the newest artifact version behind it."""
    archive = _require_archive(tier)
    if isinstance(entry_or_id, MemoryEntry):
        entry = await tier.get(entry_or_id.id) or entry_or_id
    else:
        entry = await tier.get(entry_or_id)
        if entry is None:
            raise KeyError(f"no entry {entry_or_id!r} in tier {tier.tier_name!r}")
    # The index row may have been written before the tier had an archive, so
    # fall back on the canonical id rather than insisting on a stored ref.
    artifact_id = entry.artifact_id or memory_artifact_id(tier.tier_name, entry.id)
    parent = await latest_artifact(archive, artifact_id)
    if parent is None:
        raise KeyError(f"entry {entry.id!r} has no artifact {artifact_id!r} in the archive")
    return entry, parent


async def _publish(bus: Any, tier: Any, entry: MemoryEntry, operator: Operator) -> None:
    if bus is None:
        return
    await bus.publish(
        MemoryWritten(
            tier=tier.tier_name,
            entry_id=entry.id,
            scope=entry.scope_key.to_str(),
            artifact_ref=list(entry.artifact_ref) if entry.artifact_ref else None,
            operator=operator.value,
        )
    )


async def _vote(
    tier: Any,
    entry_or_id: MemoryEntry | EntryId,
    delta: float,
    operator: Operator,
    created_by: str,
    reason: str | None,
    bus: Any,
) -> MemoryEntry:
    entry, parent = await _resolve(tier, entry_or_id)
    before = entry_utility(parent)
    after = _clamp(before + delta)
    new_ref = await record_version(
        tier.archive,
        parent,
        memory_artifact_content(tier.tier_name, entry, utility=after),
        created_by=created_by,
        operator=operator.value,
        extra_metadata={
            "reason": reason,
            "delta": float(delta),
            "utility_before": before,
            "utility_after": after,
        },
    )
    # The row carries the latest ref, so tier.write only refreshes the index
    # instead of minting a second genesis.
    entry.score = after
    entry.artifact_ref = new_ref
    await tier.write(entry)
    await _publish(bus, tier, entry, operator)
    return entry


# ---------------------------------------------------------------------------
# Public operators
# ---------------------------------------------------------------------------

async def add(
    tier: Any,
    entry: MemoryEntry,
    *,
    archive: Any | None = None,
    created_by: str = "agent",
    bus: Any | None = None,
) -> MemoryEntry:
    """ExpeL ADD: a genesis artifact plus a fresh index row.

    The tier's write already records the genesis when it holds an archive;
    this wrapper only names the operator, stamps authorship, and publishes.
    """
    if archive is not None and getattr(tier, "archive", None) is None:
        tier.archive = archive
    _require_archive(tier)
    entry.metadata.setdefault("created_by", created_by)
    await tier.write(entry)
    await _publish(bus, tier, entry, Operator.ADD)
    return entry


async def upvote(
    tier: Any,
    entry_or_id: MemoryEntry | EntryId,
    delta: float = 0.1,
    *,
    created_by: str = "agent",
    reason: str | None = None,
    bus: Any | None = None,
) -> MemoryEntry:
    """ExpeL UPVOTE: a child version whose utility rose by delta, clamped to 1."""
    return await _vote(tier, entry_or_id, abs(delta), Operator.UPVOTE, created_by, reason, bus)


async def downvote(
    tier: Any,
    entry_or_id: MemoryEntry | EntryId,
    delta: float = 0.1,
    *,
    created_by: str = "agent",
    reason: str | None = None,
    bus: Any | None = None,
) -> MemoryEntry:
    """ExpeL DOWNVOTE: a child version whose utility fell by delta, clamped to 0."""
    return await _vote(tier, entry_or_id, -abs(delta), Operator.DOWNVOTE, created_by, reason, bus)


async def edit(
    tier: Any,
    entry_or_id: MemoryEntry | EntryId,
    new_content: dict[str, Any],
    *,
    created_by: str = "agent",
    reason: str | None = None,
    bus: Any | None = None,
) -> MemoryEntry:
    """ExpeL EDIT: a child version with a rewritten payload and unchanged utility."""
    entry, parent = await _resolve(tier, entry_or_id)
    utility = entry_utility(parent)
    entry.content = dict(new_content)
    new_ref = await record_version(
        tier.archive,
        parent,
        memory_artifact_content(tier.tier_name, entry, utility=utility),
        created_by=created_by,
        operator=Operator.EDIT.value,
        extra_metadata={"reason": reason},
    )
    # The payload changed, so the stale vector must be recomputed by the tier.
    entry.embedding = None
    entry.score = utility
    entry.artifact_ref = new_ref
    await tier.write(entry)
    await _publish(bus, tier, entry, Operator.EDIT)
    return entry


async def evict_below(
    tier: Any,
    min_utility: float,
    *,
    bus: Any | None = None,
) -> int:
    """The DOWNVOTE-to-eviction path: drop live rows whose utility sank too far.

    Only the index forgets; the archive keeps every version so the lineage
    that led to eviction stays inspectable. Returns the number of rows dropped.
    """
    _require_archive(tier)
    dropped = await tier.evict(EvictionPolicy(min_score=min_utility))
    if bus is not None and dropped:
        await bus.publish(
            Consolidated(
                tier=tier.tier_name,
                entries_dropped=dropped,
                notes=f"evict_below({min_utility})",
            )
        )
    return dropped
