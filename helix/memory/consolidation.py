"""Consolidation: the 3 AM pass over episodic memory. SPEC §7.4.

The hippocampus replays the day's episodes during slow-wave sleep and keeps
only what was worth keeping; this module is that replay for a memory tier.
One pass prunes low-utility rows, merges near-duplicates, promotes entries
that proved themselves in a session to the wider user scope, and caps the
index size. Every change is an archive mutation with a parent pointer or a
fresh genesis; the index forgets, the archive never does.
"""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from helix.memory import operators as ops
from helix.memory.archive_bridge import entry_utility, memory_artifact_content, record_version
from helix.memory.base import (
    ConsolidationReport,
    EvictionPolicy,
    MemoryEntry,
    Scope,
    ScopeKey,
    ref_to_str,
)
from helix.observability.events import Consolidated

Summarizer = Callable[[list[dict[str, Any]]], Awaitable[dict[str, Any] | None]]


@dataclass
class ConsolidationPolicy:
    """Knobs for one consolidation pass.

    `summarizer` is the LLM-optional hook: given the payloads of a merge
    group it returns the merged payload, or None to fall back on keeping
    the higher-utility entry's payload verbatim.
    """

    min_utility: float = 0.1
    merge_similarity: float = 0.92
    promote_utility: float = 0.7
    max_entries: int | None = None
    summarizer: Summarizer | None = None
    promote_to: Scope = Scope.USER
    promote_key: str | None = None


def _cosine(a: list[float] | None, b: list[float] | None) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    return max(0.0, min(1.0, sum(x * y for x, y in zip(a, b))))


def _merge_groups(entries: list[MemoryEntry], threshold: float) -> list[list[MemoryEntry]]:
    """Greedy clustering: each entry joins the first group whose keeper it matches."""
    groups: list[list[MemoryEntry]] = []
    for entry in sorted(entries, key=lambda e: e.score, reverse=True):
        for group in groups:
            if _cosine(group[0].embedding, entry.embedding) >= threshold:
                group.append(entry)
                break
        else:
            groups.append([entry])
    return [g for g in groups if len(g) > 1]


async def _merge(tier: Any, policy: ConsolidationPolicy, bus: Any) -> tuple[int, int]:
    """Returns (merges performed, rows merged away)."""
    merges = 0
    merged_away = 0
    by_scope: dict[ScopeKey, list[MemoryEntry]] = {}
    for entry in await tier.list_all():
        by_scope.setdefault(entry.scope_key, []).append(entry)

    for scope_entries in by_scope.values():
        for group in _merge_groups(scope_entries, policy.merge_similarity):
            keeper, others = group[0], group[1:]
            payload = None
            if policy.summarizer is not None:
                payload = await policy.summarizer([e.content for e in group])
            if payload is None:
                payload = dict(keeper.content)
            payload["merged_ids"] = sorted(
                set(payload.get("merged_ids", [])) | {e.id for e in group}
            )
            merged_from = [ref_to_str(e.artifact_ref) for e in others if e.artifact_ref]
            await _merge_into(tier, keeper, payload, merged_from, bus)
            for e in others:
                await tier.remove(e.id)
            merges += 1
            merged_away += len(others)
    return merges, merged_away


async def _merge_into(
    tier: Any,
    keeper: MemoryEntry,
    payload: dict[str, Any],
    merged_from: list[str],
    bus: Any,
) -> MemoryEntry:
    """An EDIT child version of the keeper that names the rows it absorbed.

    The operators module's edit() only records a reason, and merged_from has
    to sit on the version itself for the lineage view, so this mints the
    child through the same bridge with the extra metadata.
    """
    entry, parent = await ops._resolve(tier, keeper)
    utility = entry_utility(parent)
    entry.content = dict(payload)
    entry.artifact_ref = await record_version(
        tier.archive,
        parent,
        memory_artifact_content(tier.tier_name, entry, utility=utility),
        created_by="consolidation",
        operator=ops.Operator.EDIT.value,
        extra_metadata={
            "reason": f"merged {len(merged_from)} near-duplicate(s)",
            "merged_from": merged_from,
        },
    )
    entry.embedding = None
    entry.score = utility
    await tier.write(entry)
    await ops._publish(bus, tier, entry, ops.Operator.EDIT)
    return entry


async def _promote(tier: Any, policy: ConsolidationPolicy, bus: Any) -> list[str]:
    """Re-add strong session entries at the wider scope as new genesis artifacts."""
    promoted: list[str] = []
    for entry in await tier.list_all():
        if entry.scope_key.scope is not Scope.SESSION or entry.score < policy.promote_utility:
            continue
        if entry.metadata.get("promoted_to"):
            continue
        key = policy.promote_key or entry.metadata.get("user_id") or entry.scope_key.key
        clone = MemoryEntry(
            id=f"{entry.id}.{policy.promote_to.value}",
            scope_key=ScopeKey(policy.promote_to, key),
            content=dict(entry.content),
            embedding=list(entry.embedding) if entry.embedding else None,
            score=entry.score,
            metadata={
                **{k: v for k, v in entry.metadata.items() if k != "created_by"},
                "promoted_from": ref_to_str(entry.artifact_ref),
            },
        )
        await ops.add(tier, clone, created_by="consolidation", bus=bus)
        # The session row stays but remembers it was promoted so the next
        # pass does not promote it again.
        entry.metadata["promoted_to"] = ref_to_str(clone.artifact_ref)
        await tier.write(entry)
        promoted.append(f"{entry.id}->{clone.id}")
    return promoted


async def consolidate_episodic(
    tier: Any,
    policy: ConsolidationPolicy | None = None,
    *,
    bus: Any | None = None,
) -> ConsolidationReport:
    """Prune, merge, promote, cap; publish one Consolidated event."""
    policy = policy or ConsolidationPolicy()
    before = await tier.count()
    # Every step below is an archive mutation, so a Chapter 2 tier with no
    # archive has nothing to consolidate against; report state and stop.
    if getattr(tier, "archive", None) is None:
        return ConsolidationReport(
            tier=tier.tier_name, entries_before=before, entries_after=before,
            notes="no archive attached; consolidation needs lineage, build the tier with archive=...",
        )

    pruned = await ops.evict_below(tier, policy.min_utility)
    merges, merged_away = await _merge(tier, policy, bus)
    promoted = await _promote(tier, policy, bus)
    capped = 0
    if policy.max_entries is not None:
        capped = await tier.evict(EvictionPolicy(max_entries=policy.max_entries))

    after = await tier.count()
    notes = (
        f"pruned={pruned} merged={merges} merged_away={merged_away} "
        f"promoted={len(promoted)} capped={capped}"
    )
    if promoted:
        notes += "; promotions: " + ", ".join(promoted)
    report = ConsolidationReport(
        tier=tier.tier_name,
        entries_before=before,
        entries_after=after,
        entries_summarized=merges,
        entries_dropped=pruned + merged_away,
        notes=notes,
    )
    if bus is not None:
        await bus.publish(
            Consolidated(
                tier=report.tier,
                entries_before=report.entries_before,
                entries_after=report.entries_after,
                entries_summarized=report.entries_summarized,
                entries_dropped=report.entries_dropped,
                notes=report.notes,
            )
        )
    return report


async def schedule_consolidation(
    tier: Any,
    policy: ConsolidationPolicy | None = None,
    interval_sec: float = 6 * 3600,
    *,
    bus: Any | None = None,
    on_report: Callable[[ConsolidationReport], Any] | None = None,
) -> asyncio.Task:
    """The 3 AM scheduler: run the pass every interval until cancelled.

    The first pass runs immediately so a short-lived process still
    consolidates once. Cancel the task to stop; cancellation is clean.
    """

    async def _loop() -> None:
        while True:
            report = await consolidate_episodic(tier, policy, bus=bus)
            if on_report is not None:
                on_report(report)
            await asyncio.sleep(interval_sec)

    return asyncio.create_task(_loop())


async def stop_consolidation(task: asyncio.Task) -> None:
    """Cancel a scheduled pass and wait for it to unwind."""
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task
