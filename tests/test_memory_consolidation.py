"""Chapter 5: the 3 AM consolidation pass over episodic memory. SPEC §7.4.

Prune forgets index rows while the archive keeps them, merge mints an EDIT
child naming what it absorbed, promote re-adds strong session entries at
user scope with provenance, and the scheduler runs the pass on a timer.
"""

from __future__ import annotations

import asyncio

import pytest

from helix.archive import SQLiteArchive
from helix.memory import operators as ops
from helix.memory.base import MemoryEntry, Scope, ScopeKey
from helix.memory.consolidation import (
    ConsolidationPolicy,
    consolidate_episodic,
    schedule_consolidation,
    stop_consolidation,
)
from helix.memory.episodic import EpisodicMemory
from helix.observability.bus import EventBus


@pytest.fixture
def archive(tmp_path):
    return SQLiteArchive(str(tmp_path / "archive.sqlite"))


@pytest.fixture
def tier(archive):
    return EpisodicMemory(":memory:", archive=archive)


def _entry(eid: str, score: float, text: str = "how do I reset my password", scope: Scope = Scope.USER, key: str = "alice") -> MemoryEntry:
    return MemoryEntry(
        id=eid,
        scope_key=ScopeKey(scope, key),
        content={"user_message": text, "final_output": f"answer for {eid}"},
        score=score,
    )


@pytest.mark.asyncio
async def test_prune_drops_low_utility_rows_but_the_archive_keeps_them(tier, archive):
    await ops.add(tier, _entry("weak", 0.05, "what is the capital of france"))
    await ops.add(tier, _entry("strong", 0.5, "how do I cancel my subscription"))

    report = await consolidate_episodic(tier, ConsolidationPolicy(min_utility=0.1))

    assert report.entries_before == 2
    assert report.entries_after == 1
    assert report.entries_dropped == 1
    assert await tier.get("weak") is None
    assert await archive.by_id("memory.episodic.weak") is not None


@pytest.mark.asyncio
async def test_merge_keeps_one_row_whose_artifact_is_an_edit_child_with_merged_from(tier, archive):
    await ops.add(tier, _entry("a", 0.6))
    await ops.add(tier, _entry("b", 0.4))

    report = await consolidate_episodic(tier, ConsolidationPolicy(min_utility=0.0))

    assert report.entries_summarized == 1
    assert report.entries_dropped == 1
    assert await tier.get("b") is None
    keeper = await tier.get("a")
    assert keeper.artifact_ref == ("memory.episodic.a", 2)
    assert sorted(keeper.content["merged_ids"]) == ["a", "b"]
    art = (await archive.by_id("memory.episodic.a", version=2)).artifact
    assert tuple(art.parent_id) == ("memory.episodic.a", 1)
    assert art.metadata["operator"] == "edit"
    assert art.metadata["merged_from"] == ["memory.episodic.b:1"]
    assert art.created_by == "consolidation"
    assert await archive.by_id("memory.episodic.b") is not None


@pytest.mark.asyncio
async def test_merge_never_crosses_scope_boundaries(tier):
    await ops.add(tier, _entry("a", 0.6, key="alice"))
    await ops.add(tier, _entry("b", 0.6, key="bob"))

    report = await consolidate_episodic(tier, ConsolidationPolicy(min_utility=0.0))

    assert report.entries_summarized == 0
    assert await tier.count() == 2


@pytest.mark.asyncio
async def test_a_summarizer_rewrites_the_merged_payload(tier):
    await ops.add(tier, _entry("a", 0.6))
    await ops.add(tier, _entry("b", 0.4))

    async def summarize(payloads):
        return {"summary": f"{len(payloads)} entries about password resets"}

    await consolidate_episodic(tier, ConsolidationPolicy(min_utility=0.0, summarizer=summarize))

    keeper = await tier.get("a")
    assert keeper.content["summary"] == "2 entries about password resets"
    assert keeper.embedding is not None


@pytest.mark.asyncio
async def test_promotion_creates_a_user_scope_genesis_with_promoted_from(tier, archive):
    entry = _entry("s1", 0.9, "the user prefers metric units", scope=Scope.SESSION, key="sess-1")
    entry.metadata["user_id"] = "alice"
    await ops.add(tier, entry)

    report = await consolidate_episodic(tier, ConsolidationPolicy(min_utility=0.0))

    assert "s1->s1.user" in report.notes
    promoted = await tier.get("s1.user")
    assert promoted.scope_key == ScopeKey(Scope.USER, "alice")
    assert promoted.metadata["promoted_from"] == "memory.episodic.s1:1"
    art = (await archive.by_id("memory.episodic.s1.user")).artifact
    assert art.parent_id is None
    assert art.metadata["promoted_from"] == "memory.episodic.s1:1"
    assert (await tier.get("s1")).metadata["promoted_to"] == "memory.episodic.s1.user:1"

    # A second pass does not promote the same session entry twice.
    again = await consolidate_episodic(tier, ConsolidationPolicy(min_utility=0.0))
    assert "promotions" not in again.notes
    assert await tier.count() == 2


@pytest.mark.asyncio
async def test_weak_session_entries_are_not_promoted(tier):
    await ops.add(tier, _entry("s1", 0.3, scope=Scope.SESSION, key="sess-1"))
    await consolidate_episodic(tier, ConsolidationPolicy(min_utility=0.0))
    assert await tier.get("s1.user") is None


@pytest.mark.asyncio
async def test_the_cap_keeps_only_the_newest_entries(tier):
    for i in range(4):
        await ops.add(tier, _entry(f"e{i}", 0.5, f"distinct question number {i} about topic {i * 7}"))

    report = await consolidate_episodic(tier, ConsolidationPolicy(min_utility=0.0, max_entries=2))

    assert report.entries_after == 2
    assert "capped=2" in report.notes


@pytest.mark.asyncio
async def test_the_consolidated_event_carries_the_report_counts(tier):
    bus = EventBus()
    seen = []
    bus.subscribe("consolidated", seen.append)
    await ops.add(tier, _entry("weak", 0.0, "unrelated question about cats"))
    await ops.add(tier, _entry("a", 0.6))
    await ops.add(tier, _entry("b", 0.4))

    report = await consolidate_episodic(tier, ConsolidationPolicy(min_utility=0.1), bus=bus)

    events = [e for e in seen if e.notes.startswith("pruned=")]
    assert len(events) == 1
    ev = events[0]
    assert ev.tier == "episodic"
    assert ev.entries_before == 3
    assert ev.entries_after == 1
    assert ev.entries_summarized == 1
    assert ev.entries_dropped == 2
    assert report.entries_dropped == 2


@pytest.mark.asyncio
async def test_episodic_consolidate_now_returns_real_counts(tier):
    await ops.add(tier, _entry("weak", 0.0, "unrelated question about cats"))
    await ops.add(tier, _entry("strong", 0.5))

    report = await tier.consolidate()

    assert report.tier == "episodic"
    assert (report.entries_before, report.entries_after) == (2, 1)
    assert report.entries_dropped == 1
    assert "stub" not in report.notes


@pytest.mark.asyncio
async def test_schedule_consolidation_runs_at_least_once_and_cancels_cleanly(tier):
    await ops.add(tier, _entry("weak", 0.0))
    reports = []

    task = await schedule_consolidation(
        tier, ConsolidationPolicy(), interval_sec=3600, on_report=reports.append
    )
    for _ in range(50):
        if reports:
            break
        await asyncio.sleep(0.02)

    assert len(reports) == 1
    assert reports[0].entries_dropped == 1
    await stop_consolidation(task)
    assert task.cancelled()


@pytest.mark.asyncio
async def test_a_tier_without_an_archive_reports_state_and_changes_nothing():
    bare = EpisodicMemory(":memory:")
    await bare.write(_entry("e1", 0.0))
    report = await consolidate_episodic(bare)
    assert (report.entries_before, report.entries_after) == (1, 1)
    assert "no archive" in report.notes
