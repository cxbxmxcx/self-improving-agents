"""Chapter 5: ExpeL's ADD / UPVOTE / DOWNVOTE / EDIT as archive mutations.

Every operator except ADD mints a child version with a parent pointer; the
tier's live index row tracks the newest ref and score, and eviction only
forgets the row while the archive keeps the whole chain. SPEC §7.3, §11.5.
"""

from __future__ import annotations

import pytest

from helix.archive import SQLiteArchive
from helix.memory import operators as ops
from helix.memory.archive_bridge import entry_payload, entry_utility
from helix.memory.base import MemoryEntry, Scope, ScopeKey
from helix.memory.episodic import EpisodicMemory
from helix.observability.bus import EventBus


@pytest.fixture
def archive(tmp_path):
    return SQLiteArchive(str(tmp_path / "archive.sqlite"))


@pytest.fixture
def tier(archive):
    return EpisodicMemory(":memory:", archive=archive)


def _entry(eid: str = "e1", score: float = 0.5) -> MemoryEntry:
    return MemoryEntry(
        id=eid,
        scope_key=ScopeKey(Scope.USER, "alice"),
        content={"user_message": "how do I reset my password", "final_output": "use the portal"},
        score=score,
    )


async def _version(archive, artifact_id: str, version: int):
    variant = await archive.by_id(artifact_id, version=version)
    assert variant is not None, f"{artifact_id} v{version} missing from archive"
    return variant.artifact


# ---------------------------------------------------------------------------
# ADD
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_add_records_a_genesis_artifact_with_the_add_operator(tier, archive):
    entry = await ops.add(tier, _entry(), created_by="human")

    assert entry.artifact_ref == ("memory.episodic.e1", 1)
    art = await _version(archive, "memory.episodic.e1", 1)
    assert art.parent_id is None
    assert art.created_by == "human"
    assert art.metadata["operator"] == "add"
    assert entry_utility(art) == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_operators_refuse_a_tier_without_an_archive():
    bare = EpisodicMemory(":memory:")
    with pytest.raises(ValueError, match="no archive"):
        await ops.add(bare, _entry())
    with pytest.raises(ValueError, match="no archive"):
        await ops.evict_below(bare, 0.1)


# ---------------------------------------------------------------------------
# UPVOTE / DOWNVOTE
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_upvote_mints_a_child_version_with_parent_pointer_and_reason(tier, archive):
    await ops.add(tier, _entry())
    entry = await ops.upvote(tier, "e1", reason="answer was reused")

    assert entry.artifact_ref == ("memory.episodic.e1", 2)
    assert entry.score == pytest.approx(0.6)
    child = await _version(archive, "memory.episodic.e1", 2)
    assert child.parent_id == ("memory.episodic.e1", 1)
    assert child.metadata["operator"] == "upvote"
    assert child.metadata["reason"] == "answer was reused"
    assert child.metadata["utility_before"] == pytest.approx(0.5)
    assert entry_utility(child) == pytest.approx(0.6)
    # The payload rides along unchanged.
    assert entry_payload(child) == _entry().content


@pytest.mark.asyncio
async def test_downvote_lowers_utility_and_accepts_an_entry_object(tier, archive):
    entry = await ops.add(tier, _entry())
    entry = await ops.downvote(tier, entry, delta=0.3)

    assert entry.score == pytest.approx(0.2)
    child = await _version(archive, "memory.episodic.e1", 2)
    assert child.metadata["operator"] == "downvote"
    assert child.metadata["delta"] == pytest.approx(-0.3)


@pytest.mark.asyncio
async def test_utility_clamps_to_the_unit_interval(tier, archive):
    await ops.add(tier, _entry(score=0.95))
    entry = await ops.upvote(tier, "e1", delta=0.5)
    assert entry.score == pytest.approx(1.0)

    entry = await ops.downvote(tier, "e1", delta=5.0)
    assert entry.score == pytest.approx(0.0)
    assert entry_utility(await _version(archive, "memory.episodic.e1", 3)) == 0.0


@pytest.mark.asyncio
async def test_votes_on_the_same_entry_chain_v1_to_v2_to_v3(tier, archive):
    await ops.add(tier, _entry())
    await ops.upvote(tier, "e1")
    await ops.upvote(tier, "e1")

    v3 = await _version(archive, "memory.episodic.e1", 3)
    chain = await archive.lineage(v3)
    assert [a.version for a in chain] == [1, 2, 3]
    assert [a.parent_id for a in chain] == [None, ("memory.episodic.e1", 1), ("memory.episodic.e1", 2)]
    assert [entry_utility(a) for a in chain] == pytest.approx([0.5, 0.6, 0.7])


@pytest.mark.asyncio
async def test_a_vote_that_restores_the_same_utility_is_still_a_new_node(tier, archive):
    """The archive dedups by content hash within an id; the revision counter
    in content keeps an up-then-down round trip from collapsing onto v1, so
    the lineage view shows every vote."""
    await ops.add(tier, _entry())
    await ops.upvote(tier, "e1", delta=0.1)
    entry = await ops.downvote(tier, "e1", delta=0.1)

    assert entry.score == pytest.approx(0.5)
    assert entry.artifact_ref == ("memory.episodic.e1", 3)
    chain = await archive.lineage((await archive.by_id("memory.episodic.e1")).artifact)
    assert [a.content["revision"] for a in chain] == [0, 1, 2]


@pytest.mark.asyncio
async def test_index_row_tracks_the_latest_ref_and_score_without_a_second_genesis(tier, archive):
    await ops.add(tier, _entry())
    await ops.upvote(tier, "e1")
    await ops.downvote(tier, "e1", delta=0.05)

    row = await tier.get("e1")
    assert row.artifact_ref == ("memory.episodic.e1", 3)
    assert row.score == pytest.approx(0.55)
    assert await tier.count() == 1
    assert await archive.next_version("memory.episodic.e1") == 4


# ---------------------------------------------------------------------------
# EDIT
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_edit_rewrites_the_payload_carries_utility_and_reembeds(tier, archive):
    await ops.add(tier, _entry(score=0.7))
    old = await tier.get("e1")
    new_content = {"user_message": "forgot password", "final_output": "portal, then 2FA"}
    entry = await ops.edit(tier, "e1", new_content, reason="tightened the answer")

    assert entry.artifact_ref == ("memory.episodic.e1", 2)
    assert entry.score == pytest.approx(0.7)
    child = await _version(archive, "memory.episodic.e1", 2)
    assert child.parent_id == ("memory.episodic.e1", 1)
    assert child.metadata["operator"] == "edit"
    assert child.metadata["reason"] == "tightened the answer"
    assert entry_payload(child) == new_content
    assert entry_utility(child) == pytest.approx(0.7)

    row = await tier.get("e1")
    assert row.content == new_content
    assert row.embedding is not None and row.embedding != old.embedding


@pytest.mark.asyncio
async def test_operators_raise_key_error_for_an_unknown_entry(tier):
    with pytest.raises(KeyError):
        await ops.upvote(tier, "ghost")


# ---------------------------------------------------------------------------
# Eviction
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_evict_below_drops_the_row_but_the_archive_keeps_every_version(tier, archive):
    await ops.add(tier, _entry("keep", score=0.5))
    await ops.add(tier, _entry("drop", score=0.3))
    await ops.downvote(tier, "drop")
    await ops.downvote(tier, "drop")

    dropped = await ops.evict_below(tier, 0.2)

    assert dropped == 1
    assert await tier.get("drop") is None
    assert await tier.get("keep") is not None
    for v in (1, 2, 3):
        await _version(archive, "memory.episodic.drop", v)


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_each_operator_publishes_a_memory_written_event(tier):
    bus = EventBus()
    seen = []
    bus.subscribe("memory_written", seen.append)

    await ops.add(tier, _entry(), bus=bus)
    await ops.upvote(tier, "e1", bus=bus)
    await ops.downvote(tier, "e1", delta=0.05, bus=bus)
    await ops.edit(tier, "e1", {"user_message": "x", "final_output": "y"}, bus=bus)

    assert [e.operator for e in seen] == ["add", "upvote", "downvote", "edit"]
    assert [e.artifact_ref for e in seen] == [["memory.episodic.e1", v] for v in (1, 2, 3, 4)]
    assert all(e.tier == "episodic" and e.entry_id == "e1" and e.scope == "user:alice" for e in seen)


@pytest.mark.asyncio
async def test_evict_below_publishes_a_consolidated_event_with_the_drop_count(tier):
    bus = EventBus()
    seen = []
    bus.subscribe("consolidated", seen.append)
    await ops.add(tier, _entry(score=0.0))

    await ops.evict_below(tier, 0.1, bus=bus)

    assert len(seen) == 1 and seen[0].entries_dropped == 1
