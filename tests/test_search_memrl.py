"""Chapter 5: MemRL, Q-learning over memory with the archive as the Q-table.

Every Q change is an UPVOTE or DOWNVOTE child version with a parent pointer
and the utility measurement recorded on that new version. No weights move;
the table that learns is the utility number in memory-artifact content.
SPEC §4.2, §7.3, §11.5.
"""

from __future__ import annotations

import pytest

from helix.archive import SQLiteArchive
from helix.artifact import ArtifactKind, genesis
from helix.memory import operators as ops
from helix.memory.archive_bridge import entry_payload, entry_utility
from helix.memory.base import MemoryEntry, Scope, ScopeKey
from helix.memory.episodic import EpisodicMemory
from helix.search.base import Search, SearchBudget, SearchKind, Variant
from helix.search.memrl import MemRLSearch, ieu
from helix.signal import GapMeasurement
from helix.signals.utility import RECALLED_KEY, UtilitySignal
from helix.trajectory import Trajectory


@pytest.fixture
def archive(tmp_path):
    return SQLiteArchive(str(tmp_path / "archive.sqlite"))


@pytest.fixture
def tier(archive):
    return EpisodicMemory(":memory:", archive=archive)


@pytest.fixture
def search(tier, archive):
    return MemRLSearch(tier, archive, UtilitySignal(), alpha=0.5)


def _entry(eid: str, score: float = 0.4, answer: str = "use the portal") -> MemoryEntry:
    return MemoryEntry(
        id=eid,
        scope_key=ScopeKey(Scope.USER, "alice"),
        content={"user_message": f"question {eid}", "final_output": answer},
        score=score,
    )


def _trajectory(entries: list[MemoryEntry], **metadata) -> Trajectory:
    """A run that recalled these entries, in this order, as MemoryHooks records it."""
    t = Trajectory(task="reset my password")
    t.metadata[RECALLED_KEY] = [
        {
            "tier": "episodic",
            "entry_id": e.id,
            "artifact_ref": list(e.artifact_ref) if e.artifact_ref else None,
            "scope": e.scope_key.to_str(),
        }
        for e in entries
    ]
    t.metadata.update(metadata)
    return t


async def _version(archive, artifact_id: str, version: int):
    variant = await archive.by_id(artifact_id, version=version)
    assert variant is not None, f"{artifact_id} v{version} missing from archive"
    return variant


# ---------------------------------------------------------------------------
# Protocol and construction
# ---------------------------------------------------------------------------

def test_memrl_is_a_search_of_the_memory_q_learning_kind(search):
    assert isinstance(search, Search)
    assert search.kind is SearchKind.MEMORY_Q_LEARNING


def test_memrl_rejects_a_learning_rate_outside_the_unit_interval(tier, archive):
    with pytest.raises(ValueError, match="alpha"):
        MemRLSearch(tier, archive, UtilitySignal(), alpha=0.0)
    with pytest.raises(ValueError, match="alpha"):
        MemRLSearch(tier, archive, UtilitySignal(), alpha=1.5)


def test_memrl_attaches_the_archive_to_a_bare_tier(archive):
    bare = EpisodicMemory(":memory:")
    MemRLSearch(bare, archive, UtilitySignal())
    assert bare.archive is archive


# ---------------------------------------------------------------------------
# The Q update
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_q_moves_toward_the_credit_by_alpha(tier, search):
    entry = await ops.add(tier, _entry("e1", score=0.4))

    updated = await search.update(_trajectory([entry]), reward=1.0)

    assert [e.id for e in updated] == ["e1"]
    # q <- 0.4 + 0.5 * (1.0 - 0.4)
    assert updated[0].score == pytest.approx(0.7)
    assert (await tier.get("e1")).score == pytest.approx(0.7)


@pytest.mark.asyncio
async def test_repeated_updates_converge_on_the_credit_without_overshooting(tier, search):
    entry = await ops.add(tier, _entry("e1", score=0.0))
    q_values = []
    for _ in range(4):
        (entry,) = await search.update(_trajectory([entry]), reward=1.0)
        q_values.append(entry.score)

    assert q_values == pytest.approx([0.5, 0.75, 0.875, 0.9375])


@pytest.mark.asyncio
async def test_positive_credit_mints_an_upvote_version_with_parent_pointer(tier, archive, search):
    entry = await ops.add(tier, _entry("e1", score=0.4))

    (entry,) = await search.update(_trajectory([entry]), reward=1.0)

    assert entry.artifact_ref == ("memory.episodic.e1", 2)
    child = (await _version(archive, "memory.episodic.e1", 2)).artifact
    assert child.parent_id == ("memory.episodic.e1", 1)
    assert child.metadata["operator"] == "upvote"
    assert child.created_by == "memrl"
    assert entry_utility(child) == pytest.approx(0.7)


@pytest.mark.asyncio
async def test_low_credit_mints_a_downvote_version(tier, archive, search):
    entry = await ops.add(tier, _entry("e1", score=0.8))

    (entry,) = await search.update(_trajectory([entry]), reward=0.0)

    # q <- 0.8 + 0.5 * (0.0 - 0.8)
    assert entry.score == pytest.approx(0.4)
    child = (await _version(archive, "memory.episodic.e1", 2)).artifact
    assert child.parent_id == ("memory.episodic.e1", 1)
    assert child.metadata["operator"] == "downvote"


@pytest.mark.asyncio
async def test_the_utility_measurement_is_recorded_on_the_new_version(tier, archive, search):
    entry = await ops.add(tier, _entry("e1", score=0.4))
    trajectory = _trajectory([entry])

    await search.update(trajectory, reward=1.0)

    v1 = await _version(archive, "memory.episodic.e1", 1)
    v2 = await _version(archive, "memory.episodic.e1", 2)
    assert v1.measurement is None
    assert v2.measurement is not None
    assert v2.measurement.score == pytest.approx(1.0)
    assert v2.measurement.metadata["operator"] == "upvote"
    assert v2.measurement.metadata["q_before"] == pytest.approx(0.4)
    assert v2.measurement.metadata["q_after"] == pytest.approx(0.7)
    assert v2.measurement.metadata["credit"] == pytest.approx(1.0)
    assert v2.measurement.metadata["trajectory_id"] == trajectory.id
    assert v2.search_method == "memrl"


@pytest.mark.asyncio
async def test_every_recalled_entry_is_updated_in_one_step(tier, search):
    a = await ops.add(tier, _entry("a", score=0.2))
    b = await ops.add(tier, _entry("b", score=0.9))

    updated = await search.update(_trajectory([a, b]), reward=1.0)

    assert {e.id: round(e.score, 3) for e in updated} == {"a": 0.6, "b": 0.95}


@pytest.mark.asyncio
async def test_credit_in_the_ambiguous_band_leaves_the_table_alone(tier, archive, search):
    entry = await ops.add(tier, _entry("e1", score=0.4))

    updated = await search.update(_trajectory([entry]), reward=0.35)

    assert updated == []
    assert await archive.next_version("memory.episodic.e1") == 2


@pytest.mark.asyncio
async def test_an_outcome_measurement_drives_the_update_when_no_reward_is_given(tier, search):
    entry = await ops.add(tier, _entry("e1", score=0.4))
    outcome = GapMeasurement(score=1.0, confidence=0.9, signal_id="checker")

    (entry,) = await search.update(_trajectory([entry]), outcome_measurement=outcome)

    assert entry.score == pytest.approx(0.7)


@pytest.mark.asyncio
async def test_a_run_that_recalled_nothing_changes_nothing(tier, search):
    await ops.add(tier, _entry("e1"))
    assert await search.update(Trajectory(task="cold start"), reward=1.0) == []


@pytest.mark.asyncio
async def test_an_entry_the_tier_forgot_is_skipped(tier, search):
    entry = await ops.add(tier, _entry("e1", score=0.0))
    trajectory = _trajectory([entry])
    await ops.evict_below(tier, 0.1)

    assert await search.update(trajectory, reward=1.0) == []


# ---------------------------------------------------------------------------
# EDIT on failure with feedback
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_a_failure_with_feedback_also_edits_the_experience(tier, archive):
    seen = {}

    async def editor(payload, feedback):
        seen["payload"], seen["feedback"] = payload, feedback
        return {**payload, "final_output": "use the portal, then confirm by email"}

    search = MemRLSearch(tier, archive, UtilitySignal(), alpha=0.5, editor=editor)
    entry = await ops.add(tier, _entry("e1", score=0.8))
    outcome = GapMeasurement(score=0.0, feedback="the answer skipped the email step")

    (entry,) = await search.update(_trajectory([entry]), outcome_measurement=outcome)

    assert seen["feedback"] == "the answer skipped the email step"
    assert entry.artifact_ref == ("memory.episodic.e1", 3)
    v2 = (await _version(archive, "memory.episodic.e1", 2)).artifact
    v3 = (await _version(archive, "memory.episodic.e1", 3)).artifact
    assert v2.metadata["operator"] == "downvote"
    assert v3.metadata["operator"] == "edit"
    assert v3.parent_id == ("memory.episodic.e1", 2)
    assert entry_payload(v3)["final_output"] == "use the portal, then confirm by email"
    assert entry_utility(v3) == pytest.approx(0.4)
    assert [v.metadata["operator"] for v in search._last_variants] == ["downvote", "edit"]


@pytest.mark.asyncio
async def test_a_success_never_calls_the_editor(tier, archive):
    async def editor(payload, feedback):
        raise AssertionError("editor must not run on success")

    search = MemRLSearch(tier, archive, UtilitySignal(), editor=editor)
    entry = await ops.add(tier, _entry("e1"))
    outcome = GapMeasurement(score=1.0, feedback="great answer")

    await search.update(_trajectory([entry]), outcome_measurement=outcome)


# ---------------------------------------------------------------------------
# propose / select
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_propose_with_a_trajectory_context_runs_the_update_and_yields_its_versions(
    tier, archive, search
):
    a = await ops.add(tier, _entry("a", score=0.2))
    b = await ops.add(tier, _entry("b", score=0.9))
    seed = genesis(id="memory.seed", kind=ArtifactKind.TEXT, content="advisory")

    variants = [
        v
        async for v in search.propose(
            seed, search.utility, archive, SearchBudget(),
            context={"trajectory": _trajectory([a, b]), "reward": 0.0},
        )
    ]

    assert [v.artifact.id for v in variants] == ["memory.episodic.a", "memory.episodic.b"]
    assert all(v.search_method == "memrl" for v in variants)
    assert all(v.artifact.version == 2 and v.parent.version == 1 for v in variants)
    assert [v.metadata["operator"] for v in variants] == ["downvote", "downvote"]
    assert variants[1].metadata["q_before"] == pytest.approx(0.9)
    assert variants[1].metadata["q_after"] == pytest.approx(0.45)
    assert variants[1].metadata["credit"] == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_propose_without_a_context_replays_the_last_update(tier, archive, search):
    entry = await ops.add(tier, _entry("e1"))
    await search.update(_trajectory([entry]), reward=1.0)
    seed = (await _version(archive, "memory.episodic.e1", 1)).artifact

    variants = [v async for v in search.propose(seed, search.utility, archive, SearchBudget())]

    assert len(variants) == 1
    assert variants[0].artifact.ref == ("memory.episodic.e1", 2)


@pytest.mark.asyncio
async def test_propose_with_a_memory_seed_yields_only_that_entry(tier, archive, search):
    a = await ops.add(tier, _entry("a"))
    b = await ops.add(tier, _entry("b"))
    await search.update(_trajectory([a, b]), reward=1.0)
    seed = (await _version(archive, "memory.episodic.b", 1)).artifact

    variants = [v async for v in search.propose(seed, search.utility, archive, SearchBudget())]

    assert [v.artifact.id for v in variants] == ["memory.episodic.b"]


@pytest.mark.asyncio
async def test_select_picks_the_highest_utility_variant_and_newest_on_ties(tier, archive, search):
    a = await ops.add(tier, _entry("a", score=0.2))
    b = await ops.add(tier, _entry("b", score=0.6))
    c = await ops.add(tier, _entry("c", score=0.6))
    await search.update(_trajectory([a, b, c]), reward=1.0)

    winner = await search.select(search._last_variants, search.utility, archive)
    assert winner.id == "memory.episodic.c"

    tied = [v for v in search._last_variants if v.artifact.id != "memory.episodic.c"]
    assert (await search.select(tied, search.utility, archive)).id == "memory.episodic.b"


@pytest.mark.asyncio
async def test_select_falls_back_to_the_best_measured_memory_entry(tier, archive, search):
    a = await ops.add(tier, _entry("a", score=0.1))
    b = await ops.add(tier, _entry("b", score=0.9))
    await search.update(_trajectory([a]), reward=1.0)
    await search.update(_trajectory([b]), reward=0.0)

    winner = await search.select([], search.utility, archive)
    assert winner.id == "memory.episodic.a"


@pytest.mark.asyncio
async def test_select_with_nothing_measured_raises(search, archive):
    with pytest.raises(ValueError):
        await search.select([], search.utility, archive)


# ---------------------------------------------------------------------------
# Reading the table: best_entries, prune, rerank, ieu
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_best_entries_ranks_by_credit_and_ignores_other_artifact_kinds(tier, archive, search):
    prompt = genesis(id="sys", kind=ArtifactKind.TEXT, content="be helpful")
    await archive.record(Variant(artifact=prompt, parent=prompt, search_method="human"), GapMeasurement(score=1.0))
    a = await ops.add(tier, _entry("a"))
    b = await ops.add(tier, _entry("b"))
    await search.update(_trajectory([a]), reward=1.0)
    await search.update(_trajectory([b]), reward=0.0)

    best = await search.best_entries(k=5)

    assert [v.artifact.id for v in best] == ["memory.episodic.a", "memory.episodic.b"]
    assert all(v.artifact.kind == ArtifactKind.MEMORY_ENTRY for v in best)


@pytest.mark.asyncio
async def test_prune_drops_low_utility_rows_but_the_archive_keeps_every_version(tier, archive, search):
    keep = await ops.add(tier, _entry("keep", score=0.6))
    drop = await ops.add(tier, _entry("drop", score=0.3))
    await search.update(_trajectory([drop]), reward=0.0)

    dropped = await search.prune()

    assert dropped == 1
    assert await tier.get("drop") is None
    assert await tier.get("keep") is not None
    for v in (1, 2):
        await _version(archive, "memory.episodic.drop", v)
    assert keep.artifact_ref == ("memory.episodic.keep", 1)


@pytest.mark.asyncio
async def test_prune_accepts_an_explicit_floor(tier, search):
    await ops.add(tier, _entry("a", score=0.3))
    await ops.add(tier, _entry("b", score=0.7))

    assert await search.prune(min_utility=0.5) == 1
    assert await tier.get("b") is not None


@pytest.mark.asyncio
async def test_rerank_orders_recalled_entries_by_learned_utility(tier, search):
    low = await ops.add(tier, _entry("low", score=0.2))
    high = await ops.add(tier, _entry("high", score=0.2))
    mid = await ops.add(tier, _entry("mid", score=0.2))
    await search.update(_trajectory([high]), reward=1.0)
    await search.update(_trajectory([mid]), reward=0.6)

    # The tier returned them by relevance; the policy reorders by utility and
    # reads the archive, so stale scores on the recalled objects do not matter.
    stale = [MemoryEntry(id=e.id, scope_key=e.scope_key, content=e.content, score=0.0, artifact_ref=e.artifact_ref) for e in (low, high, mid)]
    ranked = await search.rerank(stale)

    assert [e.id for e in ranked] == ["high", "mid", "low"]


@pytest.mark.asyncio
async def test_rerank_is_stable_for_equal_utility(tier, search):
    a = await ops.add(tier, _entry("a", score=0.5))
    b = await ops.add(tier, _entry("b", score=0.5))
    assert [e.id for e in await search.rerank([b, a])] == ["b", "a"]


@pytest.mark.asyncio
async def test_ieu_reads_the_intent_experience_utility_triplet(tier, archive, search):
    entry = await ops.add(tier, _entry("e1", score=0.4, answer="use the portal"))
    await search.update(_trajectory([entry]), reward=1.0)

    triplet = ieu((await _version(archive, "memory.episodic.e1", 2)).artifact)

    assert triplet == {
        "intent": "question e1",
        "experience": "use the portal",
        "utility": pytest.approx(0.7),
        "artifact_ref": ["memory.episodic.e1", 2],
    }


def test_ieu_understands_insight_entries_and_explicit_triplet_keys():
    insight = genesis(
        id="memory.episodic.insight-1", kind=ArtifactKind.MEMORY_ENTRY,
        content={"tier": "episodic", "scope": "global:*", "entry": {"insight": "check 2FA first"}, "utility": 0.0},
    )
    assert ieu(insight)["experience"] == "check 2FA first"
    assert ieu(insight)["intent"] is None

    explicit = genesis(
        id="memory.episodic.x", kind=ArtifactKind.MEMORY_ENTRY,
        content={"tier": "episodic", "scope": "global:*", "entry": {"intent": "i", "experience": "x", "user_message": "ignored"}, "utility": 0.3},
    )
    assert ieu(explicit) == {"intent": "i", "experience": "x", "utility": 0.3, "artifact_ref": ["memory.episodic.x", 1]}
