"""Tests for DGMSearch (SPEC section 4.2, Ch 3)."""

from __future__ import annotations

import pytest

from helix.archive import SQLiteArchive
from helix.artifact import ArtifactKind, genesis, Subtype
from helix.search.base import SearchBudget, SearchKind, Variant
from helix.search.dgm import BlindLLMMutator, DGMSearch, Mutator
from helix.signal import Cost, GapMeasurement, Preference, SignalKind


class _StubSignal:
    @property
    def kind(self):
        return SignalKind.LLM_JUDGE_ABSOLUTE
    @property
    def cost_estimate(self):
        return Cost()
    @property
    def signal_id(self):
        return "stub-signal"
    @property
    def signal_version(self):
        return 1
    async def measure(self, candidate, trajectory=None, reference=None, ground_truth=None):
        return GapMeasurement(score=0.5, confidence=1.0, cost=Cost())


class _StubMutator:
    """Deterministic mutator: appends a marker to the seed content. No LLM."""
    def __init__(self):
        self.calls = 0

    async def mutate(self, seed, last_feedback, signal):
        self.calls += 1
        base = seed.content if isinstance(seed.content, str) else str(seed.content)
        return f"{base} [mutated #{self.calls}]", Cost(tokens=10)


def test_dgm_has_correct_kind():
    dgm = DGMSearch(mutator=_StubMutator())
    assert dgm.kind == SearchKind.EVOLUTIONARY_ARCHIVE


def test_blind_mutator_satisfies_protocol():
    assert isinstance(BlindLLMMutator(), Mutator)


def test_stub_mutator_satisfies_protocol():
    assert isinstance(_StubMutator(), Mutator)


@pytest.mark.asyncio
async def test_dgm_proposes_a_candidate_from_seed_when_archive_empty():
    """With an empty archive, the seed falls back to the propose() seed."""
    archive = SQLiteArchive(":memory:")
    seed = genesis("tool_description.x", Subtype.TOOL_DESCRIPTION, "find docs")
    dgm = DGMSearch(mutator=_StubMutator(), rounds=1)

    yielded = []
    async for variant in dgm.propose(
        seed=seed, signal=_StubSignal(), archive=archive, budget=SearchBudget()
    ):
        yielded.append(variant)

    assert len(yielded) == 1
    assert "[mutated #1]" in yielded[0].artifact.content
    # Parent is the seed (the only thing available with an empty archive).
    assert yielded[0].parent.id == seed.id


@pytest.mark.asyncio
async def test_dgm_samples_from_archive_history():
    """With measured variants in the archive, DGM samples a seed from history
    (not just the propose() seed) and the candidate's parent is that sample."""
    archive = SQLiteArchive(":memory:")
    # Record three measured variants of the same artifact id.
    a1 = genesis("tool_description.x", Subtype.TOOL_DESCRIPTION, "v1")
    a2 = a1.mutate("v2", created_by="seed")
    a3 = a2.mutate("v3", created_by="seed")
    for art, score in [(a1, 0.9), (a2, 0.3), (a3, 0.4)]:
        await archive.record(
            Variant(artifact=art, parent=a1, search_method="seed"),
            GapMeasurement(score=score, signal_id="stub-signal"),
        )

    dgm = DGMSearch(mutator=_StubMutator(), rounds=1)
    yielded = []
    async for variant in dgm.propose(
        seed=a3, signal=_StubSignal(), archive=archive, budget=SearchBudget()
    ):
        yielded.append(variant)

    assert len(yielded) == 1
    # The candidate's parent should be one of the archived variants (sampled
    # from history), and the candidate content derives from it.
    parent_versions = {a1.version, a2.version, a3.version}
    assert yielded[0].parent.version in parent_versions
    assert "[mutated" in yielded[0].artifact.content


@pytest.mark.asyncio
async def test_dgm_candidate_carries_lineage_to_sampled_parent():
    archive = SQLiteArchive(":memory:")
    a1 = genesis("tool_description.x", Subtype.TOOL_DESCRIPTION, "v1")
    await archive.record(
        Variant(artifact=a1, parent=a1, search_method="seed"),
        GapMeasurement(score=0.8, signal_id="stub-signal"),
    )

    dgm = DGMSearch(mutator=_StubMutator(), rounds=1)
    yielded = []
    async for variant in dgm.propose(
        seed=a1, signal=_StubSignal(), archive=archive, budget=SearchBudget()
    ):
        yielded.append(variant)

    candidate = yielded[0].artifact
    # The candidate's parent_id points at the sampled seed (a1).
    assert candidate.parent_id == a1.ref


@pytest.mark.asyncio
async def test_dgm_respects_rounds():
    archive = SQLiteArchive(":memory:")
    seed = genesis("tool_description.x", Subtype.TOOL_DESCRIPTION, "v1")
    dgm = DGMSearch(mutator=_StubMutator(), rounds=3)

    count = 0
    async for _variant in dgm.propose(
        seed=seed, signal=_StubSignal(), archive=archive, budget=SearchBudget()
    ):
        count += 1
    assert count == 3


@pytest.mark.asyncio
async def test_dgm_select_returns_highest_scoring():
    dgm = DGMSearch(mutator=_StubMutator())
    a = genesis("x", Subtype.TOOL_DESCRIPTION, "a")
    b = a.mutate("b", created_by="dgm")
    candidates = [
        Variant(artifact=a, parent=a, search_method="dgm",
                measurement=GapMeasurement(score=0.3)),
        Variant(artifact=b, parent=a, search_method="dgm",
                measurement=GapMeasurement(score=0.7)),
    ]
    winner = await dgm.select(candidates, _StubSignal(), SQLiteArchive(":memory:"))
    assert winner.content == "b"
