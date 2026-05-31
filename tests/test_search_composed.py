"""ComposedSearch: coordinate-descent joint search over a composite. SPEC §18.4."""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from helix.artifact import ArtifactKind, Subtype, compose, genesis
from helix.search.base import SearchBudget, SearchCostModel, SearchKind, Variant
from helix.search.composed import ComposedSearch
from helix.signal import Cost, GapMeasurement


class _BumpSearch:
    """Child search that yields one mutated variant tagging the role it serves."""

    def __init__(self, tag: str) -> None:
        self.tag = tag

    @property
    def kind(self):
        return SearchKind.PAIRWISE

    @property
    def cost_model(self):
        return SearchCostModel()

    async def propose(self, seed, signal, archive, budget) -> AsyncIterator[Variant]:
        child = seed.mutate(f"{seed.content}+{self.tag}", created_by=f"bump:{self.tag}")
        yield Variant(artifact=child, parent=seed, search_method=self.tag)

    async def select(self, candidates, signal, archive):
        return candidates[0].artifact


class _NullSignal:
    @property
    def kind(self):
        return SearchKind.PAIRWISE  # unused

    @property
    def cost_estimate(self):
        return Cost()

    async def measure(self, candidate, trajectory=None, reference=None, ground_truth=None):
        return GapMeasurement(score=1.0, cost=Cost())


def _meta_composite():
    planner = genesis("plan", Subtype.PLANNER, "P0")
    monitor = genesis("mon", Subtype.MONITOR, "M0")
    mem = genesis("mem", ArtifactKind.MEMORY_ENTRY, {"s": 0})
    comp = compose(
        "meta",
        [(planner, "planner"), (monitor, "monitor"), (mem, "memory")],
        subtype=Subtype.METACOGNITION,
    )
    return comp, {"planner": planner, "monitor": monitor, "memory": mem}


@pytest.mark.asyncio
async def test_propose_bumps_one_constituent_and_versions_the_composite():
    comp, current = _meta_composite()
    search = ComposedSearch(
        {"planner": _BumpSearch("planner"), "monitor": _BumpSearch("monitor")},
        current,
    )
    budget = SearchBudget()

    variants = [v async for v in search.propose(comp, _NullSignal(), None, budget)]
    assert len(variants) == 1
    new_comp = variants[0].artifact

    # The composite is a new version, still a metacognition composite at L3.
    assert new_comp.kind is ArtifactKind.COMPOSITE
    assert new_comp.subtype is Subtype.METACOGNITION
    assert new_comp.version == 2
    assert new_comp.parent_id == ("meta", 1)
    assert new_comp.layer == 3

    # Exactly the planner constituent was bumped (round-robin starts at index 0).
    changed = variants[0].metadata["changed_role"]
    assert changed == "planner"
    refs = dict((c["role"], (c["id"], c["version"])) for c in new_comp.constituents)
    assert refs["planner"] == ("plan", 2)   # bumped
    assert refs["monitor"] == ("mon", 1)    # held
    assert refs["memory"] == ("mem", 1)     # held (no child search)


@pytest.mark.asyncio
async def test_round_robin_advances_to_next_constituent():
    comp, current = _meta_composite()
    search = ComposedSearch(
        {"planner": _BumpSearch("planner"), "monitor": _BumpSearch("monitor")},
        current,
    )
    budget = SearchBudget()

    first = [v async for v in search.propose(comp, _NullSignal(), None, budget)][0]
    second = [v async for v in search.propose(comp, _NullSignal(), None, budget)][0]

    assert first.metadata["changed_role"] == "planner"
    assert second.metadata["changed_role"] == "monitor"


@pytest.mark.asyncio
async def test_select_advances_the_accepted_constituent_champion():
    comp, current = _meta_composite()
    search = ComposedSearch({"planner": _BumpSearch("planner")}, current)
    budget = SearchBudget()

    variant = [v async for v in search.propose(comp, _NullSignal(), None, budget)][0]
    variant.measurement = GapMeasurement(score=0.9, cost=Cost())

    winner = await search.select([variant], _NullSignal(), None)
    assert winner.version == 2
    # The planner champion advanced to v2, so the next round seeds from it.
    assert search.current_constituent("planner").version == 2
    assert search.current_constituent("planner").content == "P0+planner"


@pytest.mark.asyncio
async def test_roles_without_a_child_search_are_held_fixed():
    comp, current = _meta_composite()
    # Only the memory role has a child search; planner/monitor are held.
    search = ComposedSearch({"memory": _BumpSearch("memory")}, current)
    budget = SearchBudget()

    variants = [v async for v in search.propose(comp, _NullSignal(), None, budget)]
    assert variants[0].metadata["changed_role"] == "memory"
    refs = dict((c["role"], (c["id"], c["version"])) for c in variants[0].artifact.constituents)
    assert refs["memory"] == ("mem", 2)
    assert refs["planner"] == ("plan", 1)


@pytest.mark.asyncio
async def test_composed_select_raises_on_empty_candidates():
    """select() promises -> Artifact, so empty input raises rather than
    returning None (fix #9)."""
    search = ComposedSearch({}, {})
    with pytest.raises(ValueError):
        await search.select([], _NullSignal(), None)
