"""StrategyChain behavior under various failure/promotion sequences."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass

import pytest

from helix.artifact import ArtifactKind, genesis
from helix.observability.bus import EventBus
from helix.observability.events import SearchStrategySwitched
from helix.search.base import (
    SearchBudget,
    SearchCostModel,
    SearchKind,
    Variant,
)
from helix.search.strategy_chain import StrategyChain


class _StubSearch:
    """A minimal Search that yields one mutation per propose() call."""

    def __init__(self, label: str, kind: SearchKind = SearchKind.PAIRWISE) -> None:
        self.label = label
        self._kind = kind
        self.calls = 0

    @property
    def kind(self) -> SearchKind:
        return self._kind

    @property
    def cost_model(self) -> SearchCostModel:
        return SearchCostModel()

    async def propose(self, seed, signal, archive, budget) -> AsyncIterator[Variant]:
        self.calls += 1
        child = seed.mutate(f"mutated by {self.label}", created_by=self.label)
        yield Variant(artifact=child, parent=seed, search_method=self.label)

    async def select(self, candidates, signal, archive):
        return candidates[0].artifact


@dataclass
class _StubRoundResult:
    promoted: bool


def _seed():
    return genesis("prompt.test", ArtifactKind.PROMPT, "seed content")


@pytest.mark.asyncio
async def test_strategy_chain_starts_on_first_strategy():
    a = _StubSearch("A", SearchKind.PAIRWISE)
    b = _StubSearch("B", SearchKind.GENETIC_PARETO)
    chain = StrategyChain([a, b], max_failures_per_strategy=1)
    assert chain.active_kind == SearchKind.PAIRWISE.value
    assert not chain.all_retired


@pytest.mark.asyncio
async def test_strategy_chain_delegates_propose_to_active():
    a = _StubSearch("A")
    b = _StubSearch("B")
    chain = StrategyChain([a, b])
    seed = _seed()
    variants = []
    async for v in chain.propose(seed=seed, signal=None, archive=None, budget=SearchBudget()):
        variants.append(v)
    assert a.calls == 1
    assert b.calls == 0
    assert len(variants) == 1
    assert variants[0].search_method == "A"


@pytest.mark.asyncio
async def test_strategy_chain_rotates_after_first_failure_when_max_is_one():
    bus = EventBus()
    a = _StubSearch("A", kind=SearchKind.PAIRWISE)
    b = _StubSearch("B", kind=SearchKind.GENETIC_PARETO)
    chain = StrategyChain([a, b], max_failures_per_strategy=1, bus=bus)

    switches = []
    bus.subscribe("search_strategy_switched", lambda e: switches.append(e))

    # max_failures=1 means "retire on the 1st failure" — rotate immediately.
    await chain.on_round_result(_StubRoundResult(promoted=False))
    assert chain.active_kind == SearchKind.GENETIC_PARETO.value
    assert len(switches) == 1
    assert switches[0].from_kind == SearchKind.PAIRWISE.value


@pytest.mark.asyncio
async def test_strategy_chain_tolerates_failures_below_max():
    a = _StubSearch("A", kind=SearchKind.PAIRWISE)
    b = _StubSearch("B", kind=SearchKind.GENETIC_PARETO)
    chain = StrategyChain([a, b], max_failures_per_strategy=3)

    # First two failures stay on A.
    await chain.on_round_result(_StubRoundResult(promoted=False))
    assert chain.active_kind == SearchKind.PAIRWISE.value
    await chain.on_round_result(_StubRoundResult(promoted=False))
    assert chain.active_kind == SearchKind.PAIRWISE.value
    # Third failure rotates.
    await chain.on_round_result(_StubRoundResult(promoted=False))
    assert chain.active_kind == SearchKind.GENETIC_PARETO.value


@pytest.mark.asyncio
async def test_strategy_chain_resets_failures_on_promote():
    a = _StubSearch("A", kind=SearchKind.PAIRWISE)
    b = _StubSearch("B", kind=SearchKind.GENETIC_PARETO)
    # max=2: takes 2 consecutive failures to rotate. A promote in between
    # resets the counter.
    chain = StrategyChain([a, b], max_failures_per_strategy=2)

    await chain.on_round_result(_StubRoundResult(promoted=False))  # failures = 1
    await chain.on_round_result(_StubRoundResult(promoted=True))   # reset to 0
    await chain.on_round_result(_StubRoundResult(promoted=False))  # failures = 1, still on A
    assert chain.active_kind == SearchKind.PAIRWISE.value


@pytest.mark.asyncio
async def test_strategy_chain_retires_all_then_yields_nothing():
    bus = EventBus()
    a = _StubSearch("A")
    b = _StubSearch("B")
    chain = StrategyChain([a, b], max_failures_per_strategy=1, bus=bus)

    # max_failures=1: 1 failure retires the strategy.
    await chain.on_round_result(_StubRoundResult(promoted=False))  # retires A
    await chain.on_round_result(_StubRoundResult(promoted=False))  # retires B
    assert chain.all_retired

    # Propose now yields nothing.
    yielded = []
    async for v in chain.propose(seed=_seed(), signal=None, archive=None, budget=SearchBudget()):
        yielded.append(v)
    assert yielded == []


def test_strategy_chain_rejects_max_failures_below_one():
    a = _StubSearch("A")
    with pytest.raises(ValueError):
        StrategyChain([a], max_failures_per_strategy=0)


@pytest.mark.asyncio
async def test_strategy_chain_status_reports_per_strategy_state():
    a = _StubSearch("A")
    b = _StubSearch("B")
    chain = StrategyChain([a, b], max_failures_per_strategy=3)

    await chain.on_round_result(_StubRoundResult(promoted=False))
    status = chain.status()
    assert status["active_idx"] == 0
    assert status["strategies"][0]["failures"] == 1
    assert status["strategies"][0]["retired"] is False
    assert status["strategies"][1]["failures"] == 0


@pytest.mark.asyncio
async def test_strategy_chain_tags_variant_metadata():
    a = _StubSearch("A")
    chain = StrategyChain([a])
    seed = _seed()
    async for v in chain.propose(seed=seed, signal=None, archive=None, budget=SearchBudget()):
        assert v.metadata["strategy_chain_kind"] == SearchKind.PAIRWISE.value
        assert v.metadata["strategy_chain_idx"] == 0
