"""Multiple Improvers on the same Agent targeting the same artifact."""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from helix.archive import SQLiteArchive
from helix.artifact import Artifact, ArtifactKind, genesis
from helix.eval.dataset import EvalQuestion, EvalSet
from helix.eval.source import FixedEvalSet
from helix.improvement import Improver, ImproverPolicy, Schedule
from helix.observability.bus import EventBus
from helix.search.base import SearchCostModel, SearchKind, Variant
from helix.signal import Cost, GapMeasurement, Preference, SignalKind


class _StubSignal:
    @property
    def kind(self):
        return SignalKind.LLM_JUDGE_PAIRWISE

    @property
    def cost_estimate(self):
        return Cost()

    async def measure(self, candidate, trajectory=None, reference=None, ground_truth=None):
        return GapMeasurement(
            score=1.0, preference=Preference.LEFT, feedback="stub", confidence=1.0, cost=Cost()
        )


class _StubSearch:
    def __init__(self, kind: SearchKind, label: str) -> None:
        self._kind = kind
        self.label = label

    @property
    def kind(self):
        return self._kind

    @property
    def cost_model(self):
        return SearchCostModel()

    async def propose(self, seed, signal, archive, budget) -> AsyncIterator[Variant]:
        child = seed.mutate(f"mutated by {self.label}", created_by=self.label)
        yield Variant(artifact=child, parent=seed, search_method=self._kind.value)

    async def select(self, candidates, signal, archive):
        return candidates[0].artifact


class _StubAgent:
    def __init__(self, prompt: Artifact) -> None:
        self.prompt = prompt

    async def run(self, task: str):
        from helix.trajectory import Outcome, Trajectory
        t = Trajectory(task=task)
        t.complete(f"answer to {task[:20]}", Outcome.COMPLETED)
        return t.final_output, t


def _build_agent(prompt: Artifact):
    async def go():
        return _StubAgent(prompt)
    return go()


def _eval_set() -> EvalSet:
    return EvalSet(questions=[
        EvalQuestion(id="Q1", band=1, question="X?", reference_answer="X."),
    ])


def _make_improver(*, search_kind: SearchKind, label: str) -> Improver:
    archive = SQLiteArchive(":memory:")
    seed = genesis("prompt.shared", ArtifactKind.PROMPT, "seed")
    return Improver(
        target_artifact_id="prompt.shared",
        signal=_StubSignal(),
        search=_StubSearch(search_kind, label),
        archive=archive,
        eval_source=FixedEvalSet(_eval_set()),
        build_agent_with_prompt=_build_agent,
        policy=ImproverPolicy(schedule=Schedule.MANUAL),
        seed_fallback=seed,
        improver_id=f"imp-{label}",
    )


@pytest.mark.asyncio
async def test_two_improvers_same_target_both_attach():
    """The Agent should accept multiple Improvers targeting the same artifact
    when they have distinct improver_ids."""
    from helix.agent import Agent
    art = genesis("prompt.shared", ArtifactKind.PROMPT, "seed")
    agent = Agent(system_prompt=art, model="claude-haiku-4-5")

    imp_spo = _make_improver(search_kind=SearchKind.PAIRWISE, label="spo")
    imp_gepa = _make_improver(search_kind=SearchKind.GENETIC_PARETO, label="gepa")

    agent.attach_improver(imp_spo)
    agent.attach_improver(imp_gepa)

    assert len(agent.improvers) == 2
    assert {i.improver_id for i in agent.improvers} == {"imp-spo", "imp-gepa"}


@pytest.mark.asyncio
async def test_attach_improver_idempotent_on_improver_id():
    from helix.agent import Agent
    art = genesis("prompt.shared", ArtifactKind.PROMPT, "seed")
    agent = Agent(system_prompt=art, model="claude-haiku-4-5")

    imp = _make_improver(search_kind=SearchKind.PAIRWISE, label="spo")
    agent.attach_improver(imp)
    agent.attach_improver(imp)  # second attach is a no-op

    assert len(agent.improvers) == 1


@pytest.mark.asyncio
async def test_detach_by_improver_id():
    from helix.agent import Agent
    art = genesis("prompt.shared", ArtifactKind.PROMPT, "seed")
    agent = Agent(system_prompt=art, model="claude-haiku-4-5")

    imp_spo = _make_improver(search_kind=SearchKind.PAIRWISE, label="spo")
    imp_gepa = _make_improver(search_kind=SearchKind.GENETIC_PARETO, label="gepa")
    agent.attach_improver(imp_spo)
    agent.attach_improver(imp_gepa)

    agent.detach_improver("imp-spo")
    assert len(agent.improvers) == 1
    assert agent.improvers[0].improver_id == "imp-gepa"


@pytest.mark.asyncio
async def test_two_improvers_can_share_an_archive():
    """When both Improvers use the same SQLiteArchive, both rounds record
    into it. The archive's best() then returns the overall highest scorer.

    Each round records 2 measurements (candidate + reference mirror). After
    two rounds we expect 4 measurements total in the archive across 3
    artifacts (the shared seed + candidate from each improver).
    """
    archive = SQLiteArchive(":memory:")
    seed = genesis("prompt.shared", ArtifactKind.PROMPT, "seed")

    # Both improvers say their candidate wins (LEFT) by a hefty margin.
    class _WinningSignal:
        @property
        def kind(self):
            return SignalKind.LLM_JUDGE_PAIRWISE
        @property
        def cost_estimate(self):
            return Cost()
        async def measure(self, candidate, trajectory=None, reference=None, ground_truth=None):
            return GapMeasurement(
                score=1.0, preference=Preference.LEFT, confidence=1.0, cost=Cost()
            )

    def make(label: str, kind: SearchKind) -> Improver:
        return Improver(
            target_artifact_id="prompt.shared",
            signal=_WinningSignal(),
            search=_StubSearch(kind, label),
            archive=archive,
            eval_source=FixedEvalSet(_eval_set()),
            build_agent_with_prompt=_build_agent,
            policy=ImproverPolicy(schedule=Schedule.MANUAL),
            seed_fallback=seed,
            improver_id=f"imp-{label}",
        )

    imp_spo = make("spo", SearchKind.PAIRWISE)
    imp_gepa = make("gepa", SearchKind.GENETIC_PARETO)

    await imp_spo.start()
    await imp_gepa.start()
    try:
        await imp_spo.trigger_round()
        await imp_gepa.trigger_round()
    finally:
        await imp_spo.stop()
        await imp_gepa.stop()

    # Both improvers' candidates should be in the archive. Because the
    # second improver judges against whatever archive.best() returned (which
    # is the first improver's winner), the first improver's candidate gets
    # re-measured as the new "reference" and its score reflects the second
    # round's mirror. The architectural guarantee is that all three artifacts
    # are present and that the second improver's winner has score 1.0.
    metrics = await archive.diversity_metrics()
    assert metrics.n_variants >= 3  # genesis + 2 candidates from two searches
    top = await archive.best(k=1)
    assert top[0].measurement.score == 1.0
    # The overall champion should be one of the candidates, not the genesis.
    assert top[0].artifact.created_by in {"spo", "gepa"}
