"""GEPA's internal population evaluation (Option Y).

Tests that GEPA.propose() runs a full population x generations evolution
internally and yields only the winner.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from helix.archive import SQLiteArchive
from helix.artifact import ArtifactKind, genesis, Subtype
from helix.eval.dataset import EvalQuestion, EvalSet
from helix.eval.source import FixedEvalSet
from helix.search.base import SearchBudget, SearchKind
from helix.search.gepa import GEPA
from helix.signal import Cost, GapMeasurement, Preference, SignalKind
from helix.signals.reflection import Reflection


class _StubAgent:
    def __init__(self, prompt) -> None:
        self.system_prompt = prompt
        self.max_iterations = 10
        self.max_tool_calls = 20

    def with_artifacts(self, overrides):
        new_prompt = overrides.get(self.system_prompt.id, self.system_prompt)
        return _StubAgent(new_prompt)

    async def run(self, task: str):
        from helix.trajectory import Outcome, Trajectory
        t = Trajectory(task=task)
        t.complete(f"stub answer to: {task[:30]}", Outcome.COMPLETED)
        return t.final_output, t


def _make_agent(seed):
    return _StubAgent(seed)


class _StubSignal:
    """Always says candidate wins by a margin."""
    @property
    def kind(self):
        return SignalKind.LLM_JUDGE_PAIRWISE
    @property
    def cost_estimate(self):
        return Cost()
    async def measure(self, candidate, trajectory=None, reference=None, ground_truth=None):
        return GapMeasurement(
            score=1.0, preference=Preference.LEFT, feedback="stub fb", confidence=1.0, cost=Cost(tokens=10),
        )


def _eval_set() -> EvalSet:
    return EvalSet(questions=[
        EvalQuestion(id="Q1", band=1, question="x?", reference_answer="x"),
        EvalQuestion(id="Q2", band=2, question="y?", reference_answer="y"),
    ])


@pytest.mark.asyncio
async def test_gepa_requires_agent():
    with pytest.raises(ValueError):
        GEPA(eval_source=FixedEvalSet(_eval_set()))


@pytest.mark.asyncio
async def test_gepa_requires_eval_source():
    seed = genesis("prompt.test", Subtype.PROMPT, "seed content")
    with pytest.raises(ValueError):
        GEPA(agent=_make_agent(seed))


@pytest.mark.asyncio
async def test_gepa_yields_a_single_winner_after_internal_evolution():
    """GEPA.propose() should produce population_size x generations candidates
    internally, then yield ONE winner. The Improver model expects one
    candidate per round; GEPA's complexity stays internal."""
    archive = SQLiteArchive(":memory:")
    seed = genesis("prompt.test", Subtype.PROMPT, "seed content")

    # Patch _mutate_from and _crossover so we don't call the LLM. Content
    # includes the unique next version so candidates are distinct rows (real
    # GEPA produces distinct text; identical content would collapse, §1.3).
    async def fake_mutate(self, parent, parent_feedback, sibling, archive):
        next_v = await archive.next_version(parent.id) if hasattr(archive, "next_version") else parent.version + 1
        child = parent.mutate(
            new_content=f"mutated v{next_v} from v{parent.version}",
            created_by="gepa_mutation",
            version=next_v,
        )
        if hasattr(archive, "put_artifact"):
            await archive.put_artifact(child)
        return child, 0

    async def fake_crossover(self, a, b, archive):
        next_v = await archive.next_version(a.id) if hasattr(archive, "next_version") else a.version + 1
        child = a.mutate(
            new_content=f"crossover v{next_v} ({a.version} x {b.version})",
            created_by="gepa_crossover",
            version=next_v,
        )
        if hasattr(archive, "put_artifact"):
            await archive.put_artifact(child)
        return child, 0

    gepa = GEPA(
        proposer_model="claude-haiku-4-5",  # unused with patches
        reflector=Reflection(model="claude-haiku-4-5"),  # unused
        agent=_make_agent(seed),
        eval_source=FixedEvalSet(_eval_set()),
        population_size=3,
        generations=2,
        max_concurrent_questions=5,
    )

    with patch.object(GEPA, "_mutate_from", fake_mutate), patch.object(GEPA, "_crossover", fake_crossover):
        yielded = []
        async for variant in gepa.propose(
            seed=seed, signal=_StubSignal(), archive=archive, budget=SearchBudget()
        ):
            yielded.append(variant)

    # Exactly one variant yielded.
    assert len(yielded) == 1
    # Archive should contain seed plus generation-0 (3) plus generation-1 (3) = 7
    metrics = await archive.diversity_metrics()
    assert metrics.n_variants >= 6  # at least 6 mutations (seed isn't stored by mutate path)


@pytest.mark.asyncio
async def test_gepa_records_all_intermediate_candidates_to_archive():
    archive = SQLiteArchive(":memory:")
    seed = genesis("prompt.test", Subtype.PROMPT, "seed")

    async def fake_mutate(self, parent, parent_feedback, sibling, archive):
        next_v = await archive.next_version(parent.id)
        child = parent.mutate(new_content=f"v{next_v}", created_by="gepa_mutation", version=next_v)
        archive._store_artifact(child)
        archive._conn.commit()
        return child, 0

    async def fake_crossover(self, a, b, archive):
        next_v = await archive.next_version(a.id)
        child = a.mutate(new_content=f"xv{next_v}", created_by="gepa_crossover", version=next_v)
        archive._store_artifact(child)
        archive._conn.commit()
        return child, 0

    gepa = GEPA(
        proposer_model="claude-haiku-4-5",
        reflector=Reflection(model="claude-haiku-4-5"),
        agent=_make_agent(seed),
        eval_source=FixedEvalSet(_eval_set()),
        population_size=2,
        generations=2,
    )

    with patch.object(GEPA, "_mutate_from", fake_mutate), patch.object(GEPA, "_crossover", fake_crossover):
        async for _ in gepa.propose(seed=seed, signal=_StubSignal(), archive=archive, budget=SearchBudget()):
            pass

    # Every gen-0 and gen-1 candidate should have a measurement in the archive.
    measurements = archive._conn.execute("SELECT COUNT(*) AS c FROM measurements").fetchone()
    # 4 candidates (2 per gen * 2 gens) each got a candidate measurement recorded.
    assert measurements["c"] >= 4


@pytest.mark.asyncio
async def test_gepa_token_budget_halts_evolution():
    """A tight max_tokens cap stops GEPA mid-run: it now charges proposer and
    eval tokens to the budget, so budget.exhausted() trips (SPEC §4.4, fix #4).
    Before the fix GEPA never charged, so the cap was silently ignored."""
    archive = SQLiteArchive(":memory:")
    seed = genesis("prompt.test", Subtype.PROMPT, "seed")

    async def fake_mutate(self, parent, parent_feedback, sibling, archive):
        next_v = await archive.next_version(parent.id)
        child = parent.mutate(new_content=f"v{next_v}", created_by="gepa_mutation", version=next_v)
        archive._store_artifact(child)
        archive._conn.commit()
        return child, 5  # nonzero proposer tokens

    async def fake_crossover(self, a, b, archive):
        next_v = await archive.next_version(a.id)
        child = a.mutate(new_content=f"xv{next_v}", created_by="gepa_crossover", version=next_v)
        archive._store_artifact(child)
        archive._conn.commit()
        return child, 5

    # population 4 x generations 3 = up to 12 candidates with no cap.
    gepa = GEPA(
        proposer_model="claude-haiku-4-5",
        reflector=Reflection(model="claude-haiku-4-5"),
        agent=_make_agent(seed),
        eval_source=FixedEvalSet(_eval_set()),
        population_size=4,
        generations=3,
    )
    # Each candidate costs 5 (mutate) + 2 x 10 (signal per question) = 25 tokens.
    budget = SearchBudget(max_tokens=30)

    with patch.object(GEPA, "_mutate_from", fake_mutate), patch.object(GEPA, "_crossover", fake_crossover):
        async for _ in gepa.propose(seed=seed, signal=_StubSignal(), archive=archive, budget=budget):
            pass

    metrics = await archive.diversity_metrics()
    assert budget.exhausted()
    assert metrics.n_variants <= 6  # halted well short of the 12 it could run
