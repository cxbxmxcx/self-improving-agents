"""Improver lifecycle and integration tests.

These tests use stubs for Signal and Search so we don't need an LLM. The
lifecycle (start/stop/idempotent/trigger_round/status) and the round
orchestration are what matter here.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest

from helix.archive import SQLiteArchive
from helix.artifact import Artifact, ArtifactKind, genesis
from helix.eval.dataset import EvalQuestion, EvalSet
from helix.eval.source import FixedEvalSet
from helix.improvement import Improver, ImproverPolicy, Schedule
from helix.observability.bus import EventBus
from helix.search.base import (
    Search,
    SearchBudget,
    SearchCostModel,
    SearchKind,
    Variant,
)
from helix.signal import Cost, GapMeasurement, Preference, SignalKind


class _StubSignal:
    """Always says the candidate (LEFT) wins."""

    @property
    def kind(self):
        return SignalKind.LLM_JUDGE_PAIRWISE

    @property
    def cost_estimate(self):
        return Cost()

    async def measure(self, candidate, trajectory=None, reference=None, ground_truth=None):
        return GapMeasurement(
            score=1.0,
            preference=Preference.LEFT,
            feedback="stub: left wins",
            confidence=1.0,
            cost=Cost(),
        )


class _StubSearch:
    """Yields exactly one variant: parent.mutate('mutated')."""

    @property
    def kind(self):
        return SearchKind.PAIRWISE

    @property
    def cost_model(self):
        return SearchCostModel()

    async def propose(self, seed, signal, archive, budget) -> AsyncIterator[Variant]:
        child = seed.mutate("mutated content", created_by="stub_search")
        yield Variant(
            artifact=child,
            parent=seed,
            search_method=self.kind.value,
        )

    async def select(self, candidates, signal, archive):
        return candidates[0].artifact


class _StubAgent:
    """Has a .run() that produces a trivial trajectory."""

    def __init__(self, prompt: Artifact) -> None:
        self.prompt = prompt

    async def run(self, task: str):
        from helix.trajectory import Outcome, StepKind, Trajectory
        t = Trajectory(task=task)
        t.append(StepKind.MODEL_CALL, {"response": {"content": f"stub answer to: {task[:30]}"}})
        t.complete(f"stub answer to: {task[:30]}", Outcome.COMPLETED)
        return t.final_output, t


def _stub_build_agent_with_prompt(prompt: Artifact):
    async def go():
        return _StubAgent(prompt)
    return go()


def _eval_set() -> EvalSet:
    return EvalSet(questions=[
        EvalQuestion(id="Q1", band=1, question="What is X?", reference_answer="X is a thing."),
        EvalQuestion(id="Q2", band=2, question="What is Y?", reference_answer="Y is a thing."),
    ])


def _make_improver(*, schedule: Schedule = Schedule.MANUAL) -> tuple[Improver, SQLiteArchive, Artifact, EventBus]:
    bus = EventBus()
    archive = SQLiteArchive(":memory:")
    seed = genesis("prompt.test", ArtifactKind.PROMPT, "seed content")
    improver = Improver(
        target_artifact_id="prompt.test",
        signal=_StubSignal(),
        search=_StubSearch(),
        archive=archive,
        eval_source=FixedEvalSet(_eval_set()),
        build_agent_with_prompt=_stub_build_agent_with_prompt,
        policy=ImproverPolicy(schedule=schedule),
        seed_fallback=seed,
        bus=bus,
    )
    return improver, archive, seed, bus


@pytest.mark.asyncio
async def test_improver_start_is_idempotent():
    imp, _, _, _ = _make_improver()
    await imp.start()
    first_task = imp._task
    await imp.start()
    assert imp._task is first_task  # no second task spawned
    await imp.stop()


@pytest.mark.asyncio
async def test_trigger_round_records_both_measurements():
    imp, archive, seed, _ = _make_improver()
    await imp.start()
    try:
        result = await imp.trigger_round()
    finally:
        await imp.stop()

    assert result.candidate_score == 1.0
    assert result.reference_score == 0.0
    assert result.promoted is True

    # archive should now have seed AND child, both with measurements
    top = await archive.best(k=2)
    versions = sorted(v.artifact.version for v in top)
    assert versions == [1, 2]


@pytest.mark.asyncio
async def test_status_reflects_round_state():
    imp, _, _, _ = _make_improver()
    await imp.start()
    try:
        await imp.trigger_round()
        s = imp.status
        assert s.rounds_completed == 1
        assert s.last_round_result is not None
        assert s.running is True
    finally:
        await imp.stop()
    assert imp.status.running is False


@pytest.mark.asyncio
async def test_events_emitted_during_round():
    imp, _, _, bus = _make_improver()
    received: list[str] = []
    bus.subscribe("*", lambda e: received.append(e.event_type))
    await imp.start()
    try:
        await imp.trigger_round()
    finally:
        await imp.stop()

    assert "improver_round_started" in received
    assert "improver_round_completed" in received
    assert "pair_pass_question_started" in received
    assert "pair_pass_question_completed" in received
    assert "judge_question_completed" in received


@pytest.mark.asyncio
async def test_interval_schedule_fires_rounds_automatically():
    imp, _, _, _ = _make_improver(schedule=Schedule.INTERVAL)
    imp.policy.interval_sec = 0.05  # fast for testing
    await imp.start()
    await asyncio.sleep(0.3)  # several intervals
    await imp.stop()
    assert imp.status.rounds_completed >= 1
