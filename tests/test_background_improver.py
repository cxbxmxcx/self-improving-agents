"""BackgroundImproverRunner tests.

Uses stub Improver internals (no LLM calls). Verifies thread lifecycle,
registry, pause/resume, and trigger_round behavior across threads.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator

import pytest

from helix.archive import SQLiteArchive
from helix.artifact import ArtifactKind, genesis, Subtype
from helix.eval.dataset import EvalQuestion, EvalSet
from helix.eval.source import FixedEvalSet
from helix.hooks import HookRegistry
from helix.improvement import OfflineImprover, ImproverPolicy, Schedule
from helix.improvement.background import (
    BackgroundImproverRunner,
    attach_runner,
    detach_runner,
    get_runner,
    list_runners,
    stop_all_runners,
)
from helix.search.base import SearchCostModel, SearchKind, Variant
from helix.signal import Cost, GapMeasurement, Preference, SignalKind


# ---------------- stubs ----------------

class _StubSignal:
    @property
    def kind(self): return SignalKind.LLM_JUDGE_PAIRWISE
    @property
    def cost_estimate(self): return Cost()
    async def measure(self, candidate, trajectory=None, reference=None, ground_truth=None):
        return GapMeasurement(score=1.0, preference=Preference.LEFT, confidence=1.0, cost=Cost())


class _StubSearch:
    @property
    def kind(self): return SearchKind.PAIRWISE
    @property
    def cost_model(self): return SearchCostModel()
    async def propose(self, seed, signal, archive, budget) -> AsyncIterator[Variant]:
        child = seed.mutate("mutated", created_by="stub")
        yield Variant(artifact=child, parent=seed, search_method=self.kind.value)
    async def select(self, c, s, a): return c[0].artifact


class _StubAgent:
    def __init__(self, prompt):
        self.system_prompt = prompt
        self.hooks = HookRegistry()
        self.max_iterations = 10
        self.max_tool_calls = 20

    def with_artifacts(self, overrides):
        new_prompt = overrides.get(self.system_prompt.id, self.system_prompt)
        return _StubAgent(new_prompt)

    async def run(self, task, context=None):
        from helix.trajectory import Outcome, Trajectory
        t = Trajectory(task=task); t.complete("ok", Outcome.COMPLETED); return "ok", t


def _make_improver(improver_id: str, schedule: Schedule = Schedule.MANUAL) -> OfflineImprover:
    seed = genesis("p", Subtype.PROMPT, "seed")
    es = EvalSet(questions=[EvalQuestion(id="Q1", band=1, question="?", reference_answer="x")])
    return OfflineImprover(
        agent=_StubAgent(seed),
        target_artifact_id="p",
        signal=_StubSignal(),
        search=_StubSearch(),
        archive=SQLiteArchive(":memory:", check_same_thread=False),
        eval_source=FixedEvalSet(es),
        policy=ImproverPolicy(schedule=schedule),
        seed_fallback=seed,
        improver_id=improver_id,
    )


# ---------------- tests ----------------

@pytest.fixture(autouse=True)
def cleanup_runners():
    """Drop any leftover runners between tests so the registry starts clean."""
    stop_all_runners()
    yield
    stop_all_runners()


def test_start_spawns_alive_thread():
    runner = BackgroundImproverRunner(_make_improver("test-1"))
    runner.start()
    try:
        assert runner.status().thread_alive is True
    finally:
        runner.stop()


def test_start_is_idempotent():
    runner = BackgroundImproverRunner(_make_improver("test-2"))
    runner.start()
    thread1 = runner._thread
    runner.start()  # no-op
    thread2 = runner._thread
    try:
        assert thread1 is thread2
    finally:
        runner.stop()


def test_stop_joins_thread():
    runner = BackgroundImproverRunner(_make_improver("test-3"))
    runner.start()
    assert runner.status().thread_alive
    runner.stop()
    time.sleep(0.5)
    assert runner.status().thread_alive is False


def test_trigger_round_returns_future_with_result():
    runner = BackgroundImproverRunner(_make_improver("test-4"))
    runner.start()
    try:
        future = runner.trigger_round()
        result = future.result(timeout=10.0)
        assert result.candidate_score == 1.0
        assert runner.status().rounds_completed == 1
    finally:
        runner.stop()


def test_pause_sets_paused_flag():
    runner = BackgroundImproverRunner(_make_improver("test-5"))
    runner.start()
    try:
        assert not runner.is_paused
        runner.pause()
        assert runner.is_paused
        # Improver's internal flag also flipped
        assert runner.improver._paused is True
        runner.resume()
        assert not runner.is_paused
        assert runner.improver._paused is False
    finally:
        runner.stop()


def test_trigger_round_before_start_raises():
    runner = BackgroundImproverRunner(_make_improver("test-6"))
    with pytest.raises(RuntimeError):
        runner.trigger_round()


def test_attach_runner_creates_and_registers():
    runner = attach_runner(_make_improver("test-7"))
    try:
        assert "test-7" in list_runners()
        assert get_runner("test-7") is runner
    finally:
        detach_runner("test-7")


def test_attach_runner_is_idempotent_per_id():
    """Second attach with same improver_id returns the existing runner."""
    runner1 = attach_runner(_make_improver("test-8"))
    runner1.start()
    try:
        runner2 = attach_runner(_make_improver("test-8"))
        assert runner1 is runner2
    finally:
        detach_runner("test-8")


def test_detach_runner_stops_and_removes():
    runner = attach_runner(_make_improver("test-9"))
    runner.start()
    detach_runner("test-9")
    time.sleep(0.5)
    assert "test-9" not in list_runners()
    assert get_runner("test-9") is None


def test_status_reflects_initial_state():
    runner = BackgroundImproverRunner(_make_improver("test-10"))
    s = runner.status()
    assert s.improver_id == "test-10"
    assert s.target_artifact_id == "p"
    assert s.rounds_completed == 0
    assert s.thread_alive is False


def test_status_reflects_running_state_after_start():
    runner = BackgroundImproverRunner(_make_improver("test-11"))
    runner.start()
    try:
        s = runner.status()
        assert s.thread_alive is True
        assert s.running is True
        assert s.paused is False
    finally:
        runner.stop()
