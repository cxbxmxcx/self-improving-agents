"""EvalSet loader and EvalSource implementations."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from helix.eval.dataset import EvalQuestion, EvalSet, load_eval_set
from helix.eval.source import FixedEvalSet, RecentTrajectorySource
from helix.observability.bus import EventBus
from helix.observability.events import TrajectoryCompleted


REPO_QUESTIONS = Path(__file__).resolve().parent.parent / "chapters" / "ch02" / "eval_questions.json"


def test_load_eval_set_from_repo_json():
    es = load_eval_set(REPO_QUESTIONS)
    assert len(es) == 20
    assert {q.band for q in es.questions} == {1, 2, 3, 4}
    assert es.by_id("Q1") is not None


def test_eval_set_stratified_sample_covers_all_bands():
    es = load_eval_set(REPO_QUESTIONS)
    sample = es.sample(n=8, stratified=True, seed=42)
    assert {q.band for q in sample} == {1, 2, 3, 4}


@pytest.mark.asyncio
async def test_fixed_eval_set_returns_all_questions_when_n_none():
    es = EvalSet(questions=[
        EvalQuestion(id="Q1", band=1, question="a", reference_answer="A"),
        EvalQuestion(id="Q2", band=2, question="b", reference_answer="B"),
    ])
    src = FixedEvalSet(es)
    qs = await src.get_questions(n=None)
    assert [q.id for q in qs] == ["Q1", "Q2"]


@pytest.mark.asyncio
async def test_recent_trajectory_source_buffers_from_event_bus():
    bus = EventBus()
    src = RecentTrajectorySource(buffer_size=10, bus=bus)
    assert src.size == 0
    await bus.publish(TrajectoryCompleted(trajectory_id="t1", task="What is X?", outcome="completed", num_steps=3, final_output="answer 1"))
    await bus.publish(TrajectoryCompleted(trajectory_id="t2", task="What is Y?", outcome="completed", num_steps=2, final_output="answer 2"))
    assert src.size == 2
    qs = await src.get_questions(n=None)
    assert {q.id for q in qs} == {"live::t1", "live::t2"}


@pytest.mark.asyncio
async def test_recent_trajectory_source_skips_non_completed_outcomes():
    bus = EventBus()
    src = RecentTrajectorySource(buffer_size=10, bus=bus)
    await bus.publish(TrajectoryCompleted(trajectory_id="t1", task="hello", outcome="timed_out", num_steps=10, final_output=None))
    assert src.size == 0


@pytest.mark.asyncio
async def test_recent_trajectory_source_buffer_evicts_oldest():
    bus = EventBus()
    src = RecentTrajectorySource(buffer_size=2, bus=bus)
    for i in range(5):
        await bus.publish(TrajectoryCompleted(trajectory_id=f"t{i}", task=f"task {i}", outcome="completed", num_steps=1, final_output="x"))
    assert src.size == 2
