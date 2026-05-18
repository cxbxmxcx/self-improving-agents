"""Spec conformance for the Trajectory primitive (Spec §2)."""

from __future__ import annotations

from helix.trajectory import Outcome, StepKind, Trajectory


def test_new_trajectory_starts_in_progress_with_no_steps():
    t = Trajectory(task="hello")
    assert t.outcome == Outcome.IN_PROGRESS
    assert t.steps == []
    assert t.ended_at is None


def test_append_step_returns_step_with_correct_index():
    t = Trajectory(task="hello")
    s1 = t.append(StepKind.MODEL_CALL, {"x": 1})
    s2 = t.append(StepKind.TOOL_CALL, {"y": 2})
    assert s1.index == 0
    assert s2.index == 1
    assert len(t.steps) == 2


def test_record_artifact_associates_ref_with_step_index():
    t = Trajectory(task="hello")
    t.append(StepKind.MODEL_CALL, {})
    t.record_artifact(0, ("prompt.test", 1))
    t.record_artifact(0, ("prompt.other", 3))
    assert t.artifacts_used[0] == [("prompt.test", 1), ("prompt.other", 3)]


def test_complete_sets_outcome_output_and_end_time():
    t = Trajectory(task="hello")
    t.complete("the answer")
    assert t.outcome == Outcome.COMPLETED
    assert t.final_output == "the answer"
    assert t.ended_at is not None
