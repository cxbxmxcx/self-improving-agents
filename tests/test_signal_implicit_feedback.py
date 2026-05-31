"""ImplicitFeedbackSignal tests.

No LLM calls — the signal reads from the FeedbackStore only. Tests verify
score mapping, kind weighting, and confidence behavior.
"""

from __future__ import annotations

import pytest

from helix.feedback import FeedbackStore, Outcome
from helix.signal import Preference
from helix.signals.implicit_feedback import ImplicitFeedbackSignal
from helix.trajectory import Outcome as TrajOutcome
from helix.trajectory import Trajectory


def _make_trajectory(tid: str) -> Trajectory:
    t = Trajectory(task="any task")
    t.id = tid
    t.complete("ok", outcome=TrajOutcome.COMPLETED)
    return t


@pytest.mark.asyncio
async def test_empty_feedback_yields_zero_confidence():
    store = FeedbackStore(":memory:")
    signal = ImplicitFeedbackSignal(store=store)
    traj = _make_trajectory("never_seen")

    m = await signal.measure(candidate=None, trajectory=traj)
    assert m.confidence == 0.0
    assert m.score is None
    assert m.preference == Preference.NONE


@pytest.mark.asyncio
async def test_thumbs_up_yields_high_score():
    store = FeedbackStore(":memory:")
    signal = ImplicitFeedbackSignal(store=store, min_records_for_confidence=1)
    traj = _make_trajectory("t1")
    await store.record_thumbs(trajectory_id="t1", value=1)

    m = await signal.measure(candidate=None, trajectory=traj)
    assert m.score == pytest.approx(1.0, abs=0.01)
    assert m.confidence == pytest.approx(1.0, abs=0.01)


@pytest.mark.asyncio
async def test_thumbs_down_yields_low_score():
    store = FeedbackStore(":memory:")
    signal = ImplicitFeedbackSignal(store=store, min_records_for_confidence=1)
    traj = _make_trajectory("t1")
    await store.record_thumbs(trajectory_id="t1", value=-1)

    m = await signal.measure(candidate=None, trajectory=traj)
    assert m.score == pytest.approx(0.0, abs=0.01)


@pytest.mark.asyncio
async def test_multiple_kinds_combine():
    store = FeedbackStore(":memory:")
    signal = ImplicitFeedbackSignal(store=store, min_records_for_confidence=1)
    traj = _make_trajectory("t1")
    await store.record_thumbs(trajectory_id="t1", value=1)
    await store.record_copy(trajectory_id="t1")

    m = await signal.measure(candidate=None, trajectory=traj)
    # thumbs_up (score 1.0, weight 1.0) + copy (mapped 0.9, weight 0.5)
    # weighted mean ~= (1.0*1.0 + 0.9*0.5) / 1.5 ~= 0.967
    assert m.score is not None
    assert 0.8 < m.score < 1.0


@pytest.mark.asyncio
async def test_confidence_ramps_with_record_count():
    store = FeedbackStore(":memory:")
    signal = ImplicitFeedbackSignal(store=store, min_records_for_confidence=4)
    traj = _make_trajectory("t1")
    await store.record_thumbs(trajectory_id="t1", value=1)
    m1 = await signal.measure(candidate=None, trajectory=traj)
    assert m1.confidence == pytest.approx(0.25, abs=0.01)

    await store.record_copy(trajectory_id="t1")
    await store.record_copy(trajectory_id="t1")
    await store.record_copy(trajectory_id="t1")
    m2 = await signal.measure(candidate=None, trajectory=traj)
    assert m2.confidence == pytest.approx(1.0, abs=0.01)


@pytest.mark.asyncio
async def test_returns_none_score_when_no_kinds_match():
    """If all feedback is in kinds the signal weights at 0, confidence
    follows record count but the score derivation should still work."""
    store = FeedbackStore(":memory:")
    # Custom weights that zero everything out except thumbs
    signal = ImplicitFeedbackSignal(
        store=store,
        kind_weights={"thumbs": 1.0},
        min_records_for_confidence=1,
    )
    traj = _make_trajectory("t1")
    await store.record_copy(trajectory_id="t1")  # not in weights

    m = await signal.measure(candidate=None, trajectory=traj)
    # No matching kinds; score defaults to 0.5 (neutral)
    assert m.score == 0.5


@pytest.mark.asyncio
async def test_implicit_feedback_kind_is_environment_reward():
    """Implicit feedback is environment-derived reward, not ground truth
    (fix #12: mislabeling inflated its verifiability ceiling)."""
    from helix.signal import SignalKind
    signal = ImplicitFeedbackSignal(store=FeedbackStore(":memory:"))
    assert signal.kind == SignalKind.ENVIRONMENT_REWARD
