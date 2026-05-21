"""FeedbackStore tests."""

from __future__ import annotations

import pytest

from helix.feedback import (
    FeedbackKind,
    FeedbackStore,
    Outcome,
    reset_feedback_store_for_tests,
)


@pytest.mark.asyncio
async def test_thumbs_round_trip():
    store = FeedbackStore(":memory:")
    await store.record_thumbs(trajectory_id="t1", value=1, session_id="s1", user_id="u1")
    agg = await store.aggregate_for_trajectory("t1")
    assert agg["n"] == 1
    assert agg["mean"] == 1.0
    assert agg["by_kind"]["thumbs"] == 1.0


@pytest.mark.asyncio
async def test_thumbs_rejects_invalid_value():
    store = FeedbackStore(":memory:")
    with pytest.raises(ValueError):
        await store.record_thumbs(trajectory_id="t1", value=0)


@pytest.mark.asyncio
async def test_outcome_maps_to_numeric_value():
    store = FeedbackStore(":memory:")
    await store.record_outcome(session_id="s1", outcome=Outcome.RESOLVED)
    agg = await store.aggregate_for_session("s1")
    assert agg["by_kind"]["outcome"] == 1.0

    await store.record_outcome(session_id="s2", outcome=Outcome.ABANDONED)
    agg2 = await store.aggregate_for_session("s2")
    assert agg2["by_kind"]["outcome"] == -1.0


@pytest.mark.asyncio
async def test_followup_value_decays_with_time():
    store = FeedbackStore(":memory:")
    await store.record_followup(trajectory_id="t1", seconds_after=0.0)    # -1.0
    await store.record_followup(trajectory_id="t2", seconds_after=60.0)   # -0.5
    await store.record_followup(trajectory_id="t3", seconds_after=120.0)  # 0.0

    rec1 = await store.aggregate_for_trajectory("t1")
    rec2 = await store.aggregate_for_trajectory("t2")
    rec3 = await store.aggregate_for_trajectory("t3")
    assert rec1["mean"] == pytest.approx(-1.0, abs=0.01)
    assert rec2["mean"] == pytest.approx(-0.5, abs=0.01)
    assert rec3["mean"] == pytest.approx(0.0, abs=0.01)


@pytest.mark.asyncio
async def test_copy_records_positive():
    store = FeedbackStore(":memory:")
    await store.record_copy(trajectory_id="t1")
    agg = await store.aggregate_for_trajectory("t1")
    assert agg["by_kind"]["copy"] == 0.8


@pytest.mark.asyncio
async def test_regenerate_records_negative():
    store = FeedbackStore(":memory:")
    await store.record_regenerate(trajectory_id="t1")
    agg = await store.aggregate_for_trajectory("t1")
    assert agg["by_kind"]["regenerate"] == -0.8


@pytest.mark.asyncio
async def test_aggregate_for_missing_trajectory_returns_empty():
    store = FeedbackStore(":memory:")
    agg = await store.aggregate_for_trajectory("never_seen")
    assert agg == {"n": 0, "mean": 0.0, "by_kind": {}}


@pytest.mark.asyncio
async def test_recent_returns_records():
    store = FeedbackStore(":memory:")
    for i in range(3):
        await store.record_thumbs(trajectory_id=f"t{i}", value=1)
    recent = await store.recent(n=5)
    assert len(recent) == 3
    assert {r.kind for r in recent} == {FeedbackKind.THUMBS}


def test_get_feedback_store_singleton():
    """get_feedback_store() returns the same instance across calls."""
    reset_feedback_store_for_tests()
    from helix.feedback import get_feedback_store
    s1 = get_feedback_store()
    s2 = get_feedback_store()
    assert s1 is s2
