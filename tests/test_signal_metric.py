"""SignalThreshold and MetricSignal tests."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from helix.artifact import ArtifactKind, genesis
from helix.signal import (
    GapMeasurement,
    Preference,
    Signal,
    SignalKind,
    SignalThreshold,
)
from helix.signals.metric import MetricSignal
from helix.trajectory import Outcome, Step, StepKind, Trajectory


# ---------------- SignalThreshold ----------------


def test_threshold_ratio_minimize_below_baseline_is_good():
    th = SignalThreshold(
        baseline=1000, threshold=500, direction="minimize", normalizer="ratio"
    )
    # raw < baseline → ratio < 1, then 1 - ratio = good score
    score = th.normalize(500)
    assert 0.0 <= score <= 1.0
    assert score > 0.5  # below baseline by half is favorable when minimizing


def test_threshold_ratio_minimize_above_baseline_is_bad():
    th = SignalThreshold(
        baseline=1000, threshold=500, direction="minimize", normalizer="ratio"
    )
    score = th.normalize(2000)
    # raw = 2x baseline (minimize) → score < 0.5
    assert score < 0.5
    assert score == pytest.approx(0.25)  # b/r/2 = 1000/2000/2


def test_threshold_ratio_maximize_above_baseline_is_good():
    th = SignalThreshold(
        baseline=0.5, threshold=0.1, direction="maximize", normalizer="ratio"
    )
    # raw = 0.9 against baseline 0.5 (maximize) → (0.9/0.5)/2 = 0.9
    score = th.normalize(0.9)
    assert score > 0.5
    assert score == pytest.approx(0.9)


def test_threshold_is_triggered_minimize():
    th = SignalThreshold(
        baseline=1000, threshold=500, direction="minimize"
    )
    assert th.is_triggered(1600) is True   # 600 above baseline crosses
    assert th.is_triggered(1400) is False  # 400 above baseline does not
    assert th.is_triggered(500) is False   # well below baseline


def test_threshold_is_triggered_maximize():
    th = SignalThreshold(
        baseline=0.8, threshold=0.1, direction="maximize"
    )
    assert th.is_triggered(0.6) is True    # 0.2 below baseline crosses
    assert th.is_triggered(0.75) is False  # 0.05 below does not


def test_threshold_callable_baseline():
    calls = []

    def baseline_fn() -> float:
        calls.append(1)
        return 100.0

    th = SignalThreshold(
        baseline=baseline_fn, threshold=50, direction="minimize", normalizer="ratio"
    )
    th.normalize(150)
    th.is_triggered(150)
    assert len(calls) == 2  # resolved each time


def test_threshold_minmax_normalizer():
    th = SignalThreshold(
        normalizer="minmax",
        min_value=0,
        max_value=10,
        direction="maximize",
    )
    assert th.normalize(0) == pytest.approx(0.0)
    assert th.normalize(5) == pytest.approx(0.5)
    assert th.normalize(10) == pytest.approx(1.0)
    assert th.normalize(15) == pytest.approx(1.0)  # clipped


def test_threshold_zscore_normalizer():
    th = SignalThreshold(
        baseline=100,
        scale=10,
        normalizer="zscore",
        direction="maximize",
    )
    assert th.normalize(100) == pytest.approx(0.5)  # at baseline
    assert th.normalize(130) > 0.5  # above baseline, maximize → higher score
    assert th.normalize(70) < 0.5


def test_threshold_no_threshold_never_triggers():
    th = SignalThreshold(baseline=100, threshold=None)
    assert th.is_triggered(1000) is False


def test_threshold_no_baseline_never_triggers():
    th = SignalThreshold(baseline=None, threshold=10)
    assert th.is_triggered(1000) is False


# ---------------- MetricSignal ----------------


def _make_trajectory(
    *,
    tool_calls: int = 0,
    model_calls: int = 0,
    tokens: int = 0,
    duration_sec: float = 0.0,
) -> Trajectory:
    started = datetime.now(timezone.utc)
    traj = Trajectory(task="test", started_at=started)
    for i in range(model_calls):
        traj.steps.append(
            Step(
                index=len(traj.steps),
                kind=StepKind.MODEL_CALL,
                payload={"usage": {"total_tokens": tokens // max(model_calls, 1)}},
            )
        )
    for i in range(tool_calls):
        traj.steps.append(
            Step(index=len(traj.steps), kind=StepKind.TOOL_CALL, payload={"name": "test"})
        )
    if duration_sec > 0:
        traj.ended_at = started + timedelta(seconds=duration_sec)
        traj.outcome = Outcome.COMPLETED
    return traj


@pytest.mark.asyncio
async def test_metric_signal_tokens_extraction():
    sig = MetricSignal(
        metric="tokens",
        threshold=SignalThreshold(baseline=1000, threshold=500, direction="minimize"),
    )
    traj = _make_trajectory(model_calls=3, tokens=2400)
    m = await sig.measure(
        candidate=genesis("art.x", ArtifactKind.PROMPT, "x"),
        trajectory=traj,
    )
    assert m.raw_value == pytest.approx(2400.0)
    # 2400 > 1000 + 500, triggered
    assert m.triggered is True
    assert 0.0 <= m.score <= 1.0


@pytest.mark.asyncio
async def test_metric_signal_latency_extraction():
    sig = MetricSignal(
        metric="latency_sec",
        threshold=SignalThreshold(baseline=1.0, threshold=0.5, direction="minimize"),
    )
    traj = _make_trajectory(duration_sec=2.0)
    m = await sig.measure(
        candidate=genesis("art.x", ArtifactKind.PROMPT, "x"),
        trajectory=traj,
    )
    assert m.raw_value == pytest.approx(2.0, abs=0.01)
    assert m.triggered is True


@pytest.mark.asyncio
async def test_metric_signal_tool_calls_count():
    sig = MetricSignal(
        metric="tool_calls",
        threshold=SignalThreshold(baseline=2, threshold=1, direction="minimize"),
    )
    traj = _make_trajectory(tool_calls=4)
    m = await sig.measure(
        candidate=genesis("art.x", ArtifactKind.PROMPT, "x"),
        trajectory=traj,
    )
    assert m.raw_value == pytest.approx(4.0)
    assert m.triggered is True


@pytest.mark.asyncio
async def test_metric_signal_model_calls_count():
    sig = MetricSignal(
        metric="model_calls",
        threshold=SignalThreshold(baseline=3, threshold=1, direction="minimize"),
    )
    traj = _make_trajectory(model_calls=5)
    m = await sig.measure(
        candidate=genesis("art.x", ArtifactKind.PROMPT, "x"),
        trajectory=traj,
    )
    assert m.raw_value == pytest.approx(5.0)
    assert m.triggered is True


@pytest.mark.asyncio
async def test_metric_signal_no_trajectory_returns_empty():
    sig = MetricSignal(
        metric="tokens",
        threshold=SignalThreshold(baseline=1000, threshold=500),
    )
    m = await sig.measure(
        candidate=genesis("art.x", ArtifactKind.PROMPT, "x"),
        trajectory=None,
    )
    assert m.score is None
    assert m.raw_value is None
    assert m.confidence == 0.0
    assert m.triggered is False


@pytest.mark.asyncio
async def test_metric_signal_below_threshold_not_triggered():
    sig = MetricSignal(
        metric="tokens",
        threshold=SignalThreshold(baseline=1000, threshold=500, direction="minimize"),
    )
    traj = _make_trajectory(model_calls=1, tokens=800)
    m = await sig.measure(
        candidate=genesis("art.x", ArtifactKind.PROMPT, "x"),
        trajectory=traj,
    )
    # 800 < 1000 + 500
    assert m.triggered is False
    # below baseline when minimizing → favorable score
    assert m.score > 0.5


def test_metric_signal_unknown_metric_raises():
    with pytest.raises(ValueError, match="unknown metric"):
        MetricSignal(metric="nonsense", threshold=SignalThreshold())  # type: ignore[arg-type]


def test_metric_signal_satisfies_protocol():
    sig = MetricSignal(
        metric="tokens",
        threshold=SignalThreshold(baseline=1000, threshold=500),
    )
    assert isinstance(sig, Signal)
    assert sig.kind == SignalKind.METRIC
    assert isinstance(sig.signal_id, str)
    assert sig.signal_version == 1


def test_two_metric_signals_with_different_configs_have_different_ids():
    a = MetricSignal(
        metric="tokens",
        threshold=SignalThreshold(baseline=1000, threshold=500, direction="minimize"),
    )
    b = MetricSignal(
        metric="tokens",
        threshold=SignalThreshold(baseline=2000, threshold=500, direction="minimize"),
    )
    assert a.signal_id != b.signal_id


def test_two_metric_signals_with_identical_configs_share_id():
    a = MetricSignal(
        metric="latency_sec",
        threshold=SignalThreshold(baseline=1.0, threshold=0.5, direction="minimize"),
    )
    b = MetricSignal(
        metric="latency_sec",
        threshold=SignalThreshold(baseline=1.0, threshold=0.5, direction="minimize"),
    )
    assert a.signal_id == b.signal_id
