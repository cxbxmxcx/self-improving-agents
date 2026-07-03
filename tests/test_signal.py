"""Signal protocol and CompositeSignal conformance tests."""

from __future__ import annotations

import pytest

from helix.signal import (
    CompositeSignal,
    Cost,
    GapMeasurement,
    Preference,
    Signal,
    SignalKind,
)


def test_gap_measurement_round_trip_via_dict():
    m = GapMeasurement(
        score=0.7,
        preference=Preference.LEFT,
        feedback="candidate is sharper",
        confidence=0.85,
        rubric_id=("rubric.test", 2),
        cost=Cost(tokens=500, dollars=0.002),
        metadata={"judge_model": "gpt-4o"},
    )
    rt = GapMeasurement.from_dict(m.to_dict())
    assert rt.score == m.score
    assert rt.preference == m.preference
    assert rt.feedback == m.feedback
    assert rt.confidence == m.confidence
    assert rt.rubric_id == m.rubric_id
    assert rt.cost.tokens == 500
    assert rt.metadata["judge_model"] == "gpt-4o"


def test_proven_marks_only_computed_truths_and_survives_round_trip():
    # A formal proof: the score was computed, not estimated.
    proof = GapMeasurement(score=1.0, proven=True)
    assert GapMeasurement.from_dict(proof.to_dict()).proven is True

    # Every other family defaults to unproven, including a deterministic
    # ground-truth check: a checker is code that can be wrong.
    ground_truth = GapMeasurement(score=1.0, confidence=1.0)
    assert ground_truth.proven is False

    # Rows persisted before the field existed deserialize as unproven.
    legacy = ground_truth.to_dict()
    del legacy["proven"]
    assert GapMeasurement.from_dict(legacy).proven is False


class _FakeAbsoluteSignal:
    """A toy absolute signal that returns a fixed score."""

    def __init__(self, score: float, confidence: float = 1.0) -> None:
        self._score = score
        self._confidence = confidence

    @property
    def kind(self) -> SignalKind:
        return SignalKind.LLM_JUDGE_ABSOLUTE

    @property
    def cost_estimate(self) -> Cost:
        return Cost(tokens=100)

    @property
    def signal_id(self) -> str:
        return f"FakeAbsoluteSignal:{self._score}"

    @property
    def signal_version(self) -> int:
        return 1

    async def measure(self, candidate, trajectory=None, reference=None, ground_truth=None):
        return GapMeasurement(
            score=self._score,
            confidence=self._confidence,
            signal_id=self.signal_id,
            signal_version=self.signal_version,
            cost=Cost(tokens=100),
        )


class _FakePairwiseSignal:
    def __init__(self, preference: Preference, confidence: float = 1.0) -> None:
        self._pref = preference
        self._confidence = confidence

    @property
    def kind(self) -> SignalKind:
        return SignalKind.LLM_JUDGE_PAIRWISE

    @property
    def cost_estimate(self) -> Cost:
        return Cost(tokens=100)

    @property
    def signal_id(self) -> str:
        return f"FakePairwiseSignal:{self._pref.value}"

    @property
    def signal_version(self) -> int:
        return 1

    async def measure(self, candidate, trajectory=None, reference=None, ground_truth=None):
        score = 1.0 if self._pref == Preference.LEFT else (0.0 if self._pref == Preference.RIGHT else 0.5)
        return GapMeasurement(
            score=score,
            preference=self._pref,
            confidence=self._confidence,
            signal_id=self.signal_id,
            signal_version=self.signal_version,
            cost=Cost(tokens=100),
        )


@pytest.mark.asyncio
async def test_composite_signal_mean_aggregates_scores():
    composite = CompositeSignal(
        signals=[_FakeAbsoluteSignal(0.4), _FakeAbsoluteSignal(0.8), _FakeAbsoluteSignal(0.6)],
        aggregator="mean",
    )
    m = await composite.measure(candidate=None)  # type: ignore[arg-type]
    assert m.score == pytest.approx(0.6, abs=1e-6)


@pytest.mark.asyncio
async def test_composite_signal_conservative_min_takes_worst():
    composite = CompositeSignal(
        signals=[_FakeAbsoluteSignal(0.4), _FakeAbsoluteSignal(0.8)],
        aggregator="conservative_min",
    )
    m = await composite.measure(candidate=None)  # type: ignore[arg-type]
    assert m.score == 0.4


@pytest.mark.asyncio
async def test_composite_signal_weighted_mean_respects_weights():
    composite = CompositeSignal(
        signals=[_FakeAbsoluteSignal(0.0), _FakeAbsoluteSignal(1.0)],
        aggregator="weighted_mean",
        weights=[3.0, 1.0],
    )
    m = await composite.measure(candidate=None)  # type: ignore[arg-type]
    # (0.0 * 3 + 1.0 * 1) / 4 = 0.25
    assert m.score == pytest.approx(0.25)


@pytest.mark.asyncio
async def test_composite_signal_majority_preference_wins():
    composite = CompositeSignal(
        signals=[
            _FakePairwiseSignal(Preference.LEFT),
            _FakePairwiseSignal(Preference.LEFT),
            _FakePairwiseSignal(Preference.RIGHT),
        ],
    )
    m = await composite.measure(candidate=None)  # type: ignore[arg-type]
    assert m.preference == Preference.LEFT


@pytest.mark.asyncio
async def test_composite_signal_split_preference_returns_tie():
    composite = CompositeSignal(
        signals=[_FakePairwiseSignal(Preference.LEFT), _FakePairwiseSignal(Preference.RIGHT)],
    )
    m = await composite.measure(candidate=None)  # type: ignore[arg-type]
    assert m.preference == Preference.TIE


def test_signal_protocol_is_satisfied_by_implementations():
    """Both fake signals should structurally satisfy the Signal protocol."""
    assert isinstance(_FakeAbsoluteSignal(0.5), Signal)
    assert isinstance(_FakePairwiseSignal(Preference.LEFT), Signal)
