"""Chapter 5: the utility signal credits an outcome back to recalled memory.

Builds trajectories by hand with the `memory_recalled` metadata the memory
hooks leave behind, so no tier, embedder, or model is involved.
"""

from __future__ import annotations

import pytest

from helix.archive import SQLiteArchive
from helix.artifact import ArtifactKind, genesis
from helix.signal import Cost, GapMeasurement, Preference, Signal, SignalKind
from helix.signals.utility import UtilitySignal, position_weights
from helix.trajectory import Outcome, Trajectory

IDS = ["memory.episodic.a", "memory.semantic.b", "memory.episodic.c"]


def _trajectory(refs: list[tuple[str, int] | None] | None = None) -> Trajectory:
    t = Trajectory(task="reset password", outcome=Outcome.COMPLETED)
    if refs is None:
        refs = [(i, 1) for i in IDS]
    t.metadata["memory_recalled"] = [
        {
            "tier": (r[0].split(".")[1] if r else "episodic"),
            "entry_id": (r[0].split(".")[2] if r else f"noref{i}"),
            "artifact_ref": list(r) if r else None,
            "scope": "user:alice",
        }
        for i, r in enumerate(refs)
    ]
    return t


class _StubOutcome:
    def __init__(self, score: float, confidence: float) -> None:
        self._score = score
        self._confidence = confidence

    kind = SignalKind.LLM_JUDGE_ABSOLUTE
    cost_estimate = Cost(tokens=10)
    signal_id = "stub_judge"
    signal_version = 3

    async def measure(self, candidate, trajectory=None, reference=None, ground_truth=None):
        return GapMeasurement(
            score=self._score,
            confidence=self._confidence,
            signal_id=self.signal_id,
            signal_version=self.signal_version,
        )


@pytest.fixture
def archive(tmp_path):
    return SQLiteArchive(str(tmp_path / "archive.sqlite"))


def test_utility_signal_is_the_utility_kind_and_satisfies_the_signal_protocol():
    sig = UtilitySignal()
    assert sig.kind is SignalKind.UTILITY
    assert isinstance(sig, Signal)
    assert sig.signal_id == "utility" and sig.signal_version == 1


def test_unknown_attribution_is_rejected_at_construction():
    with pytest.raises(ValueError):
        UtilitySignal(attribution="random")


@pytest.mark.asyncio
async def test_equal_attribution_gives_every_recalled_entry_the_full_reward():
    credits = await UtilitySignal(attribution="equal").attribute(_trajectory(), reward=0.8)
    assert set(credits) == set(IDS)
    for artifact_id, m in credits.items():
        assert isinstance(m, GapMeasurement)
        assert m.score == pytest.approx(0.8)
        assert m.raw_value == pytest.approx(0.8)
        assert m.confidence == 1.0
        assert m.preference is Preference.NONE
        assert m.signal_id == "utility" and m.signal_version == 1
        assert m.metadata["n_recalled"] == 3
        assert m.metadata["attribution"] == "equal"
        assert m.metadata["entry_id"] == artifact_id.split(".")[2]
        assert m.metadata["tier"] == artifact_id.split(".")[1]
        assert m.metadata["outcome_signal_id"] is None


@pytest.mark.asyncio
async def test_position_attribution_halves_credit_by_recall_rank():
    credits = await UtilitySignal(attribution="position").attribute(_trajectory(), reward=0.8)
    assert [credits[i].score for i in IDS] == pytest.approx([0.8, 0.4, 0.2])
    assert all(credits[i].raw_value == pytest.approx(0.8) for i in IDS)
    assert [credits[i].metadata["rank"] for i in IDS] == [0, 1, 2]
    assert position_weights(4) == [1.0, 0.5, 0.25, 0.125]


@pytest.mark.asyncio
async def test_credit_never_leaves_the_unit_interval():
    credits = await UtilitySignal().attribute(_trajectory(), reward=3.0)
    assert all(m.score == 1.0 for m in credits.values())
    credits = await UtilitySignal().attribute(_trajectory(), reward=-1.0)
    assert all(m.score == 0.0 for m in credits.values())


@pytest.mark.asyncio
async def test_entries_recalled_without_an_artifact_ref_are_skipped():
    t = _trajectory([("memory.episodic.a", 2), None])
    credits = await UtilitySignal().attribute(t, reward=1.0)
    assert list(credits) == ["memory.episodic.a"]
    assert credits["memory.episodic.a"].metadata["artifact_ref"] == ["memory.episodic.a", 2]


@pytest.mark.asyncio
async def test_a_run_that_recalled_nothing_yields_no_credit_and_score_none():
    t = Trajectory(task="hello")
    sig = UtilitySignal()
    assert await sig.attribute(t, reward=1.0) == {}
    m = await sig.measure(genesis(id=IDS[0], kind=ArtifactKind.MEMORY_ENTRY, content={}), t, ground_truth=1.0)
    assert m.score is None
    assert m.metadata["reason"] == "not recalled"
    assert m.signal_id == "utility"


@pytest.mark.asyncio
async def test_measure_returns_credit_for_a_recalled_artifact_and_none_otherwise():
    sig = UtilitySignal(outcome=_StubOutcome(0.6, 1.0))
    t = _trajectory()
    recalled = genesis(id=IDS[1], kind=ArtifactKind.MEMORY_ENTRY, content={"x": 1})
    stranger = genesis(id="memory.episodic.zzz", kind=ArtifactKind.MEMORY_ENTRY, content={"y": 2})

    hit = await sig.measure(recalled, t)
    assert hit.score == pytest.approx(0.6)
    assert hit.metadata["entry_id"] == "b"

    miss = await sig.measure(stranger, t)
    assert miss.score is None and miss.confidence == 0.0
    assert miss.metadata["reason"] == "not recalled"

    no_traj = await sig.measure(recalled)
    assert no_traj.score is None


@pytest.mark.asyncio
async def test_confidence_and_signal_id_propagate_from_the_outcome_signal():
    sig = UtilitySignal(outcome=_StubOutcome(0.9, 0.4), attribution="position")
    credits = await sig.attribute(_trajectory())
    assert credits[IDS[0]].score == pytest.approx(0.9)
    assert credits[IDS[2]].score == pytest.approx(0.225)
    for m in credits.values():
        assert m.confidence == pytest.approx(0.4)
        assert m.metadata["outcome_signal_id"] == "stub_judge"
    assert sig.cost_estimate.tokens == 10


@pytest.mark.asyncio
async def test_an_explicit_outcome_measurement_is_used_instead_of_the_wrapped_signal():
    sig = UtilitySignal(outcome=_StubOutcome(0.1, 0.1))
    given = GapMeasurement(score=0.7, confidence=0.5, signal_id="ground_truth_checker")
    credits = await sig.attribute(_trajectory(), outcome_measurement=given)
    assert all(m.score == pytest.approx(0.7) and m.confidence == 0.5 for m in credits.values())
    assert credits[IDS[0]].metadata["outcome_signal_id"] == "ground_truth_checker"


@pytest.mark.asyncio
async def test_an_outcome_without_a_score_produces_no_credit():
    sig = UtilitySignal(outcome=_StubOutcome(None, 0.0))  # type: ignore[arg-type]
    assert await sig.attribute(_trajectory()) == {}
    assert await UtilitySignal().attribute(_trajectory()) == {}


@pytest.mark.asyncio
async def test_record_all_writes_credit_against_the_recalled_version(archive):
    for i in IDS:
        await archive.put_artifact(
            genesis(id=i, kind=ArtifactKind.MEMORY_ENTRY, content={"entry": i, "utility": None})
        )
    t = _trajectory()
    written = await UtilitySignal(attribution="position").record_all(archive, t, reward=1.0)
    assert [w[0] for w in written] == IDS

    history = await archive.get_measurement_history(IDS[1], 1)
    assert len(history) == 1
    assert history[0].score == pytest.approx(0.5)
    assert history[0].metadata["trajectory_id"] == t.id
    assert history[0].metadata["attribution"] == "position"

    # The archive still holds exactly one version; recording credit is not a mutation.
    assert await archive.next_version(IDS[1]) == 2

    # A second run accumulates rather than overwrites.
    await UtilitySignal().record_all(archive, _trajectory(), reward=0.0)
    assert [m.score for m in await archive.get_measurement_history(IDS[1], 1)] == pytest.approx([0.5, 0.0])


@pytest.mark.asyncio
async def test_record_all_skips_refs_the_archive_does_not_hold(archive):
    await archive.put_artifact(genesis(id=IDS[0], kind=ArtifactKind.MEMORY_ENTRY, content={"k": 1}))
    written = await UtilitySignal().record_all(archive, _trajectory(), reward=0.5)
    assert [w[0] for w in written] == [IDS[0]]
