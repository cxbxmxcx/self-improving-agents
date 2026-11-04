"""The group round: N candidates, an absolute signal, advantage over the mean.

SPEC §4.2.1 calls this the third round shape after pairwise and archive.
Every test uses a stub signal so nothing calls a model.
"""

from __future__ import annotations

import pytest

from helix.archive import SQLiteArchive
from helix.artifact import Artifact, Subtype, genesis
from helix.eval.dataset import EvalQuestion
from helix.improvement.group_round import GroupRoundResult, group_advantages, run_group_round
from helix.observability.bus import EventBus
from helix.search.base import SearchBudget, SearchKind, Variant
from helix.signal import Cost, GapMeasurement, SignalKind


class FixedScoreSignal:
    """Scores each artifact by a table keyed on its text content."""

    def __init__(self, table: dict[str, float], cost_tokens: int = 10) -> None:
        self.table = table
        self.cost_tokens = cost_tokens
        self.calls: list[tuple[str, object]] = []

    kind = SignalKind.GROUND_TRUTH
    cost_estimate = Cost()
    signal_id = "fixed"
    signal_version = 1

    async def measure(self, candidate, trajectory=None, reference=None, ground_truth=None):
        self.calls.append((candidate.content, trajectory))
        return GapMeasurement(
            score=self.table[candidate.content],
            signal_id=self.signal_id,
            signal_version=1,
            cost=Cost(tokens=self.cost_tokens),
        )


@pytest.fixture
def archive(tmp_path):
    return SQLiteArchive(str(tmp_path / "archive.sqlite"))


def _group(seed: Artifact, texts: list[str]) -> list[Variant]:
    return [
        Variant(
            artifact=seed.mutate(t, created_by="test", version=i + 2),
            parent=seed,
            search_method="test",
            metadata={"group_index": i},
        )
        for i, t in enumerate(texts)
    ]


@pytest.fixture
def seed():
    return genesis("p", Subtype.PROMPT, "seed prompt")


def test_group_advantages_match_hand_computed_values():
    mean, std, adv, norm = group_advantages([0.2, 0.4, 0.9])
    assert mean == pytest.approx(0.5)
    assert std == pytest.approx(0.294392, abs=1e-5)
    assert adv == pytest.approx([-0.3, -0.1, 0.4])
    assert norm == pytest.approx([-0.3 / std, -0.1 / std, 0.4 / std])


def test_group_advantages_normalize_to_zero_when_every_candidate_ties():
    mean, std, adv, norm = group_advantages([0.5, 0.5])
    assert (mean, std) == (0.5, 0.0)
    assert adv == [0.0, 0.0] and norm == [0.0, 0.0]


@pytest.mark.asyncio
async def test_positive_selection_keeps_only_above_mean_candidates(archive, seed):
    signal = FixedScoreSignal({"a": 0.2, "b": 0.4, "c": 0.9, "d": 0.5})
    result = await run_group_round(_group(seed, ["a", "b", "c", "d"]), signal=signal, archive=archive, bus=EventBus())
    assert isinstance(result, GroupRoundResult)
    assert result.group_mean == pytest.approx(0.5)
    assert [v.artifact.content for v in result.selected] == ["c"]
    assert result.best.artifact.content == "c"
    assert result.n_candidates == 4


@pytest.mark.asyncio
async def test_top_k_selection_keeps_the_k_highest_scores(archive, seed):
    signal = FixedScoreSignal({"a": 0.2, "b": 0.4, "c": 0.9, "d": 0.5})
    result = await run_group_round(
        _group(seed, ["a", "b", "c", "d"]), signal=signal, archive=archive, bus=EventBus(), select="top_k", k=2
    )
    assert [v.artifact.content for v in result.selected] == ["c", "d"]


@pytest.mark.asyncio
async def test_measurements_are_recorded_with_group_metadata(archive, seed):
    signal = FixedScoreSignal({"a": 0.2, "b": 0.4, "c": 0.9})
    result = await run_group_round(_group(seed, ["a", "b", "c"]), signal=signal, archive=archive, bus=EventBus(), round_index=3)
    stored = await archive.by_id("p", 2)
    md = stored.measurement.metadata
    assert md["group_round"] == 3
    assert md["group_size"] == 3
    assert md["group_mean"] == pytest.approx(0.5)
    assert md["advantage"] == pytest.approx(-0.3)
    assert md["advantage_norm"] == pytest.approx(-0.3 / result.group_std)
    assert stored.artifact.parent_id == ("p", 1), "lineage survives the record"
    assert result.cost.tokens == 30, "cost is the sum of the measurement costs"


@pytest.mark.asyncio
async def test_record_false_scores_without_touching_the_archive(archive, seed):
    signal = FixedScoreSignal({"a": 0.2, "b": 0.8})
    await run_group_round(_group(seed, ["a", "b"]), signal=signal, archive=archive, bus=EventBus(), record=False)
    assert await archive.by_id("p", 2) is None


@pytest.mark.asyncio
async def test_round_events_and_candidate_wins_are_published(archive, seed):
    bus = EventBus()
    seen: list = []
    bus.subscribe("*", seen.append)
    signal = FixedScoreSignal({"a": 0.2, "b": 0.8})
    await run_group_round(_group(seed, ["a", "b"]), signal=signal, archive=archive, bus=bus, mode="online")

    types = [e.event_type for e in seen]
    assert types[0] == "improver_round_started"
    assert "signal_identified" in types
    assert types[-1] == "improver_round_completed"
    wins = [e for e in seen if e.event_type == "candidate_wins"]
    assert len(wins) == 1
    assert wins[0].candidate_version == 3 and wins[0].reference_score == pytest.approx(0.5)
    assert wins[0].mode == "online"
    completed = seen[-1]
    assert (completed.n_candidate_wins, completed.n_reference_wins, completed.n_ties) == (1, 1, 0)
    assert completed.promoted is True


@pytest.mark.asyncio
async def test_no_candidate_wins_when_the_group_ties(archive, seed):
    bus = EventBus()
    seen: list = []
    bus.subscribe("candidate_wins", seen.append)
    signal = FixedScoreSignal({"a": 0.5, "b": 0.5})
    result = await run_group_round(_group(seed, ["a", "b"]), signal=signal, archive=archive, bus=bus)
    assert seen == []
    assert result.selected == []
    assert result.best is not None


class CountingProposer:
    """Yields more variants than asked so the round must stop pulling."""

    kind = SearchKind.GROUP_RELATIVE
    group_size = 3

    def __init__(self, seed: Artifact) -> None:
        self.seed = seed
        self.yielded = 0

    async def propose(self, seed, signal, archive, budget):
        for i in range(100):
            self.yielded += 1
            yield Variant(
                artifact=seed.mutate(f"v{i}", created_by="proposer", version=i + 2),
                parent=seed,
                search_method="group_relative",
            )

    async def select(self, candidates, signal, archive):
        return candidates[0].artifact


@pytest.mark.asyncio
async def test_proposer_path_pulls_exactly_the_group_size(archive, seed):
    proposer = CountingProposer(seed)
    signal = FixedScoreSignal({f"v{i}": i / 10 for i in range(10)})
    result = await run_group_round(
        None, proposer=proposer, seed=seed, signal=signal, archive=archive, bus=EventBus()
    )
    assert result.n_candidates == 3
    assert proposer.yielded == 3
    assert [v.artifact.content for v in result.selected] == ["v2"]

    explicit = await run_group_round(
        None, proposer=CountingProposer(seed), seed=seed, group_size=5, signal=signal, archive=archive, bus=EventBus()
    )
    assert explicit.n_candidates == 5


@pytest.mark.asyncio
async def test_direct_scoring_calls_the_signal_without_a_trajectory(archive, seed):
    signal = FixedScoreSignal({"a": 0.1, "b": 0.9})
    await run_group_round(_group(seed, ["a", "b"]), signal=signal, archive=archive, bus=EventBus())
    assert [t for _, t in signal.calls] == [None, None]


class _StubAgent:
    def __init__(self, artifact: Artifact) -> None:
        self.artifact = artifact

    async def run(self, question: str):
        from helix.trajectory import Trajectory

        traj = Trajectory(task=question)
        traj.complete(f"{self.artifact.content}:{question}")
        return traj.final_output, traj


@pytest.mark.asyncio
async def test_question_path_runs_each_candidate_and_averages_the_scores(archive, seed):
    questions = [
        EvalQuestion(id="q1", band=1, question="one", reference_answer="1"),
        EvalQuestion(id="q2", band=1, question="two", reference_answer="2"),
    ]

    class PerQuestionSignal(FixedScoreSignal):
        async def measure(self, candidate, trajectory=None, reference=None, ground_truth=None):
            assert trajectory is not None and ground_truth["question"] in ("one", "two")
            base = self.table[candidate.content]
            bonus = 0.2 if ground_truth["question"] == "two" else 0.0
            return GapMeasurement(score=base + bonus, cost=Cost(tokens=5))

    signal = PerQuestionSignal({"a": 0.1, "b": 0.5})
    result = await run_group_round(
        _group(seed, ["a", "b"]),
        signal=signal,
        archive=archive,
        bus=EventBus(),
        questions=questions,
        agent_factory=lambda art: _StubAgent(art),
    )
    by_text = {v.artifact.content: m for v, m in result.scores}
    assert by_text["a"].score == pytest.approx(0.2)
    assert by_text["b"].score == pytest.approx(0.6)
    assert by_text["a"].metadata["n_questions"] == 2
    assert result.cost.tokens == 20


@pytest.mark.asyncio
async def test_budget_is_charged_for_every_measurement(archive, seed):
    budget = SearchBudget(max_tokens=1000)
    signal = FixedScoreSignal({"a": 0.1, "b": 0.9}, cost_tokens=7)
    await run_group_round(_group(seed, ["a", "b"]), signal=signal, archive=archive, bus=EventBus(), budget=budget)
    assert budget.remaining()["tokens"] == 1000 - 14


@pytest.mark.asyncio
async def test_invalid_selection_rule_is_rejected(archive, seed):
    with pytest.raises(ValueError):
        await run_group_round(_group(seed, ["a"]), signal=FixedScoreSignal({"a": 1.0}), archive=archive, select="median")
