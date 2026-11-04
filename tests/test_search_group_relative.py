"""GroupRelativeSearch and its samplers. SPEC §4.2, §4.2.1.

The signal here scores by content length, so the hand-computed winner is
always the longest candidate and no model is needed.
"""

from __future__ import annotations

import pytest

from helix.archive import SQLiteArchive
from helix.artifact import ArtifactKind, Subtype, genesis
from helix.improvement.group_round import run_group_round
from helix.memory.archive_bridge import entry_payload, entry_utility
from helix.observability.bus import EventBus
from helix.search.base import Search, SearchBudget, SearchKind, Variant
from helix.search.group_relative import (
    GroupRelativeSearch,
    InsightSampler,
    SplitSampler,
    parse_numbered_lines,
)
from helix.signal import Cost, GapMeasurement, SignalKind
from helix.trajectory import StepKind, Trajectory


class LengthSignal:
    """Scores an artifact by how much text it carries, clipped to [0, 1]."""

    kind = SignalKind.GROUND_TRUTH
    cost_estimate = Cost()
    signal_id = "length"
    signal_version = 1

    async def measure(self, candidate, trajectory=None, reference=None, ground_truth=None):
        text = candidate.content if isinstance(candidate.content, str) else str(entry_payload(candidate).get("insight", ""))
        return GapMeasurement(score=min(1.0, len(text) / 100.0), cost=Cost(tokens=1))


@pytest.fixture
def archive(tmp_path):
    return SQLiteArchive(str(tmp_path / "archive.sqlite"))


SEED_TEXT = "Short one. A somewhat longer second sentence here. The third sentence is the longest of them all by a margin."


def test_group_relative_search_satisfies_the_search_protocol():
    search = GroupRelativeSearch(SplitSampler(), group_size=3)
    assert isinstance(search, Search)
    assert search.kind is SearchKind.GROUP_RELATIVE
    assert search.cost_model.proposes_per_round == 3


@pytest.mark.asyncio
async def test_propose_yields_the_group_with_distinct_versions_and_group_metadata(archive):
    seed = genesis("p", Subtype.PROMPT, SEED_TEXT)
    search = GroupRelativeSearch(SplitSampler(), group_size=3, bus=EventBus())
    variants = [v async for v in search.propose(seed, LengthSignal(), archive, SearchBudget())]
    assert len(variants) == 3
    assert [v.metadata for v in variants] == [{"group_index": i, "group_size": 3} for i in range(3)]
    assert [v.artifact.version for v in variants] == [2, 3, 4]
    assert all(v.artifact.parent_id == ("p", 1) and v.search_method == "group_relative" for v in variants)
    assert [v.artifact.content for v in variants] == [
        "Short one.",
        "A somewhat longer second sentence here.",
        "The third sentence is the longest of them all by a margin.",
    ]


@pytest.mark.asyncio
async def test_propose_publishes_search_and_mutation_events(archive):
    bus = EventBus()
    seen: list = []
    bus.subscribe("*", seen.append)
    seed = genesis("p", Subtype.PROMPT, SEED_TEXT)
    search = GroupRelativeSearch(SplitSampler(), group_size=2, bus=bus)
    _ = [v async for v in search.propose(seed, LengthSignal(), archive, SearchBudget())]
    types = [e.event_type for e in seen]
    assert types == ["search_started", "mutation_proposed", "mutation_proposed", "search_completed"]
    assert seen[-1].candidates_evaluated == 2


@pytest.mark.asyncio
async def test_select_returns_the_highest_advantage_candidate(archive):
    seed = genesis("p", Subtype.PROMPT, "x")
    search = GroupRelativeSearch(SplitSampler())
    variants = []
    for i, adv in enumerate([-0.2, 0.3, -0.1]):
        v = Variant(artifact=seed.mutate(f"c{i}", created_by="t", version=i + 2), parent=seed, search_method="group_relative")
        v.measurement = GapMeasurement(score=0.5 + adv, metadata={"advantage": adv})
        variants.append(v)
    chosen = await search.select(variants, LengthSignal(), archive)
    assert chosen.content == "c1"


@pytest.mark.asyncio
async def test_select_falls_back_to_score_when_no_advantage_was_recorded(archive):
    seed = genesis("p", Subtype.PROMPT, "x")
    search = GroupRelativeSearch(SplitSampler())
    a = Variant(artifact=seed.mutate("a", created_by="t", version=2), parent=seed, search_method="g", measurement=GapMeasurement(score=0.2))
    b = Variant(artifact=seed.mutate("b", created_by="t", version=3), parent=seed, search_method="g", measurement=GapMeasurement(score=0.7))
    assert (await search.select([a, b], LengthSignal(), archive)).content == "b"


@pytest.mark.asyncio
async def test_split_sampler_round_end_to_end_keeps_the_above_mean_chunks(archive):
    seed = genesis("p", Subtype.PROMPT, SEED_TEXT)
    search = GroupRelativeSearch(SplitSampler(), group_size=3, bus=EventBus())
    result = await run_group_round(
        None, proposer=search, seed=seed, signal=LengthSignal(), archive=archive, bus=EventBus()
    )
    assert result.n_candidates == 3
    # Lengths 10, 39, 58 give a mean of 35.67: two chunks sit above it.
    assert [v.artifact.content for v in result.selected] == [
        "The third sentence is the longest of them all by a margin.",
        "A somewhat longer second sentence here.",
    ]
    winner = await search.select([v for v, _ in result.scores], LengthSignal(), archive)
    assert winner == result.best.artifact
    stored = await archive.by_id("p", 4)
    assert stored.measurement.metadata["advantage"] > 0
    assert stored.artifact.parent_id == ("p", 1)


def test_parse_numbered_lines_strips_markers_and_caps_at_n():
    text = "1. Check the cache first.\n2) Prefer wired.\n- Ask for the OS.\n\n4. Extra lesson."
    assert parse_numbered_lines(text, 3) == ["Check the cache first.", "Prefer wired.", "Ask for the OS."]


@pytest.mark.asyncio
async def test_insight_sampler_mints_memory_entry_artifacts_from_a_trajectory(archive):
    prompts: list[str] = []

    async def llm(prompt: str) -> str:
        prompts.append(prompt)
        return "1. Ask which VPN client is installed before suggesting fixes.\n2. Suggest the wired profile.\n3. Log the hour of each drop."

    traj = Trajectory(task="my vpn drops every hour")
    traj.append(StepKind.MODEL_CALL, {"content": "Try the wired profile"})
    traj.complete("switch to the wired profile")

    sampler = InsightSampler(llm, n=3)
    seed = genesis("memory.episodic.src", ArtifactKind.MEMORY_ENTRY, {"tier": "episodic", "scope": "user:alice", "entry": {}, "utility": 0.0})
    search = GroupRelativeSearch(sampler, group_size=3, context={"trajectory": traj}, bus=EventBus())
    variants = [v async for v in search.propose(seed, LengthSignal(), archive, SearchBudget())]

    assert len(variants) == 3
    assert "my vpn drops every hour" in prompts[0] and "switch to the wired profile" in prompts[0]
    arts = [v.artifact for v in variants]
    assert all(a.kind is ArtifactKind.MEMORY_ENTRY and a.parent_id is None for a in arts)
    assert all(a.id.startswith("memory.episodic.insight-") for a in arts)
    assert len({a.id for a in arts}) == 3
    assert entry_payload(arts[0]) == {
        "insight": "Ask which VPN client is installed before suggesting fixes.",
        "source_trajectory_id": traj.id,
    }
    assert all(entry_utility(a) == 0.0 for a in arts)
    assert arts[0].content["scope"] == "user:alice"
    assert arts[0].metadata["operator"] == "add"
    assert arts[1].metadata["group_index"] == 1

    result = await run_group_round(variants, signal=LengthSignal(), archive=archive, bus=EventBus())
    assert result.best.artifact.id == arts[0].id, "the longest insight wins under the length signal"
    assert (await archive.by_id(arts[0].id)).measurement.metadata["group_size"] == 3
