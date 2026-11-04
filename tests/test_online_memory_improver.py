"""The online memory improver: live traffic drives MemRL with no human gate.

End-to-end through the real Agent loop with a fake LLM: a seeded entry is
recalled, the run ends, the improver turns the outcome into a reward, and
the entry's utility moves through an operator mutation with lineage.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from helix.agent import Agent
from helix.archive import SQLiteArchive
from helix.artifact import ArtifactKind, Subtype, genesis
from helix.improvement.online_memory import OnlineMemoryImprover
from helix.memory.archive_bridge import entry_utility
from helix.memory.base import MemoryContext, MemoryEntry, Scope, ScopeKey
from helix.memory.episodic import EpisodicMemory
from helix.memory.learning import MemoryLearner
from helix.signal import GapMeasurement, SignalKind
from helix.signals.utility import UtilitySignal


class _FakeResponse:
    def __init__(self, content: str):
        self.choices = [SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=None))]
        self.usage = SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2)


@pytest.fixture
def fake_llm():
    async def fake_acompletion(*, model, messages, **kwargs):
        return _FakeResponse("stub answer")

    with patch("helix.agent.helix_acompletion", side_effect=fake_acompletion):
        yield


class _Outcome:
    """An outcome signal that scores by a flag in the trajectory metadata."""

    kind = SignalKind.GROUND_TRUTH
    signal_id = "flag"
    signal_version = 1

    async def measure(self, candidate=None, trajectory=None, reference=None, ground_truth=None):
        flag = (trajectory.metadata or {}).get("solved")
        if flag is None:
            return GapMeasurement(score=None)
        return GapMeasurement(score=1.0 if flag else 0.0, signal_id="flag", signal_version=1)


async def _setup(tmp_path, reward_fn=None, outcome=None):
    archive = SQLiteArchive(str(tmp_path / "a.sqlite"))
    tier = EpisodicMemory(":memory:", archive=archive)
    seed = MemoryEntry(
        id="seed",
        scope_key=ScopeKey(Scope.USER, "alice"),
        content={"user_message": "vpn keeps dropping", "final_output": "use the wired profile"},
        score=0.5,
    )
    await tier.write(seed)
    agent = Agent(
        system_prompt=genesis("p", Subtype.PROMPT, "be helpful"),
        model="claude-haiku-4-5",
        memory_tiers={"episodic": tier},
    )
    learner = MemoryLearner(tier, archive, UtilitySignal(), alpha=0.5)
    improver = OnlineMemoryImprover(agent, learner=learner, outcome=outcome, reward_fn=reward_fn)
    agent.attach_improver(improver)
    await improver.start()
    return archive, tier, agent, improver


@pytest.mark.asyncio
async def test_a_rewarded_run_upvotes_the_entries_it_recalled_with_lineage(fake_llm, tmp_path):
    archive, tier, agent, improver = await _setup(tmp_path, reward_fn=lambda t: 1.0)
    await agent.run("my vpn drops every hour", context=MemoryContext(session_id="s1", user_id="alice"))

    row = await tier.get("seed")
    assert row.artifact_ref == ("memory.episodic.seed", 2)
    assert row.score == pytest.approx(0.75)
    v2 = (await archive.by_id("memory.episodic.seed", 2)).artifact
    assert v2.parent_id == ("memory.episodic.seed", 1)
    assert v2.metadata["operator"] == "upvote"
    assert entry_utility(v2) == pytest.approx(0.75)
    status = improver.status()
    assert status.trajectories_seen == 1 and status.updates_applied == 1 and status.entries_touched == 1


@pytest.mark.asyncio
async def test_a_failed_run_downvotes_and_an_unscored_run_is_skipped(fake_llm, tmp_path):
    archive, tier, agent, improver = await _setup(tmp_path, outcome=_Outcome())
    # The outcome signal reads a flag the test sets after the run starts; a
    # SESSION_START hook is the honest place to plant it.
    from helix.hooks import HookPoint

    flags = iter([False, None])

    async def plant(trajectory, **_):
        trajectory.metadata["solved"] = next(flags)

    agent.hooks.register(HookPoint.SESSION_START, plant)
    await agent.run("vpn again", context=MemoryContext(session_id="s1", user_id="alice"))
    row = await tier.get("seed")
    assert row.artifact_ref == ("memory.episodic.seed", 2)
    assert row.score == pytest.approx(0.25)
    assert (await archive.by_id("memory.episodic.seed", 2)).artifact.metadata["operator"] == "downvote"

    await agent.run("vpn once more", context=MemoryContext(session_id="s1", user_id="alice"))
    assert improver.status().skipped_no_reward == 1
    assert (await tier.get("seed")).artifact_ref == ("memory.episodic.seed", 2)


@pytest.mark.asyncio
async def test_a_stopped_improver_leaves_memory_alone(fake_llm, tmp_path):
    archive, tier, agent, improver = await _setup(tmp_path, reward_fn=lambda t: 1.0)
    await improver.stop()
    await agent.run("vpn", context=MemoryContext(session_id="s1", user_id="alice"))
    assert (await tier.get("seed")).artifact_ref == ("memory.episodic.seed", 1)


@pytest.mark.asyncio
async def test_a_learning_bug_never_breaks_the_hot_path(fake_llm, tmp_path):
    def boom(trajectory):
        raise RuntimeError("reward service down")

    archive, tier, agent, improver = await _setup(tmp_path, reward_fn=boom)
    answer, traj = await agent.run("vpn", context=MemoryContext(session_id="s1", user_id="alice"))
    assert answer == "stub answer"
    assert "reward service down" in improver.status().last_error


def test_construction_requires_a_reward_source(tmp_path):
    with pytest.raises(ValueError):
        OnlineMemoryImprover(agent=None, search=None)  # type: ignore[arg-type]
