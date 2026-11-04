"""The fixed memory harness applies learned values across runs and instances."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from helix.agent import Agent
from helix.archive import SQLiteArchive
from helix.artifact import Subtype, genesis
from helix.improvement.online_memory import OnlineMemoryImprover
from helix.memory import MemoryLearner
from helix.memory.base import MemoryContext, MemoryEntry, Scope, ScopeKey
from helix.memory.episodic import EpisodicMemory
from helix.memory.hooks import RECALLED_KEY
from helix.search.memrl import MemRLSearch
from helix.signals.utility import UtilitySignal


@pytest.fixture(autouse=True)
def local_embeddings():
    # Equal relevance isolates value learning from model and embedding variance.
    with patch("helix.memory.episodic._embed", side_effect=lambda texts: [[1.0, 0.0] for _ in texts]):
        yield


def make_agent(tier):
    agent = Agent(
        system_prompt=genesis("prompt.memory-test", Subtype.PROMPT, "Use relevant prior experience."),
        model="test-model",
        memory_tiers={"episodic": tier},
    )
    agent.memory_hooks.episodic_k = 1
    return agent


def recalled_ids(trajectory):
    return [row["entry_id"] for row in trajectory.metadata[RECALLED_KEY]]


async def test_fixed_harness_uses_changed_values_within_sessions_and_across_agent_instances(tmp_path):
    archive = SQLiteArchive(tmp_path / "archive.sqlite")
    tier = EpisodicMemory(tmp_path / "memory.sqlite", archive=archive, utility_weight=0.35, recency_weight=0.0)
    for eid, value in [("first", 0.60), ("alternative", 0.55)]:
        await tier.write(MemoryEntry(
            id=eid, scope_key=ScopeKey(Scope.USER, "alice"), score=value,
            content={"user_message": "connection problem", "final_output": eid + " solution"},
        ))
    agent = make_agent(tier)
    learner = MemoryLearner(tier, archive, UtilitySignal(attribution="equal"), alpha=0.5,
                            upvote_threshold=0.5, downvote_threshold=0.5)
    rewards = iter([0.0, 1.0])
    improver = OnlineMemoryImprover(agent, learner=learner, reward_fn=lambda _: next(rewards))
    agent.attach_improver(improver)
    await improver.start()
    hook_wrapper = agent.memory_hooks
    fixed = (tier.utility_weight, tier.recency_weight, hook_wrapper.episodic_k,
             learner.alpha, learner.utility.attribution, learner.algorithm,
             agent.model, agent.system_prompt.ref, agent.system_prompt.content)
    contexts = []

    async def model(*, messages, **_):
        contexts.append(str(messages[0]["content"]))
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="completed request", tool_calls=None))],
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )

    with patch("helix.agent.helix_acompletion", side_effect=model):
        _, first = await agent.run("connection problem", context=MemoryContext(user_id="alice", session_id="s1"))
        _, second = await agent.run("connection problem", context=MemoryContext(user_id="alice", session_id="s1"))
        await improver.stop()

        assert recalled_ids(first) == ["first"]
        assert recalled_ids(second) == ["alternative"]
        assert "first solution" in contexts[0] and "alternative solution" in contexts[1]
        assert (await tier.get("first")).score == pytest.approx(0.30)
        assert (await tier.get("alternative")).score == pytest.approx(0.775)
        assert (await tier.get("first")).content["final_output"] == "first solution"
        new_entries = [entry for entry in await tier.list_all() if entry.id not in ("first", "alternative")]
        assert len(new_entries) == 2 and all(entry.score == 0.5 for entry in new_entries)
        assert agent.memory_hooks is hook_wrapper
        assert fixed == (tier.utility_weight, tier.recency_weight, hook_wrapper.episodic_k,
                         learner.alpha, learner.utility.attribution, learner.algorithm,
                         agent.model, agent.system_prompt.ref, agent.system_prompt.content)
        assert improver.status().updates_applied == 2 and improver.status().last_error is None

        # A new harness reopens the retained learning; it does not need a learner
        # to apply it. A different user's identity cannot recall Alice's records.
        reopened = EpisodicMemory(tier.path, archive=archive, utility_weight=0.35, recency_weight=0.0)
        other_agent = make_agent(reopened)
        other_agent.memory_hooks.read_only = True
        _, later = await other_agent.run("connection problem", context=MemoryContext(user_id="alice", session_id="s2"))
        _, unrelated = await other_agent.run("connection problem", context=MemoryContext(user_id="bob", session_id="s3"))
        assert recalled_ids(later) == ["alternative"]
        assert recalled_ids(unrelated) == []

    child = await archive.by_id("memory.episodic.alternative", 2)
    assert child.artifact.parent_id == ("memory.episodic.alternative", 1)
    assert child.measurement.metadata["learning_algorithm"] == "one_step_value_update"
    assert child.measurement.metadata["alpha"] == 0.5


def test_online_learning_requires_the_memory_tier_installed_in_the_harness(tmp_path):
    archive = SQLiteArchive(tmp_path / "archive.sqlite")
    installed = EpisodicMemory(":memory:", archive=archive)
    disconnected = EpisodicMemory(":memory:", archive=archive)
    learner = MemoryLearner(disconnected, archive, UtilitySignal())
    with pytest.raises(ValueError, match="tier installed in the agent harness"):
        OnlineMemoryImprover(make_agent(installed), learner=learner, reward_fn=lambda _: 1.0)


def test_learning_requires_the_memory_tiers_archive(tmp_path):
    archive = SQLiteArchive(tmp_path / "archive.sqlite")
    other_archive = SQLiteArchive(tmp_path / "other.sqlite")
    tier = EpisodicMemory(":memory:", archive=archive)
    with pytest.raises(ValueError, match="memory tier's archive"):
        MemoryLearner(tier, other_archive, UtilitySignal())


def test_legacy_search_keyword_still_connects_the_same_learner(tmp_path):
    archive = SQLiteArchive(tmp_path / "archive.sqlite")
    tier = EpisodicMemory(":memory:", archive=archive)
    search = MemRLSearch(tier, archive, UtilitySignal())
    improver = OnlineMemoryImprover(make_agent(tier), search=search, reward_fn=lambda _: 1.0)
    assert improver.learner is search and improver.search is search


def test_two_different_learner_arguments_are_rejected(tmp_path):
    archive = SQLiteArchive(tmp_path / "archive.sqlite")
    tier = EpisodicMemory(":memory:", archive=archive)
    learner = MemoryLearner(tier, archive, UtilitySignal())
    other = MemRLSearch(tier, archive, UtilitySignal())
    with pytest.raises(ValueError, match="different objects"):
        OnlineMemoryImprover(make_agent(tier), learner=learner, search=other, reward_fn=lambda _: 1.0)
