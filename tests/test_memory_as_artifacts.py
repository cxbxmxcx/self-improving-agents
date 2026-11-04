"""Chapter 5 foundation: memory entries are artifacts, memory is a hook pair.

Covers SPEC §7.3 (entries recorded as MEMORY_ENTRY artifacts with lineage),
§6.3 (PRE_MODEL recall and SESSION_END write-back as a wrapper that clones
keep), and §5.2 (archive pool filters). Uses the fake LLM from the memory
integration tests so nothing calls a model.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from helix.agent import Agent
from helix.archive import SQLiteArchive
from helix.archive.archive import SamplingStrategy
from helix.artifact import ArtifactKind, Subtype, genesis
from helix.hooks import HookPoint
from helix.memory.archive_bridge import entry_payload, entry_utility
from helix.memory.base import MemoryContext, MemoryEntry, Scope, ScopeKey
from helix.memory.episodic import EpisodicMemory
from helix.memory.hooks import RECALLED_KEY, WRITTEN_KEY
from helix.memory.semantic import SemanticMemory
from helix.search.base import Variant
from helix.signal import GapMeasurement


class _FakeMessage:
    def __init__(self, content: str, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


class _FakeResponse:
    def __init__(self, content: str):
        self.choices = [SimpleNamespace(message=_FakeMessage(content))]
        self.usage = SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2)


@pytest.fixture
def fake_llm():
    captured: dict = {}

    async def fake_acompletion(*, model, messages, **kwargs):
        captured["messages"] = messages
        return _FakeResponse("stub answer")

    with patch("helix.agent.helix_acompletion", side_effect=fake_acompletion):
        yield captured


@pytest.fixture
def archive(tmp_path):
    return SQLiteArchive(str(tmp_path / "archive.sqlite"))


# ---------------------------------------------------------------------------
# Entries as artifacts
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_episodic_write_records_a_memory_entry_genesis_artifact(archive):
    em = EpisodicMemory(":memory:", archive=archive)
    entry = MemoryEntry(
        id="e1",
        scope_key=ScopeKey(Scope.USER, "alice"),
        content={"user_message": "how do I reset my password", "final_output": "use the portal"},
    )
    await em.write(entry)

    assert entry.artifact_ref == ("memory.episodic.e1", 1)
    stored = await em.get("e1")
    assert stored is not None and stored.artifact_ref == ("memory.episodic.e1", 1)

    variant = await archive.by_id("memory.episodic.e1")
    assert variant is not None
    art = variant.artifact
    assert art.kind is ArtifactKind.MEMORY_ENTRY
    assert art.parent_id is None
    assert art.layer == 2
    assert entry_payload(art)["user_message"] == "how do I reset my password"
    assert entry_utility(art) == 0.0
    assert art.metadata["operator"] == "add"
    assert art.metadata["scope"] == "user:alice"


@pytest.mark.asyncio
async def test_episodic_write_without_archive_keeps_chapter_two_behavior():
    em = EpisodicMemory(":memory:")
    entry = MemoryEntry(id="e1", scope_key=ScopeKey(Scope.USER, "alice"), content={"user_message": "hi"})
    await em.write(entry)
    assert entry.artifact_ref is None
    assert (await em.get("e1")).artifact_ref is None


@pytest.mark.asyncio
async def test_semantic_fact_update_produces_a_child_version_not_an_overwrite(archive):
    sm = SemanticMemory(":memory:", archive=archive)
    sk = ScopeKey(Scope.USER, "alice")
    eid = await sm.write_fact("preferred_units", "metric", sk)
    eid2 = await sm.write_fact("preferred_units", "imperial", sk)
    assert eid == eid2, "the index row keeps one id per fact"

    v2 = await archive.by_id(f"memory.semantic.{eid}")
    assert v2 is not None and v2.artifact.version == 2
    assert v2.artifact.parent_id == (f"memory.semantic.{eid}", 1)
    assert entry_payload(v2.artifact)["value"] == "imperial"
    assert v2.artifact.metadata["operator"] == "edit"
    v1 = await archive.by_id(f"memory.semantic.{eid}", 1)
    assert entry_payload(v1.artifact)["value"] == "metric", "history survives the update"
    assert await sm.read_fact("preferred_units", sk) == "imperial"


@pytest.mark.asyncio
async def test_episodic_store_created_before_v03_gains_the_artifact_ref_column(tmp_path):
    import sqlite3

    path = tmp_path / "old.sqlite"
    conn = sqlite3.connect(str(path))
    conn.executescript(
        """
        CREATE TABLE episodic_entries (
            id TEXT PRIMARY KEY, scope TEXT NOT NULL, scope_key TEXT NOT NULL,
            searchable_text TEXT NOT NULL, content_json TEXT NOT NULL, embedding_json TEXT,
            score REAL NOT NULL DEFAULT 0, metadata_json TEXT NOT NULL, created_at TEXT NOT NULL,
            last_accessed_at TEXT, access_count INTEGER NOT NULL DEFAULT 0
        );
        INSERT INTO episodic_entries VALUES ('old', 'user', 'bob', 'x', '{"user_message": "x"}', NULL, 0, '{}', '2026-01-01T00:00:00+00:00', NULL, 0);
        """
    )
    conn.commit()
    conn.close()
    em = EpisodicMemory(path)
    old = await em.get("old")
    assert old is not None and old.artifact_ref is None


# ---------------------------------------------------------------------------
# Memory as a hook pair
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_agent_registers_memory_hooks_on_pre_model_and_session_end():
    em = EpisodicMemory(":memory:")
    agent = Agent(system_prompt=genesis("p", Subtype.PROMPT, "be helpful"), memory_tiers={"episodic": em})
    assert agent.memory_hooks is not None
    assert agent.memory_hooks.inject in agent.hooks._hooks[HookPoint.PRE_MODEL]
    assert agent.memory_hooks.write_back in agent.hooks._hooks[HookPoint.SESSION_END]


@pytest.mark.asyncio
async def test_with_artifacts_clone_keeps_the_memory_hooks(fake_llm):
    em = EpisodicMemory(":memory:")
    agent = Agent(
        system_prompt=genesis("p", Subtype.PROMPT, "be helpful"),
        model="claude-haiku-4-5",
        memory_tiers={"episodic": em},
    )
    candidate = agent.system_prompt.mutate("be terse", created_by="spo")
    clone = agent.with_artifacts({"p": candidate})
    assert clone.memory_hooks is not None and clone.memory_hooks is not agent.memory_hooks
    await clone.run("hello", context=MemoryContext(session_id="s1"))
    assert await em.count() == 1, "the clone wrote back through its own hook"


@pytest.mark.asyncio
async def test_recalled_entries_are_recorded_on_the_trajectory_for_attribution(fake_llm, archive):
    em = EpisodicMemory(":memory:", archive=archive)
    seeded = MemoryEntry(
        id="seed",
        scope_key=ScopeKey(Scope.USER, "alice"),
        content={"user_message": "vpn keeps dropping", "final_output": "switch to the wired profile"},
    )
    await em.write(seeded)
    agent = Agent(
        system_prompt=genesis("p", Subtype.PROMPT, "be helpful"),
        model="claude-haiku-4-5",
        memory_tiers={"episodic": em},
    )
    _, traj = await agent.run("my vpn drops every hour", context=MemoryContext(session_id="s2", user_id="alice"))

    recalled = traj.metadata[RECALLED_KEY]
    assert [r["entry_id"] for r in recalled] == ["seed"]
    assert recalled[0]["artifact_ref"] == ["memory.episodic.seed", 1]
    assert ("memory.episodic.seed", 1) in traj.artifacts_used[0]
    written = traj.metadata[WRITTEN_KEY]
    assert written["artifact_ref"][0].startswith("memory.episodic.")
    assert (await em.get(written["entry_id"])).score == 0.5, "a fresh entry starts half way"
    # The system message still carries the recalled text, as in Chapter 2.
    system = fake_llm["messages"][0]["content"]
    text = system[0]["text"] if isinstance(system, list) else system
    assert "switch to the wired profile" in text


@pytest.mark.asyncio
async def test_memory_is_injected_once_even_when_the_loop_iterates(fake_llm):
    em = EpisodicMemory(":memory:")
    await em.write(MemoryEntry(id="a", scope_key=ScopeKey(Scope.USER, "u"), content={"user_message": "q", "final_output": "a"}))
    agent = Agent(system_prompt=genesis("p", Subtype.PROMPT, "be helpful"), memory_tiers={"episodic": em})
    from helix.trajectory import Trajectory, StepKind

    traj = Trajectory(task="q again")
    traj.metadata["memory_context"] = {"session_id": "s", "user_id": "u", "org_id": None}
    messages = [{"role": "system", "content": "be helpful"}, {"role": "user", "content": "q again"}]
    await agent.memory_hooks.inject(messages=messages, trajectory=traj)
    first = messages[0]["content"]
    traj.append(StepKind.MODEL_CALL, {})
    await agent.memory_hooks.inject(messages=messages, trajectory=traj)
    assert messages[0]["content"] == first
    assert first.count("Relevant past interactions") == 1


@pytest.mark.asyncio
async def test_refused_runs_are_not_written_to_memory():
    em = EpisodicMemory(":memory:")
    agent = Agent(system_prompt=genesis("p", Subtype.PROMPT, "be helpful"), memory_tiers={"episodic": em})
    from helix.trajectory import Trajectory

    traj = Trajectory(task="forbidden")
    traj.metadata["memory_context"] = {"session_id": "s", "user_id": None, "org_id": None}
    traj.metadata["refusal"] = {"reason": "policy", "source": "guard", "phase": "pre_model"}
    traj.complete("[refused] policy")
    await agent.memory_hooks.write_back(trajectory=traj)
    assert await em.count() == 0


# ---------------------------------------------------------------------------
# Archive pool filters
# ---------------------------------------------------------------------------

async def _record(archive, art, score):
    await archive.record(
        Variant(artifact=art, parent=art, search_method="test"),
        GapMeasurement(score=score, signal_id="sig", signal_version=1),
    )


@pytest.mark.asyncio
async def test_best_and_sample_can_be_narrowed_to_a_kind_or_an_artifact_id(archive):
    prompt = genesis("prompt.x", Subtype.PROMPT, "a prompt")
    mem_a = genesis("memory.episodic.a", ArtifactKind.MEMORY_ENTRY, {"entry": {}, "utility": 0.2})
    mem_b = genesis("memory.episodic.b", ArtifactKind.MEMORY_ENTRY, {"entry": {}, "utility": 0.9})
    await _record(archive, prompt, 1.0)
    await _record(archive, mem_a, 0.2)
    await _record(archive, mem_b, 0.9)

    assert (await archive.best(k=1))[0].artifact.id == "prompt.x"
    top_mem = await archive.best(k=5, kind=ArtifactKind.MEMORY_ENTRY)
    assert [v.artifact.id for v in top_mem] == ["memory.episodic.b", "memory.episodic.a"]
    only_a = await archive.best(k=5, artifact_id="memory.episodic.a")
    assert [v.artifact.id for v in only_a] == ["memory.episodic.a"]

    for _ in range(10):
        v = await archive.sample(SamplingStrategy.RANDOM, kind=ArtifactKind.MEMORY_ENTRY)
        assert v.artifact.kind is ArtifactKind.MEMORY_ENTRY
    assert (await archive.sample(SamplingStrategy.BEST, kind=ArtifactKind.MEMORY_ENTRY)).artifact.id == "memory.episodic.b"
    assert (await archive.sample(SamplingStrategy.BEST, artifact_id="memory.episodic.a")).artifact.id == "memory.episodic.a"
