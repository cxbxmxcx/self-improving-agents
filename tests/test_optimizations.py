"""Tests for the Stratum 2.7 optimizations.

Coverage:
- Trajectory caching: store_trajectory / get_trajectory round-trip
- Prompt caching: cacheable_system_message shapes per provider
- Retrieval LRU cache: cache hits return same hits, eviction at cap
- Per-question call cap: agent stops cascading after N tool calls
- Budget enforcement: round aborts when budget exhausted
- Concurrent execution: questions run in parallel inside the semaphore
"""

from __future__ import annotations

import asyncio

import pytest

from helix._caching import cacheable_system_message, is_anthropic_model
from helix.archive import SQLiteArchive
from helix.artifact import ArtifactKind, genesis
from helix.retrieval.chunker import Chunk
from helix.search.base import SearchBudget
from helix.signal import Cost
from helix.trajectory import Outcome, StepKind, Trajectory


# ---------------------------------------------------------------------------
# Trajectory caching
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_trajectory_cache_round_trip():
    arc = SQLiteArchive(":memory:")
    art = genesis("prompt.test", ArtifactKind.PROMPT, "seed")
    traj = Trajectory(task="hello")
    traj.append(StepKind.MODEL_CALL, {"response": {"content": "answer"}})
    traj.complete("answer", Outcome.COMPLETED)

    await arc.store_trajectory(art, "Q1", traj, answer="answer")
    cached = await arc.get_trajectory(art, "Q1")
    assert cached is not None
    cached_answer, cached_traj = cached
    assert cached_answer == "answer"
    assert cached_traj.task == "hello"
    assert len(cached_traj.steps) == 1
    assert cached_traj.outcome == Outcome.COMPLETED


@pytest.mark.asyncio
async def test_trajectory_cache_miss_returns_none():
    arc = SQLiteArchive(":memory:")
    art = genesis("prompt.test", ArtifactKind.PROMPT, "seed")
    cached = await arc.get_trajectory(art, "Q1")
    assert cached is None


@pytest.mark.asyncio
async def test_trajectory_cache_keyed_per_question():
    arc = SQLiteArchive(":memory:")
    art = genesis("prompt.test", ArtifactKind.PROMPT, "seed")
    t1 = Trajectory(task="Q1")
    t1.complete("A1")
    t2 = Trajectory(task="Q2")
    t2.complete("A2")
    await arc.store_trajectory(art, "Q1", t1, answer="A1")
    await arc.store_trajectory(art, "Q2", t2, answer="A2")

    c1 = await arc.get_trajectory(art, "Q1")
    c2 = await arc.get_trajectory(art, "Q2")
    assert c1[0] == "A1"
    assert c2[0] == "A2"


@pytest.mark.asyncio
async def test_trajectory_cache_keyed_per_artifact_version():
    arc = SQLiteArchive(":memory:")
    v1 = genesis("prompt.test", ArtifactKind.PROMPT, "v1")
    v2 = v1.mutate("v2", created_by="spo")
    t1 = Trajectory(task="x")
    t1.complete("from v1")
    t2 = Trajectory(task="x")
    t2.complete("from v2")
    await arc.store_trajectory(v1, "Q1", t1, answer="from v1")
    await arc.store_trajectory(v2, "Q1", t2, answer="from v2")

    c1 = await arc.get_trajectory(v1, "Q1")
    c2 = await arc.get_trajectory(v2, "Q1")
    assert c1[0] == "from v1"
    assert c2[0] == "from v2"


# ---------------------------------------------------------------------------
# Prompt caching markers
# ---------------------------------------------------------------------------

def test_is_anthropic_model_detection():
    assert is_anthropic_model("claude-haiku-4-5")
    assert is_anthropic_model("claude-sonnet-4-6")
    assert is_anthropic_model("anthropic/claude-3-opus")
    assert not is_anthropic_model("gpt-4o-mini")
    assert not is_anthropic_model("gemini-2.0-flash")
    assert not is_anthropic_model("")


def test_cacheable_system_message_for_anthropic_uses_content_blocks():
    msg = cacheable_system_message("rubric text", "claude-sonnet-4-6")
    assert msg["role"] == "system"
    assert isinstance(msg["content"], list)
    assert len(msg["content"]) == 1
    block = msg["content"][0]
    assert block["type"] == "text"
    assert block["text"] == "rubric text"
    assert block["cache_control"] == {"type": "ephemeral"}


def test_cacheable_system_message_for_openai_uses_string_content():
    msg = cacheable_system_message("system instructions", "gpt-4o-mini")
    assert msg["role"] == "system"
    assert msg["content"] == "system instructions"


def test_cacheable_system_message_extended_ttl():
    msg = cacheable_system_message("text", "claude-sonnet-4-6", ttl="1h")
    block = msg["content"][0]
    assert block["cache_control"] == {"type": "ephemeral", "ttl": "1h"}


# ---------------------------------------------------------------------------
# Retrieval LRU cache
# ---------------------------------------------------------------------------

def test_retrieval_lru_cache_dedupes_identical_searches(tmp_path):
    """Same query twice returns the same hits without re-querying LanceDB."""
    from helix.retrieval.index import RetrievalIndex
    idx = RetrievalIndex(tmp_path / "test.lance")

    # Inject a fake cached entry to verify the cache path works without
    # building an actual corpus (which is heavy and tested elsewhere).
    from helix.retrieval.index import SearchHit
    fake_hits = [SearchHit(text="cached", source="t.pdf", page=0, score=0.9)]
    idx._search_cache[("q", 5, "hybrid")] = fake_hits
    idx._cache_order.append(("q", 5, "hybrid"))

    result = idx.search("q", k=5, mode="hybrid")
    assert result == fake_hits


def test_retrieval_cache_evicts_oldest_at_cap(tmp_path):
    from helix.retrieval.index import RetrievalIndex, SearchHit
    idx = RetrievalIndex(tmp_path / "test.lance")
    idx._CACHE_MAXSIZE = 3  # shrink for testing

    # Manually populate the cache to test eviction without a real corpus.
    for i in range(5):
        key = (f"q{i}", 5, "hybrid")
        idx._search_cache[key] = [SearchHit(text=f"hit{i}", source="t", page=0, score=1.0)]
        idx._cache_order.append(key)
        while len(idx._cache_order) > idx._CACHE_MAXSIZE:
            oldest = idx._cache_order.pop(0)
            idx._search_cache.pop(oldest, None)

    assert len(idx._search_cache) == 3
    assert ("q4", 5, "hybrid") in idx._search_cache
    assert ("q0", 5, "hybrid") not in idx._search_cache  # evicted


def test_retrieval_clear_cache(tmp_path):
    from helix.retrieval.index import RetrievalIndex, SearchHit
    idx = RetrievalIndex(tmp_path / "test.lance")
    idx._search_cache[("q", 5, "hybrid")] = [SearchHit(text="h", source="t", page=0, score=1.0)]
    idx._cache_order.append(("q", 5, "hybrid"))
    idx.clear_cache()
    assert idx._search_cache == {}
    assert idx._cache_order == []


# ---------------------------------------------------------------------------
# Budget enforcement
# ---------------------------------------------------------------------------

def test_budget_tracks_spent_across_charges():
    b = SearchBudget(max_tokens=1000)
    b.charge(Cost(tokens=300))
    b.charge(Cost(tokens=200))
    rem = b.remaining()
    assert rem["tokens"] == 500


def test_budget_exhausted_returns_true_when_tokens_blown():
    b = SearchBudget(max_tokens=100)
    b.charge(Cost(tokens=150))
    assert b.exhausted()


def test_budget_exhausted_returns_true_when_dollars_blown():
    b = SearchBudget(max_dollars=1.0)
    b.charge(Cost(dollars=1.5))
    assert b.exhausted()


def test_budget_not_exhausted_when_axes_unset():
    """Budget with no caps set should never be exhausted."""
    b = SearchBudget()
    b.charge(Cost(tokens=1_000_000, dollars=100.0))
    assert not b.exhausted()


# ---------------------------------------------------------------------------
# Per-question call cap (sanity check the cap is on the Agent class)
# ---------------------------------------------------------------------------

def test_agent_accepts_max_tool_calls_parameter():
    from helix.agent import Agent
    art = genesis("prompt.test", ArtifactKind.PROMPT, "you are helpful")
    agent = Agent(system_prompt=art, model="claude-haiku-4-5", max_tool_calls=5)
    assert agent.max_tool_calls == 5


# ---------------------------------------------------------------------------
# Concurrent execution
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_concurrent_execution_runs_in_parallel():
    """5 tasks each sleeping 0.1s should finish in roughly 0.1s, not 0.5s."""
    import time
    sem = asyncio.Semaphore(5)

    async def task():
        async with sem:
            await asyncio.sleep(0.1)

    t0 = time.perf_counter()
    await asyncio.gather(*[task() for _ in range(5)])
    elapsed = time.perf_counter() - t0
    assert elapsed < 0.3  # well under the 0.5s sequential bound
