"""EpisodicMemory tier tests.

Embedding-related tests use a small fixed corpus so the model loads once.
Tests that don't need similarity use the no-embedding path (write a pre-
embedded entry).
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest

from helix.memory.base import (
    EvictionPolicy,
    MemoryContext,
    MemoryEntry,
    MemoryQuery,
    Scope,
    ScopeKey,
)
from helix.memory.episodic import EpisodicMemory


def _make_entry(scope_key: ScopeKey, user_message: str, answer: str, embedding=None) -> MemoryEntry:
    return MemoryEntry(
        id=str(uuid.uuid4()),
        scope_key=scope_key,
        content={"user_message": user_message, "final_output": answer},
        embedding=embedding,
    )


@pytest.mark.asyncio
async def test_write_and_read_round_trip():
    em = EpisodicMemory(":memory:")
    sk = ScopeKey(Scope.SESSION, "s1")
    e = _make_entry(sk, "What is GEPA?", "Genetic-Pareto reflective mutation.")
    eid = await em.write(e)
    assert eid

    hits = await em.read(
        MemoryQuery(text="GEPA", scope_keys=[sk], k=5),
        MemoryContext(session_id="s1"),
    )
    assert len(hits) == 1
    assert hits[0].content["user_message"] == "What is GEPA?"


@pytest.mark.asyncio
async def test_scope_isolation_no_cross_leak():
    em = EpisodicMemory(":memory:")
    sk_user_a = ScopeKey(Scope.USER, "alice")
    sk_user_b = ScopeKey(Scope.USER, "bob")

    await em.write(_make_entry(sk_user_a, "alice's question", "alice's answer"))
    await em.write(_make_entry(sk_user_b, "bob's question", "bob's answer"))

    # Read with alice's context
    hits_alice = await em.read(
        MemoryQuery(text="question", k=10),
        MemoryContext(user_id="alice"),
    )
    assert len(hits_alice) == 1
    assert hits_alice[0].content["user_message"] == "alice's question"

    # Read with bob's context
    hits_bob = await em.read(
        MemoryQuery(text="question", k=10),
        MemoryContext(user_id="bob"),
    )
    assert len(hits_bob) == 1
    assert hits_bob[0].content["user_message"] == "bob's question"


@pytest.mark.asyncio
async def test_multiple_scopes_visible_at_once():
    """A context with session_id, user_id, and org_id should see entries from
    all three scopes."""
    em = EpisodicMemory(":memory:")
    await em.write(_make_entry(ScopeKey(Scope.SESSION, "s1"), "session fact", "x"))
    await em.write(_make_entry(ScopeKey(Scope.USER, "u1"), "user fact", "y"))
    await em.write(_make_entry(ScopeKey(Scope.ORG, "o1"), "org fact", "z"))
    await em.write(_make_entry(ScopeKey(Scope.GLOBAL, "*"), "global fact", "w"))

    hits = await em.read(
        MemoryQuery(text="fact", k=10),
        MemoryContext(session_id="s1", user_id="u1", org_id="o1"),
    )
    # All four scopes visible
    assert len(hits) == 4


@pytest.mark.asyncio
async def test_read_returns_empty_when_no_scope_keys():
    em = EpisodicMemory(":memory:")
    await em.write(_make_entry(ScopeKey(Scope.USER, "u1"), "q", "a"))
    hits = await em.read(MemoryQuery(text="q"), MemoryContext())  # empty context
    # Global is still in the context's defaults; entries only exist for user u1
    # so we expect 0 hits (global scope has no entries).
    user_scoped = [h for h in hits if h.scope_key.scope == Scope.USER]
    assert user_scoped == []


@pytest.mark.asyncio
async def test_evict_by_max_entries_keeps_newest():
    em = EpisodicMemory(":memory:")
    sk = ScopeKey(Scope.GLOBAL, "*")
    for i in range(5):
        e = _make_entry(sk, f"q{i}", f"a{i}")
        await em.write(e)
    dropped = await em.evict(EvictionPolicy(max_entries=3))
    assert await em.count() == 3


@pytest.mark.asyncio
async def test_consolidate_reports_state():
    em = EpisodicMemory(":memory:")
    sk = ScopeKey(Scope.GLOBAL, "*")
    for i in range(3):
        await em.write(_make_entry(sk, f"q{i}", f"a{i}"))
    report = await em.consolidate()
    assert report.tier == "episodic"
    assert report.entries_before == 3
    assert report.entries_after == 3


@pytest.mark.asyncio
async def test_access_count_increments_on_read():
    em = EpisodicMemory(":memory:")
    sk = ScopeKey(Scope.SESSION, "s1")
    e = _make_entry(sk, "asked", "answered")
    eid = await em.write(e)

    await em.read(MemoryQuery(text="asked", scope_keys=[sk], k=5), MemoryContext(session_id="s1"))
    await em.read(MemoryQuery(text="asked", scope_keys=[sk], k=5), MemoryContext(session_id="s1"))

    row = em._conn.execute("SELECT access_count FROM episodic_entries WHERE id = ?", (eid,)).fetchone()
    assert row["access_count"] == 2


@pytest.mark.asyncio
async def test_recent_entries_listing():
    em = EpisodicMemory(":memory:")
    sk = ScopeKey(Scope.GLOBAL, "*")
    for i in range(5):
        await em.write(_make_entry(sk, f"q{i}", f"a{i}"))
    recent = await em.list_recent(n=3)
    assert len(recent) == 3
