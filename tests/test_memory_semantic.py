"""SemanticMemory tier tests."""

from __future__ import annotations

import pytest

from helix.memory.base import (
    EvictionPolicy,
    MemoryContext,
    MemoryEntry,
    MemoryQuery,
    Scope,
    ScopeKey,
)
from helix.memory.semantic import SemanticMemory


@pytest.mark.asyncio
async def test_write_fact_and_read_fact_round_trip():
    sm = SemanticMemory(":memory:")
    sk = ScopeKey(Scope.USER, "alice")
    await sm.write_fact("preferred_units", "metric", sk, description="user prefers metric")
    assert await sm.read_fact("preferred_units", sk) == "metric"


@pytest.mark.asyncio
async def test_write_fact_upserts():
    sm = SemanticMemory(":memory:")
    sk = ScopeKey(Scope.USER, "alice")
    await sm.write_fact("preferred_units", "imperial", sk)
    await sm.write_fact("preferred_units", "metric", sk)
    assert await sm.read_fact("preferred_units", sk) == "metric"


@pytest.mark.asyncio
async def test_list_facts_returns_all_for_scope():
    sm = SemanticMemory(":memory:")
    sk = ScopeKey(Scope.USER, "alice")
    await sm.write_fact("preferred_units", "metric", sk)
    await sm.write_fact("os", "linux", sk)
    facts = await sm.list_facts(sk)
    assert dict(facts) == {"preferred_units": "metric", "os": "linux"}


@pytest.mark.asyncio
async def test_scope_isolation():
    sm = SemanticMemory(":memory:")
    await sm.write_fact("k", "alice_value", ScopeKey(Scope.USER, "alice"))
    await sm.write_fact("k", "bob_value", ScopeKey(Scope.USER, "bob"))
    assert await sm.read_fact("k", ScopeKey(Scope.USER, "alice")) == "alice_value"
    assert await sm.read_fact("k", ScopeKey(Scope.USER, "bob")) == "bob_value"


@pytest.mark.asyncio
async def test_read_via_query_interface_returns_entries():
    sm = SemanticMemory(":memory:")
    sk = ScopeKey(Scope.USER, "alice")
    await sm.write_fact("preferred_units", "metric", sk, description="user prefers metric units")

    hits = await sm.read(
        MemoryQuery(text="units", k=5),
        MemoryContext(user_id="alice"),
    )
    assert len(hits) >= 1
    assert hits[0].content["key"] == "preferred_units"


@pytest.mark.asyncio
async def test_write_via_entry_interface_requires_key_value():
    sm = SemanticMemory(":memory:")
    bad = MemoryEntry(
        id="x",
        scope_key=ScopeKey(Scope.USER, "u"),
        content={"not_key": "x"},
    )
    with pytest.raises(ValueError):
        await sm.write(bad)


@pytest.mark.asyncio
async def test_read_fact_returns_none_when_missing():
    sm = SemanticMemory(":memory:")
    assert await sm.read_fact("missing", ScopeKey(Scope.USER, "u")) is None
