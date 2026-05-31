"""ProceduralMemory tier tests."""

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
from helix.memory.procedural import ProceduralMemory


@pytest.mark.asyncio
async def test_save_and_get_skill():
    pm = ProceduralMemory(":memory:")
    await pm.save_skill(
        name="summarize_paper",
        description="Take a paper abstract and produce 3 bullet takeaways.",
        body="(instructions)",
    )
    skill = await pm.get_skill("summarize_paper", ScopeKey(Scope.GLOBAL, "*"))
    assert skill is not None
    assert skill["name"] == "summarize_paper"
    assert skill["usage_count"] == 0


@pytest.mark.asyncio
async def test_record_use_increments_count():
    pm = ProceduralMemory(":memory:")
    sk = ScopeKey(Scope.GLOBAL, "*")
    await pm.save_skill(name="s", description="d", scope_key=sk)
    await pm.record_use("s", sk)
    await pm.record_use("s", sk)
    skill = await pm.get_skill("s", sk)
    assert skill["usage_count"] == 2


@pytest.mark.asyncio
async def test_list_skills_orders_by_usage_count():
    pm = ProceduralMemory(":memory:")
    sk = ScopeKey(Scope.GLOBAL, "*")
    await pm.save_skill(name="rare", description="rare", scope_key=sk)
    await pm.save_skill(name="popular", description="popular", scope_key=sk)
    for _ in range(3):
        await pm.record_use("popular", sk)
    skills = await pm.list_skills(sk)
    assert [s["name"] for s in skills] == ["popular", "rare"]


@pytest.mark.asyncio
async def test_save_skill_upserts():
    pm = ProceduralMemory(":memory:")
    sk = ScopeKey(Scope.GLOBAL, "*")
    await pm.save_skill(name="s", description="v1", scope_key=sk)
    await pm.save_skill(name="s", description="v2", scope_key=sk)
    skill = await pm.get_skill("s", sk)
    assert skill["description"] == "v2"


@pytest.mark.asyncio
async def test_get_skill_returns_none_when_missing():
    pm = ProceduralMemory(":memory:")
    assert await pm.get_skill("missing", ScopeKey(Scope.GLOBAL, "*")) is None


@pytest.mark.asyncio
async def test_write_via_entry_requires_skill_name():
    pm = ProceduralMemory(":memory:")
    bad = MemoryEntry(
        id="x",
        scope_key=ScopeKey(Scope.GLOBAL, "*"),
        content={"description": "no name"},
    )
    with pytest.raises(ValueError):
        await pm.write(bad)


@pytest.mark.asyncio
async def test_read_via_query_returns_skills_in_scope():
    pm = ProceduralMemory(":memory:")
    sk = ScopeKey(Scope.GLOBAL, "*")
    await pm.save_skill(name="alpha", description="alpha skill", scope_key=sk)
    await pm.save_skill(name="beta", description="beta skill", scope_key=sk)
    hits = await pm.read(
        MemoryQuery(text="skill", k=10),
        MemoryContext(),  # global is always visible
    )
    names = {h.content["skill_name"] for h in hits}
    assert names == {"alpha", "beta"}


@pytest.mark.asyncio
async def test_evict_by_max_entries_drops_excess_and_counts():
    """max_entries must evict and be counted (fix #7: procedural ignored it)."""
    pm = ProceduralMemory(":memory:")
    sk = ScopeKey(Scope.GLOBAL, "*")
    for i in range(5):
        await pm.save_skill(name=f"s{i}", description="d", scope_key=sk)
    dropped = await pm.evict(EvictionPolicy(max_entries=2))
    assert dropped == 3
    assert len(await pm.list_skills(sk)) == 2
