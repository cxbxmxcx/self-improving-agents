"""MemRL's recall policy: learned utility changes what gets retrieved.

The tier's default keeps chapter 2 behavior (similarity and recency only);
with a utility weight, a higher-Q entry outranks an equally similar one,
which is the only way a Q-value ever changes the next run's behavior.
"""

from __future__ import annotations

import pytest

from helix.memory.base import MemoryContext, MemoryEntry, MemoryQuery, Scope, ScopeKey
from helix.memory.episodic import EpisodicMemory


def _twin_entries():
    sk = ScopeKey(Scope.USER, "u")
    high = MemoryEntry(id="high", scope_key=sk, score=0.9,
                       content={"user_message": "vpn drops on wifi", "final_output": "use the wired profile"})
    low = MemoryEntry(id="low", scope_key=sk, score=0.1,
                      content={"user_message": "vpn drops on wifi", "final_output": "reboot the laptop"})
    return high, low


@pytest.mark.asyncio
async def test_zero_weight_keeps_similarity_and_recency_ranking():
    em = EpisodicMemory(":memory:")
    high, low = _twin_entries()
    await em.write(low)
    await em.write(high)
    hits = await em.read(MemoryQuery(text="vpn drops on wifi", k=2), MemoryContext(user_id="u"))
    assert {h.id for h in hits} == {"high", "low"}, "both stay retrievable"


@pytest.mark.asyncio
async def test_a_higher_utility_entry_outranks_an_equally_similar_one():
    em = EpisodicMemory(":memory:", utility_weight=0.35)
    high, low = _twin_entries()
    await em.write(low)
    await em.write(high)
    hits = await em.read(MemoryQuery(text="vpn drops on wifi", k=1), MemoryContext(user_id="u"))
    assert [h.id for h in hits] == ["high"]


@pytest.mark.asyncio
async def test_utility_weight_one_ranks_purely_by_value():
    em = EpisodicMemory(":memory:", utility_weight=1.0)
    sk = ScopeKey(Scope.USER, "u")
    await em.write(MemoryEntry(id="relevant", scope_key=sk, score=0.2,
                               content={"user_message": "printer offline", "final_output": "restart the spooler"}))
    await em.write(MemoryEntry(id="valuable", scope_key=sk, score=0.9,
                               content={"user_message": "vpn drops", "final_output": "wired profile"}))
    hits = await em.read(MemoryQuery(text="printer offline", k=1), MemoryContext(user_id="u"))
    assert [h.id for h in hits] == ["valuable"], "weight 1.0 ignores the query, which is why it is not the default"


@pytest.mark.asyncio
async def test_the_weight_is_clamped_to_the_unit_interval():
    assert EpisodicMemory(":memory:", utility_weight=2.0).utility_weight == 1.0
    assert EpisodicMemory(":memory:", utility_weight=-1.0).utility_weight == 0.0
