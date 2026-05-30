"""Agent + memory tier integration tests.

Verifies that agent.run(context=MemoryContext(...)) properly:
  - Injects episodic memory into the system prompt
  - Injects semantic facts into the system prompt
  - Writes the completed trajectory to episodic memory at session end

Uses a patched litellm.acompletion so no real LLM calls are made.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from helix.agent import Agent
from helix.artifact import ArtifactKind, genesis, Subtype
from helix.memory.base import (
    MemoryContext,
    MemoryEntry,
    Scope,
    ScopeKey,
)
from helix.memory.episodic import EpisodicMemory
from helix.memory.semantic import SemanticMemory


class _FakeMessage:
    def __init__(self, content: str, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


class _FakeUsage:
    prompt_tokens = 100
    completion_tokens = 50
    total_tokens = 150


class _FakeChoice:
    def __init__(self, content: str):
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content: str):
        self.choices = [_FakeChoice(content)]
        self.usage = _FakeUsage()


@pytest.fixture
def fake_llm():
    """Patch helix_acompletion so the agent never makes real LLM calls."""
    captured: dict = {}

    async def fake_acompletion(*, model, messages, **kwargs):
        captured["model"] = model
        captured["messages"] = messages
        captured["kwargs"] = kwargs
        return _FakeResponse("stub answer to the question")

    with patch("helix.agent.helix_acompletion", side_effect=fake_acompletion) as mock:
        yield captured


@pytest.mark.asyncio
async def test_agent_runs_without_memory_when_none_attached(fake_llm):
    agent = Agent(
        system_prompt=genesis("p", Subtype.PROMPT, "you are helpful"),
        model="claude-haiku-4-5",
    )
    answer, traj = await agent.run("hello")
    assert answer == "stub answer to the question"
    assert traj.outcome.value == "completed"
    # No memory tiers; nothing should have been read or written


@pytest.mark.asyncio
async def test_agent_writes_to_episodic_on_session_end(fake_llm):
    em = EpisodicMemory(":memory:")
    agent = Agent(
        system_prompt=genesis("p", Subtype.PROMPT, "you are helpful"),
        model="claude-haiku-4-5",
        memory_tiers={"episodic": em},
    )
    await agent.run("what's the capital of france?", context=MemoryContext(session_id="s1"))
    count = await em.count()
    assert count == 1
    recent = await em.list_recent(n=1)
    assert recent[0].content["user_message"] == "what's the capital of france?"
    assert recent[0].content["final_output"] == "stub answer to the question"


@pytest.mark.asyncio
async def test_episodic_recall_appears_in_system_prompt(fake_llm):
    em = EpisodicMemory(":memory:")
    # Seed an existing episodic entry for this user
    await em.write(MemoryEntry(
        id="e1",
        scope_key=ScopeKey(Scope.USER, "alice"),
        content={"user_message": "I work on Linux servers", "final_output": "noted"},
    ))

    agent = Agent(
        system_prompt=genesis("p", Subtype.PROMPT, "you are helpful"),
        model="claude-haiku-4-5",
        memory_tiers={"episodic": em},
    )
    await agent.run("got it", context=MemoryContext(user_id="alice"))

    # The system message should include the prior interaction
    messages = fake_llm["messages"]
    system_msg = messages[0]
    # Anthropic-style cacheable system has content as a list of blocks
    if isinstance(system_msg["content"], list):
        system_text = system_msg["content"][0]["text"]
    else:
        system_text = system_msg["content"]
    assert "I work on Linux servers" in system_text


@pytest.mark.asyncio
async def test_semantic_facts_appear_in_system_prompt(fake_llm):
    sm = SemanticMemory(":memory:")
    await sm.write_fact("preferred_units", "metric", ScopeKey(Scope.USER, "alice"))
    await sm.write_fact("os", "Linux", ScopeKey(Scope.USER, "alice"))

    agent = Agent(
        system_prompt=genesis("p", Subtype.PROMPT, "you are helpful"),
        model="claude-haiku-4-5",
        memory_tiers={"semantic": sm},
    )
    await agent.run("convert to celsius", context=MemoryContext(user_id="alice"))

    messages = fake_llm["messages"]
    system_msg = messages[0]
    if isinstance(system_msg["content"], list):
        system_text = system_msg["content"][0]["text"]
    else:
        system_text = system_msg["content"]
    # At least one of the facts should appear
    assert "preferred_units" in system_text or "metric" in system_text


@pytest.mark.asyncio
async def test_scope_isolation_in_episodic_recall(fake_llm):
    """Alice's memory should not bleed into Bob's session prompt."""
    em = EpisodicMemory(":memory:")
    await em.write(MemoryEntry(
        id="e_alice",
        scope_key=ScopeKey(Scope.USER, "alice"),
        content={"user_message": "I'm an alice-specific fact", "final_output": "ok"},
    ))

    agent = Agent(
        system_prompt=genesis("p", Subtype.PROMPT, "you are helpful"),
        model="claude-haiku-4-5",
        memory_tiers={"episodic": em},
    )
    await agent.run("hello", context=MemoryContext(user_id="bob"))

    messages = fake_llm["messages"]
    system_msg = messages[0]
    if isinstance(system_msg["content"], list):
        system_text = system_msg["content"][0]["text"]
    else:
        system_text = system_msg["content"]
    assert "alice-specific fact" not in system_text


@pytest.mark.asyncio
async def test_agent_run_without_context_skips_memory_write(fake_llm):
    """Stateless mode: no session_id, no episodic write."""
    em = EpisodicMemory(":memory:")
    agent = Agent(
        system_prompt=genesis("p", Subtype.PROMPT, "you are helpful"),
        model="claude-haiku-4-5",
        memory_tiers={"episodic": em},
    )
    await agent.run("hello")  # no context
    assert await em.count() == 0


@pytest.mark.asyncio
async def test_memory_write_failure_does_not_break_run(fake_llm):
    """If episodic memory raises on write, the agent should still return its answer."""

    class _BrokenEpisodic:
        tier_name = "episodic"

        async def read(self, query, context):
            return []

        async def write(self, entry):
            raise RuntimeError("disk full")

        async def score(self, entry):
            return 0.0

        async def evict(self, policy):
            return 0

        async def consolidate(self):
            from helix.memory.base import ConsolidationReport
            return ConsolidationReport(tier="episodic")

    agent = Agent(
        system_prompt=genesis("p", Subtype.PROMPT, "you are helpful"),
        model="claude-haiku-4-5",
        memory_tiers={"episodic": _BrokenEpisodic()},
    )
    answer, _ = await agent.run("hello", context=MemoryContext(session_id="s1"))
    assert answer == "stub answer to the question"
