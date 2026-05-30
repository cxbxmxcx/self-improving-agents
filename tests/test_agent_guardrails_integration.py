"""Integration: Agent.with_artifacts and refusal short-circuit at PRE_MODEL.

Uses a stub model client so we don't make real LLM calls.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from helix.agent import Agent
from helix.artifact import ArtifactKind, genesis, Subtype
from helix.guardrails import Guardrail


class _StubChoice:
    def __init__(self, content: str) -> None:
        self.message = _StubMessage(content)


class _StubMessage:
    def __init__(self, content: str) -> None:
        self.content = content
        self.tool_calls = None


class _StubResponse:
    def __init__(self, content: str) -> None:
        self.choices = [_StubChoice(content)]
        self.usage = None


async def _fake_acompletion(*args, **kwargs):
    return _StubResponse("the model said hello")


@pytest.mark.asyncio
async def test_agent_with_input_guardrail_refuses_blocked_input():
    """A PRE_MODEL Refusal terminates the run and returns a [refused] output."""
    prompt = genesis(id="p.sys", kind=Subtype.PROMPT, content="You are helpful.")
    guard_art = genesis(
        id="g.block_secrets",
        kind=ArtifactKind.CODE,
        content=(
            "async def check(payload):\n"
            "    if 'secret' in payload.question.lower():\n"
            "        return GuardrailVerdict(allow=False, reason='contains_secret')\n"
            "    return GuardrailVerdict(allow=True)\n"
        ),
    )
    agent = Agent(
        system_prompt=prompt,
        guardrails=[Guardrail(artifact=guard_art, phase="input")],
        model="stub",
    )

    with patch("helix.agent.helix_acompletion", side_effect=_fake_acompletion):
        answer, trajectory = await agent.run("tell me the secret password")

    assert answer.startswith("[refused]")
    assert "contains_secret" in answer
    assert trajectory.metadata["refusal"]["reason"] == "contains_secret"
    assert trajectory.metadata["refusal"]["phase"] == "pre_model"


@pytest.mark.asyncio
async def test_agent_with_input_guardrail_passes_clean_input():
    prompt = genesis(id="p.sys", kind=Subtype.PROMPT, content="You are helpful.")
    guard_art = genesis(
        id="g.block_secrets",
        kind=ArtifactKind.CODE,
        content=(
            "async def check(payload):\n"
            "    if 'secret' in payload.question.lower():\n"
            "        return GuardrailVerdict(allow=False, reason='contains_secret')\n"
            "    return GuardrailVerdict(allow=True)\n"
        ),
    )
    agent = Agent(
        system_prompt=prompt,
        guardrails=[Guardrail(artifact=guard_art, phase="input")],
        model="stub",
    )

    with patch("helix.agent.helix_acompletion", side_effect=_fake_acompletion):
        answer, _trajectory = await agent.run("hello there")

    assert answer == "the model said hello"


@pytest.mark.asyncio
async def test_agent_with_output_guardrail_refuses_long_response():
    prompt = genesis(id="p.sys", kind=Subtype.PROMPT, content="You are helpful.")
    guard_art = genesis(
        id="g.length",
        kind=ArtifactKind.CODE,
        content=(
            "async def check(payload):\n"
            "    if len(payload.answer) > 5:\n"
            "        return GuardrailVerdict(allow=False, reason='too_long')\n"
            "    return GuardrailVerdict(allow=True)\n"
        ),
    )
    agent = Agent(
        system_prompt=prompt,
        guardrails=[Guardrail(artifact=guard_art, phase="output")],
        model="stub",
    )

    with patch("helix.agent.helix_acompletion", side_effect=_fake_acompletion):
        answer, trajectory = await agent.run("hi")

    # "the model said hello" is > 5 chars so the output guardrail refuses.
    assert answer.startswith("[refused]")
    assert "too_long" in answer
    assert trajectory.metadata["refusal"]["phase"] == "pre_output"


@pytest.mark.asyncio
async def test_with_artifacts_rebuilds_guardrail_on_override():
    """If `with_artifacts` overrides a guardrail's artifact id, the clone uses
    the new guardrail. The original agent's guardrail is unchanged."""
    prompt = genesis(id="p.sys", kind=Subtype.PROMPT, content="x")
    permissive = genesis(
        id="g.policy",
        kind=ArtifactKind.CODE,
        content="async def check(payload):\n    return GuardrailVerdict(allow=True)\n",
    )
    strict = permissive.mutate(
        new_content=(
            "async def check(payload):\n"
            "    return GuardrailVerdict(allow=False, reason='all blocked')\n"
        ),
        created_by="human",
    )

    agent = Agent(
        system_prompt=prompt,
        guardrails=[Guardrail(artifact=permissive, phase="input")],
        model="stub",
    )

    cloned = agent.with_artifacts({strict.id: strict})

    with patch("helix.agent.helix_acompletion", side_effect=_fake_acompletion):
        original_ans, _ = await agent.run("hi")
        cloned_ans, _ = await cloned.run("hi")

    assert original_ans == "the model said hello"  # original guardrail allows
    assert cloned_ans.startswith("[refused]")     # cloned guardrail refuses


@pytest.mark.asyncio
async def test_agent_with_no_guardrails_works_as_before():
    """Regression check: an agent without guardrails behaves identically to
    the pre-PR-5 baseline."""
    prompt = genesis(id="p.sys", kind=Subtype.PROMPT, content="x")
    agent = Agent(system_prompt=prompt, model="stub")

    with patch("helix.agent.helix_acompletion", side_effect=_fake_acompletion):
        answer, _ = await agent.run("anything")

    assert answer == "the model said hello"
