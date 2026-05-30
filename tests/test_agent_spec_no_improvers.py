"""Tests for an agent spec that intentionally declares no improvers.

The `researcher` spec exports `build()` and `build_genesis_prompt()` but
not `build_improvers()` or `list_improvable_artifacts()`. This exercises
the platform's no-improvers fallback paths:

  - helix.agents.describe_agent reports has_improvers=False
  - helix.agents.load_improvers returns []
  - helix.agents.list_improvable_artifacts falls back to [agent.system_prompt]

These are spec-loader contract tests; no LLM calls, no corpus access.
"""

from __future__ import annotations

import pytest

from helix.agents import (
    describe_agent,
    list_agents,
    list_improvable_artifacts,
    load_genesis_prompt,
    load_improvers,
)


def test_researcher_spec_is_discoverable():
    """The researcher module appears in list_agents()."""
    assert "researcher" in list_agents()


def test_describe_researcher_reports_no_improvers():
    """describe_agent surfaces the absent build_improvers function."""
    meta = describe_agent("researcher")
    assert meta["has_build"] is True
    assert meta["has_genesis"] is True
    assert meta["has_improvers"] is False
    assert meta["has_list_artifacts"] is False


def test_load_genesis_prompt_works_for_researcher():
    """Genesis prompt loads without needing the corpus or LLM."""
    art = load_genesis_prompt("researcher")
    assert art.id == "prompt.researcher.system"
    assert art.parent_id is None
    assert art.created_by == "human"


def test_load_improvers_returns_empty_for_researcher():
    """load_improvers() must not raise when build_improvers is absent.

    We construct a fake Agent rather than calling load_agent('researcher')
    because that spec opens a LanceDB corpus on build(); the loader
    contract is what we're testing here, not the corpus dependency.
    """
    from helix.agent import Agent
    from helix.artifact import ArtifactKind, genesis, Subtype

    fake_agent = Agent(
        system_prompt=genesis("p.fake", Subtype.PROMPT, "fake"),
        model="claude-haiku-4-5",
    )
    result = load_improvers("researcher", fake_agent)
    assert result == []


def test_list_improvable_artifacts_falls_back_for_researcher():
    """When the spec doesn't export list_improvable_artifacts, the loader
    returns [agent.system_prompt] as the canonical improvable artifact."""
    from helix.agent import Agent
    from helix.artifact import ArtifactKind, genesis, Subtype

    seed = genesis("p.researcher.test", Subtype.PROMPT, "hello")
    fake_agent = Agent(system_prompt=seed, model="claude-haiku-4-5")

    arts = list_improvable_artifacts("researcher", fake_agent)
    assert len(arts) == 1
    assert arts[0].id == "p.researcher.test"


def test_researcher_genesis_differs_from_helpdesk():
    """Two specs with different prompt IDs do not collide in the loader."""
    a = load_genesis_prompt("researcher")
    b = load_genesis_prompt("helpdesk")
    assert a.id != b.id
    assert a.content != b.content
