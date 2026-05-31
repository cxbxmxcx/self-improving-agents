"""Agent composite binding and replace-enforcement. SPEC §18.3, §18.7."""

from __future__ import annotations

import pytest

from helix.agent import Agent
from helix.archive import SQLiteArchive
from helix.artifact import Subtype, compose, genesis
from helix.improvement import ImproverPolicy, OfflineImprover, OnlineImprover


def _agent_with_metacognition():
    prompt = genesis("p.sys", Subtype.PROMPT, "you are helpful")
    planner = genesis("plan", Subtype.PLANNER, "decompose the task")
    monitor = genesis("mon", Subtype.MONITOR, "watch for loops")
    meta = compose(
        "meta",
        [(planner, "planner"), (monitor, "monitor")],
        subtype=Subtype.METACOGNITION,
    )
    agent = Agent(system_prompt=prompt, model="claude-haiku-4-5", composites=[meta])
    return agent, meta


def test_agent_finds_composite_and_lists_its_constituents():
    agent, meta = _agent_with_metacognition()
    assert agent.find_artifact("meta") is meta
    assert agent.find_artifact("meta").layer == 3
    assert agent.composite_constituent_ids() == {"plan", "mon"}


def test_with_artifacts_carries_and_swaps_a_composite():
    agent, meta = _agent_with_metacognition()
    meta_v2 = meta.mutate(meta.content, created_by="composed")
    clone = agent.with_artifacts({"meta": meta_v2})
    assert clone.find_artifact("meta").version == 2
    assert agent.find_artifact("meta").version == 1  # original untouched


def test_offline_improver_refuses_a_composite_constituent():
    agent, _ = _agent_with_metacognition()
    with pytest.raises(ValueError, match="constituent"):
        OfflineImprover(
            agent=agent,
            target_artifact_id="plan",  # a constituent of meta
            signal=object(), search=object(),
            archive=SQLiteArchive(":memory:"),
            eval_source=object(), policy=ImproverPolicy(),
        )


def test_online_improver_refuses_a_composite_constituent():
    agent, _ = _agent_with_metacognition()
    with pytest.raises(ValueError, match="constituent"):
        OnlineImprover(
            agent=agent,
            target_artifact_id="mon",  # a constituent of meta
            signal=object(), search=object(),
            archive=SQLiteArchive(":memory:"), policy=ImproverPolicy(),
        )


def test_offline_improver_may_target_the_composite_itself():
    """Targeting the composite (joint search) is allowed; only solo-targeting a
    constituent is refused. The composite is L3, so offline is the right mode."""
    agent, _ = _agent_with_metacognition()
    imp = OfflineImprover(
        agent=agent,
        target_artifact_id="meta",
        signal=object(), search=object(),
        archive=SQLiteArchive(":memory:"),
        eval_source=object(), policy=ImproverPolicy(),
    )
    assert imp.target_artifact_id == "meta"
