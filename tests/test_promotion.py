"""Tests for the promotion model.

Promotion separates two concepts that used to be one:
  - 'best candidate' (archive.best): highest-scoring artifact, updated
    every round.
  - 'live champion' (archive.live_champion): what the running agent
    serves, updated only when archive.promote() is called.

Online improvers auto-promote via a default hook handler. Offline
improvers leave promotion to humans. See DESIGN_NOTES.md section 10.
"""

from __future__ import annotations

import asyncio

import pytest

from helix.archive import SQLiteArchive, PromotionRecord
from helix.artifact import ArtifactKind, genesis
from helix.improvement import Improver, ImproverMode, ImproverPolicy
from helix.improvement.promotion import (
    default_promotion_handler,
    ensure_default_handler_registered,
    register_improver_archive,
    unregister_improver_archive,
)
from helix.observability.bus import EventBus
from helix.observability.events import CandidateWins, Promoted
from helix.search.base import Variant
from helix.signal import Cost, GapMeasurement, Preference


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _seed_two_versions(arc: SQLiteArchive, artifact_id: str = "p.test") -> None:
    """Record v1 and v2 of an artifact in the archive."""
    a1 = genesis(id=artifact_id, kind=ArtifactKind.PROMPT, content="v1", created_by="human")
    a2 = a1.mutate(new_content="v2", created_by="SPO")
    m = GapMeasurement(
        score=0.5, preference=Preference.LEFT, feedback="", confidence=0.5,
        rubric_id=None, cost=Cost(0, 0.0, 0.0), metadata={},
    )
    await arc.record(Variant(artifact=a1, parent=a1, search_method="human", metadata={}), m)
    await arc.record(Variant(artifact=a2, parent=a1, search_method="SPO", metadata={}), m)


def _genesis_l1() -> object:
    return genesis(id="p.l1", kind=ArtifactKind.PROMPT, content="x", created_by="human")


def _genesis_l3() -> object:
    return genesis(id="plan.l3", kind=ArtifactKind.PLANNER, content="x", created_by="human")


def _genesis_l4() -> object:
    return genesis(id="code.l4", kind=ArtifactKind.CODE, content="x", created_by="human")


class _Stub:
    """Stand-in for Signal / Search / EvalSource in safety-check tests."""


async def _build_agent(*_args, **_kwargs):
    return None


# ---------------------------------------------------------------------------
# Archive primitives: promote, live_champion, promotion_history
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_live_champion_is_none_before_any_promotion():
    arc = SQLiteArchive(":memory:")
    await _seed_two_versions(arc)
    assert await arc.live_champion("p.test") is None


@pytest.mark.asyncio
async def test_promote_records_approver_reason_and_timestamp():
    arc = SQLiteArchive(":memory:")
    await _seed_two_versions(arc)

    rec = await arc.promote("p.test", 1, approver="alice", reason="initial deploy")

    assert isinstance(rec, PromotionRecord)
    assert rec.artifact_id == "p.test"
    assert rec.version == 1
    assert rec.approver == "alice"
    assert rec.reason == "initial deploy"
    assert rec.promoted_at  # non-empty ISO timestamp


@pytest.mark.asyncio
async def test_live_champion_returns_most_recent_promotion():
    arc = SQLiteArchive(":memory:")
    await _seed_two_versions(arc)

    await arc.promote("p.test", 1, approver="alice", reason="initial")
    live = await arc.live_champion("p.test")
    assert live is not None and live.version == 1

    await arc.promote("p.test", 2, approver="bob", reason="upgrade")
    live = await arc.live_champion("p.test")
    assert live.version == 2


@pytest.mark.asyncio
async def test_rollback_is_just_another_promote_of_earlier_version():
    arc = SQLiteArchive(":memory:")
    await _seed_two_versions(arc)

    await arc.promote("p.test", 1, approver="alice", reason="initial")
    await arc.promote("p.test", 2, approver="bob", reason="upgrade")
    await arc.promote("p.test", 1, approver="carol", reason="rollback after regression")

    live = await arc.live_champion("p.test")
    assert live.version == 1


@pytest.mark.asyncio
async def test_promotion_history_is_oldest_first():
    arc = SQLiteArchive(":memory:")
    await _seed_two_versions(arc)

    await arc.promote("p.test", 1, approver="alice", reason="r1")
    await arc.promote("p.test", 2, approver="bob", reason="r2")
    await arc.promote("p.test", 1, approver="carol", reason="r3")

    history = await arc.promotion_history("p.test")
    assert [h.version for h in history] == [1, 2, 1]
    assert [h.approver for h in history] == ["alice", "bob", "carol"]


@pytest.mark.asyncio
async def test_promote_raises_on_nonexistent_version():
    arc = SQLiteArchive(":memory:")
    await _seed_two_versions(arc)

    with pytest.raises(ValueError, match="no such artifact"):
        await arc.promote("p.test", 99, approver="x", reason="y")


@pytest.mark.asyncio
async def test_promote_publishes_promoted_event():
    bus = EventBus()
    arc = SQLiteArchive(":memory:", bus=bus)
    await _seed_two_versions(arc)

    captured: list[Promoted] = []

    async def collect(ev):
        captured.append(ev)

    bus.subscribe("promoted", collect)

    await arc.promote("p.test", 2, approver="improver:test-1", reason="auto")

    assert len(captured) == 1
    ev = captured[0]
    assert ev.artifact_id == "p.test"
    assert ev.version == 2
    assert ev.approver == "improver:test-1"
    assert ev.reason == "auto"


# ---------------------------------------------------------------------------
# Default promotion hook: offline vs online behavior
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_offline_candidate_wins_does_not_promote():
    bus = EventBus()
    arc = SQLiteArchive(":memory:", bus=bus)
    await _seed_two_versions(arc)

    register_improver_archive("test-offline", arc)
    ensure_default_handler_registered(bus)

    promoted_events: list[Promoted] = []
    bus.subscribe("promoted", lambda ev: promoted_events.append(ev))

    await bus.publish(CandidateWins(
        improver_id="test-offline",
        target_artifact_id="p.test",
        candidate_version=2, reference_version=1,
        candidate_score=0.9, reference_score=0.4,
        mode="offline", auto_promote=True,
    ))

    assert await arc.live_champion("p.test") is None
    assert promoted_events == []

    unregister_improver_archive("test-offline")


@pytest.mark.asyncio
async def test_online_candidate_wins_auto_promotes():
    bus = EventBus()
    arc = SQLiteArchive(":memory:", bus=bus)
    await _seed_two_versions(arc)

    register_improver_archive("test-online", arc)
    ensure_default_handler_registered(bus)

    await bus.publish(CandidateWins(
        improver_id="test-online",
        target_artifact_id="p.test",
        candidate_version=2, reference_version=1,
        candidate_score=0.9, reference_score=0.4,
        mode="online", auto_promote=True,
    ))

    live = await arc.live_champion("p.test")
    assert live is not None and live.version == 2

    unregister_improver_archive("test-online")


@pytest.mark.asyncio
async def test_online_with_auto_promote_disabled_does_not_promote():
    """auto_promote=False is the escape hatch for custom promotion policies."""
    bus = EventBus()
    arc = SQLiteArchive(":memory:", bus=bus)
    await _seed_two_versions(arc)

    register_improver_archive("test-online-off", arc)
    ensure_default_handler_registered(bus)

    await bus.publish(CandidateWins(
        improver_id="test-online-off",
        target_artifact_id="p.test",
        candidate_version=2, reference_version=1,
        candidate_score=0.9, reference_score=0.4,
        mode="online", auto_promote=False,
    ))

    assert await arc.live_champion("p.test") is None
    unregister_improver_archive("test-online-off")


@pytest.mark.asyncio
async def test_auto_promote_records_improver_id_as_approver():
    bus = EventBus()
    arc = SQLiteArchive(":memory:", bus=bus)
    await _seed_two_versions(arc)

    register_improver_archive("imp-xyz", arc)
    ensure_default_handler_registered(bus)

    await bus.publish(CandidateWins(
        improver_id="imp-xyz",
        target_artifact_id="p.test",
        candidate_version=2, reference_version=1,
        candidate_score=0.9, reference_score=0.4,
        mode="online", auto_promote=True,
    ))

    history = await arc.promotion_history("p.test")
    assert len(history) == 1
    assert history[0].approver == "improver:imp-xyz"
    assert "auto-promoted by imp-xyz" in history[0].reason
    assert "v2" in history[0].reason and "v1" in history[0].reason

    unregister_improver_archive("imp-xyz")


# ---------------------------------------------------------------------------
# Online safety check: refuse L3/L4 artifacts at construction
# ---------------------------------------------------------------------------

def test_online_improver_refuses_l3_artifact():
    arc = SQLiteArchive(":memory:")
    with pytest.raises(ValueError, match="L3"):
        Improver(
            target_artifact_id="plan.l3",
            signal=_Stub(), search=_Stub(), archive=arc,
            eval_source=_Stub(), build_agent_with_prompt=_build_agent,
            policy=ImproverPolicy(mode=ImproverMode.ONLINE),
            seed_fallback=_genesis_l3(),
        )


def test_online_improver_refuses_l4_artifact():
    arc = SQLiteArchive(":memory:")
    with pytest.raises(ValueError, match="L4"):
        Improver(
            target_artifact_id="code.l4",
            signal=_Stub(), search=_Stub(), archive=arc,
            eval_source=_Stub(), build_agent_with_prompt=_build_agent,
            policy=ImproverPolicy(mode=ImproverMode.ONLINE),
            seed_fallback=_genesis_l4(),
        )


def test_online_improver_allows_l1_artifact():
    arc = SQLiteArchive(":memory:")
    imp = Improver(
        target_artifact_id="p.l1",
        signal=_Stub(), search=_Stub(), archive=arc,
        eval_source=_Stub(), build_agent_with_prompt=_build_agent,
        policy=ImproverPolicy(mode=ImproverMode.ONLINE),
        seed_fallback=_genesis_l1(),
    )
    assert imp.policy.mode == ImproverMode.ONLINE


def test_offline_improver_allows_l4_artifact():
    """Offline mode imposes no layer restriction — the deploy gate handles it."""
    arc = SQLiteArchive(":memory:")
    imp = Improver(
        target_artifact_id="code.l4",
        signal=_Stub(), search=_Stub(), archive=arc,
        eval_source=_Stub(), build_agent_with_prompt=_build_agent,
        policy=ImproverPolicy(mode=ImproverMode.OFFLINE),
        seed_fallback=_genesis_l4(),
    )
    assert imp.policy.mode == ImproverMode.OFFLINE


def test_default_policy_is_offline_with_auto_promote_true():
    p = ImproverPolicy()
    assert p.mode == ImproverMode.OFFLINE
    assert p.auto_promote is True


# ---------------------------------------------------------------------------
# Artifact layer mapping
# ---------------------------------------------------------------------------

def test_artifact_kind_layer_mapping():
    """The L1-L4 mapping that drives the online safety check."""
    assert ArtifactKind.PROMPT.layer == 1
    assert ArtifactKind.SKILL.layer == 1
    assert ArtifactKind.TOOL_DESCRIPTION.layer == 1
    assert ArtifactKind.RUBRIC.layer == 1
    assert ArtifactKind.MEMORY_ENTRY.layer == 2
    assert ArtifactKind.PLANNER.layer == 3
    assert ArtifactKind.MONITOR.layer == 3
    assert ArtifactKind.CODE.layer == 4
