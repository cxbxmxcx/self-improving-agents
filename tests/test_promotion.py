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
from helix.artifact import ArtifactKind, genesis, Subtype
from helix.hooks import HookRegistry
from helix.improvement import OfflineImprover, OnlineImprover, ImproverMode, ImproverPolicy
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
    a1 = genesis(id=artifact_id, kind=Subtype.PROMPT, content="v1", created_by="human")
    a2 = a1.mutate(new_content="v2", created_by="SPO")
    m = GapMeasurement(
        score=0.5, preference=Preference.LEFT, feedback="", confidence=0.5,
        rubric_id=None, cost=Cost(0, 0.0, 0.0), metadata={},
    )
    await arc.record(Variant(artifact=a1, parent=a1, search_method="human", metadata={}), m)
    await arc.record(Variant(artifact=a2, parent=a1, search_method="SPO", metadata={}), m)


def _genesis_l1() -> object:
    return genesis(id="p.l1", kind=Subtype.PROMPT, content="x", created_by="human")


def _genesis_l3() -> object:
    # A standalone planner is L1 now; the L3 risk lives on the metacognition
    # composite that binds a planner, a monitor, and memory/state (SPEC §18.2.1).
    return genesis(
        id="meta.l3", kind=ArtifactKind.COMPOSITE, content=[],
        subtype=Subtype.METACOGNITION, created_by="human",
    )


def _genesis_l4() -> object:
    return genesis(id="code.l4", kind=ArtifactKind.CODE, content="x", created_by="human")


class _Stub:
    """Stand-in for Signal / Search / EvalSource in safety-check tests."""


class _StubAgent:
    """Minimal Agent stand-in: has system_prompt, hooks, and with_artifacts."""

    def __init__(self, prompt) -> None:
        self.system_prompt = prompt
        self.hooks = HookRegistry()
        self.max_iterations = 10
        self.max_tool_calls = 20

    def with_artifacts(self, overrides):
        new_prompt = overrides.get(self.system_prompt.id, self.system_prompt)
        return _StubAgent(new_prompt)


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
    agent = _StubAgent(_genesis_l3())
    with pytest.raises(ValueError, match="L3"):
        OnlineImprover(
            agent=agent,
            target_artifact_id="meta.l3",
            signal=_Stub(), search=_Stub(), archive=arc,
            policy=ImproverPolicy(),
        )


def test_online_improver_refuses_l4_artifact():
    arc = SQLiteArchive(":memory:")
    agent = _StubAgent(_genesis_l4())
    with pytest.raises(ValueError, match="L4"):
        OnlineImprover(
            agent=agent,
            target_artifact_id="code.l4",
            signal=_Stub(), search=_Stub(), archive=arc,
            policy=ImproverPolicy(),
        )


def test_online_improver_refuses_bound_l4_artifact_that_is_not_the_system_prompt():
    """The layer check resolves any bound artifact, not just the system prompt,
    so an L4 artifact bound elsewhere can no longer slip past (fix #2, §17.3)."""
    arc = SQLiteArchive(":memory:")
    code = _genesis_l4()

    class _AgentWithBoundCode(_StubAgent):
        def find_artifact(self, artifact_id):
            for a in (self.system_prompt, code):
                if a.id == artifact_id:
                    return a
            return None

    agent = _AgentWithBoundCode(_genesis_l1())  # L1 system prompt
    with pytest.raises(ValueError, match="L4"):
        OnlineImprover(
            agent=agent,
            target_artifact_id="code.l4",  # bound, but not the system prompt
            signal=_Stub(), search=_Stub(), archive=arc,
            policy=ImproverPolicy(),
        )


@pytest.mark.asyncio
async def test_attach_improver_subscribes_online_to_session_end():
    """attach_improver is the wiring step: after attach, the agent's SESSION_END
    hook routes to the online improver (SPEC §17.1, fix #13)."""
    from helix.agent import Agent
    from helix.hooks import HookPoint
    from helix.signal import Cost, GapMeasurement, SignalKind
    from helix.trajectory import Outcome, Trajectory

    class _ScoreSignal:
        signal_id = "s"
        signal_version = 1

        @property
        def kind(self):
            return SignalKind.LLM_JUDGE_ABSOLUTE

        @property
        def cost_estimate(self):
            return Cost()

        async def measure(self, candidate, trajectory=None, reference=None, ground_truth=None):
            return GapMeasurement(score=0.9, cost=Cost())

    prompt = genesis("p.sys", Subtype.PROMPT, "you are helpful")
    agent = Agent(system_prompt=prompt, model="claude-haiku-4-5")
    online = OnlineImprover(
        agent=agent, target_artifact_id="p.sys",
        signal=_ScoreSignal(), search=_Stub(),
        archive=SQLiteArchive(":memory:"), policy=ImproverPolicy(),
    )

    agent.attach_improver(online)
    assert online._subscribed  # wired at attach, before start()

    await online.start()
    traj = Trajectory(task="q")
    traj.complete("a", Outcome.COMPLETED)
    await agent.hooks.fire(HookPoint.SESSION_END, trajectory=traj)

    assert online.status.trajectories_seen == 1
    assert online.status.spot_checks_done == 1


def test_online_improver_allows_l1_artifact():
    arc = SQLiteArchive(":memory:")
    agent = _StubAgent(_genesis_l1())
    imp = OnlineImprover(
        agent=agent,
        target_artifact_id="p.l1",
        signal=_Stub(), search=_Stub(), archive=arc,
        policy=ImproverPolicy(),
    )
    assert imp.target_artifact_id == "p.l1"


def test_offline_improver_allows_l4_artifact():
    """Offline class imposes no layer restriction — the deploy gate handles it."""
    arc = SQLiteArchive(":memory:")
    agent = _StubAgent(_genesis_l4())
    imp = OfflineImprover(
        agent=agent,
        target_artifact_id="code.l4",
        signal=_Stub(), search=_Stub(), archive=arc,
        eval_source=_Stub(),
        policy=ImproverPolicy(),
        seed_fallback=_genesis_l4(),
    )
    assert imp.target_artifact_id == "code.l4"


def test_default_policy_is_offline_with_auto_promote_true():
    p = ImproverPolicy()
    assert p.mode == ImproverMode.OFFLINE
    assert p.auto_promote is True


# ---------------------------------------------------------------------------
# Online promotion gate: candidate must clear promote_threshold_win_rate
# on a per-request basis, not merely beat the reference on average.
# ---------------------------------------------------------------------------

class _RunAgent:
    """Agent stand-in with a run() the online shadow path can call."""

    def __init__(self, prompt) -> None:
        self.system_prompt = prompt
        self.hooks = HookRegistry()
        self.max_iterations = 10
        self.max_tool_calls = 20

    def with_artifacts(self, overrides):
        return _RunAgent(overrides.get(self.system_prompt.id, self.system_prompt))

    async def run(self, task):
        from helix.trajectory import Outcome, Trajectory
        t = Trajectory(task=task)
        t.complete(f"ans:{task[:20]}", Outcome.COMPLETED)
        return t.final_output, t


class _VersionSignal:
    """Scores the reference (v1) at 0.5 and the candidate (v>=2) from a fixed
    list, consumed one per shadow request. Lets a test dial the candidate's
    per-request win rate precisely."""

    def __init__(self, cand_scores) -> None:
        self._cand = list(cand_scores)
        self._i = 0
        self.signal_id = "stub-online"
        self.signal_version = 1

    @property
    def kind(self):
        from helix.signal import SignalKind
        return SignalKind.LLM_JUDGE_ABSOLUTE

    @property
    def cost_estimate(self):
        return Cost()

    async def measure(self, candidate, trajectory=None, reference=None, ground_truth=None):
        if getattr(candidate, "version", 1) >= 2:
            score = self._cand[self._i % len(self._cand)]
            self._i += 1
        else:
            score = 0.5
        return GapMeasurement(
            score=score, signal_id=self.signal_id,
            signal_version=self.signal_version, cost=Cost(),
        )


class _OneShotSearch:
    """Proposes a single mutated candidate (v2)."""

    @property
    def kind(self):
        from helix.search.base import SearchKind
        return SearchKind.PAIRWISE

    @property
    def cost_model(self):
        from helix.search.base import SearchCostModel
        return SearchCostModel()

    async def propose(self, seed, signal, archive, budget):
        yield Variant(
            artifact=seed.mutate("mutated", created_by="stub"),
            parent=seed, search_method="stub",
        )

    async def select(self, candidates, signal, archive):
        return candidates[0].artifact


async def _drive_online(improver, n_requests: int) -> None:
    """Feed n completed trajectories through the SESSION_END handler."""
    from helix.trajectory import Trajectory
    for i in range(n_requests):
        await improver._on_session_end(Trajectory(task=f"q{i}"))


def _make_online(cand_scores, *, promote_threshold_win_rate: float):
    bus = EventBus()
    archive = SQLiteArchive(":memory:")
    seed = genesis("prompt.online", Subtype.PROMPT, "seed", created_by="human")
    agent = _RunAgent(seed)
    imp = OnlineImprover(
        agent=agent,
        target_artifact_id="prompt.online",
        signal=_VersionSignal(cand_scores),
        search=_OneShotSearch(),
        archive=archive,
        # rolling_window 4 + shadow_sample 3 -> 7 requests drives one full cycle.
        policy=ImproverPolicy(promote_threshold_win_rate=promote_threshold_win_rate),
        bus=bus,
    )
    return imp, archive


@pytest.mark.asyncio
async def test_online_below_win_rate_threshold_does_not_promote():
    """Candidate beats the reference on mean shadow score (0.566 vs 0.5) but
    wins only 1 of 3 requests; a 0.5 win-rate bar must block auto-promotion."""
    imp, archive = _make_online([0.9, 0.4, 0.4], promote_threshold_win_rate=0.5)
    await imp.start()
    await _drive_online(imp, 7)  # 4 spot-checks trigger a proposal, 3 shadow runs
    assert imp._last_error is None
    assert await archive.live_champion("prompt.online") is None
    assert imp.status.candidates_promoted == 0


@pytest.mark.asyncio
async def test_online_above_win_rate_threshold_auto_promotes():
    """Candidate wins all 3 shadow requests, clearing the 0.5 bar, so the
    default handler auto-promotes it to live champion."""
    imp, archive = _make_online([0.9, 0.9, 0.9], promote_threshold_win_rate=0.5)
    await imp.start()
    await _drive_online(imp, 7)
    assert imp._last_error is None
    live = await archive.live_champion("prompt.online")
    assert live is not None and live.version == 2
    assert imp.status.candidates_promoted == 1


# ---------------------------------------------------------------------------
# Artifact layer mapping
# ---------------------------------------------------------------------------

def test_artifact_layer_mapping():
    """The L1-L4 mapping that drives the online safety check (SPEC §1.2, §18.2)."""
    from helix.artifact import layer_of

    # All text is L1 in v0.2, including planner/monitor standalone.
    assert layer_of(ArtifactKind.TEXT, Subtype.PROMPT) == 1
    assert layer_of(ArtifactKind.TEXT, Subtype.SKILL) == 1
    assert layer_of(ArtifactKind.TEXT, Subtype.TOOL_DESCRIPTION) == 1
    assert layer_of(ArtifactKind.TEXT, Subtype.RUBRIC) == 1
    assert layer_of(ArtifactKind.TEXT, Subtype.PLANNER) == 1
    assert layer_of(ArtifactKind.TEXT, Subtype.MONITOR) == 1
    assert layer_of(ArtifactKind.MEMORY_ENTRY) == 2
    assert layer_of(ArtifactKind.CODE) == 4
    # L3 is the metacognition composite: its subtype floor lifts it above its
    # constituents (an L1 planner, an L1 monitor, an L2 memory).
    assert layer_of(ArtifactKind.COMPOSITE, Subtype.METACOGNITION) == 3
    assert layer_of(ArtifactKind.COMPOSITE, Subtype.METACOGNITION, (1, 1, 2)) == 3
    # The constituent term keeps a code-bearing bundle at L4 even with no floor.
    assert layer_of(ArtifactKind.COMPOSITE, None, (1, 4)) == 4
