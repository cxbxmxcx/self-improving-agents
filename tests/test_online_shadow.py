"""Shadow runs are measured, never seen: they run under the live request's
memory context, never write memory, and can be built over a forked
environment. SPEC §17.2."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

from helix.agent import Agent
from helix.archive import SQLiteArchive
from helix.artifact import Subtype, genesis
from helix.improvement import ImproverPolicy, OnlineImprover
from helix.memory.base import MemoryContext
from helix.search.base import Variant
from helix.signal import GapMeasurement, SignalKind
from helix.trajectory import Trajectory


class _ScoreSignal:
    id = "score"
    version = 1

    @property
    def kind(self) -> SignalKind:
        return SignalKind.LLM_JUDGE

    async def measure(self, candidate=None, trajectory=None, **_):
        return GapMeasurement(score=0.5)


class _Stub:
    pass


def _fake_clone():
    run = AsyncMock(return_value=("answer", Trajectory(task="t")))
    return SimpleNamespace(run=run, memory_hooks=SimpleNamespace(read_only=False))


def _improver(agent, **kwargs) -> OnlineImprover:
    online = OnlineImprover(agent=agent, target_artifact_id="p.sys", signal=_ScoreSignal(), search=_Stub(),
                            archive=SQLiteArchive(":memory:"), policy=ImproverPolicy(), **kwargs)
    candidate = genesis("p.sys", Subtype.PROMPT, "candidate prompt")
    online._candidate_in_flight = SimpleNamespace(artifact=candidate)
    online._reference = agent.system_prompt
    online._shadow_requests_remaining = 2  # stay in shadow mode after one request
    return online


def _live_trajectory() -> Trajectory:
    trajectory = Trajectory(task="Book a writing block.")
    trajectory.metadata["memory_context"] = {"session_id": "maya-monday", "user_id": "maya", "org_id": None}
    return trajectory


async def test_shadow_run_uses_the_factory_clone_with_read_only_memory_and_the_live_context():
    agent = Agent(system_prompt=genesis("p.sys", Subtype.PROMPT, "live prompt"), model="m")
    clone = _fake_clone()
    calls = []

    def factory(base, overrides):
        calls.append((base, set(overrides)))
        return clone

    online = _improver(agent, shadow_factory=factory)
    await online._handle_shadow_request(_live_trajectory())
    assert calls == [(agent, {"p.sys"})]
    assert clone.memory_hooks.read_only is True
    task, context = clone.run.await_args.args
    assert task == "Book a writing block."
    assert context == MemoryContext(session_id="maya-monday", user_id="maya", org_id=None)


async def test_shadow_run_without_a_factory_clones_the_agent_and_still_never_writes_memory():
    agent = Agent(system_prompt=genesis("p.sys", Subtype.PROMPT, "live prompt"), model="m")
    clone = _fake_clone()
    agent.with_artifacts = lambda overrides: clone  # the default path
    online = _improver(agent)
    await online._handle_shadow_request(_live_trajectory())
    assert clone.memory_hooks.read_only is True
    assert clone.run.await_args.args[1].user_id == "maya"


async def test_a_winning_candidate_that_repeats_earlier_content_promotes_that_version():
    agent = Agent(system_prompt=genesis("p.sys", Subtype.PROMPT, "live prompt"), model="m")
    online = _improver(agent)
    archive = online.archive
    await archive.put_artifact(agent.system_prompt)
    await archive.promote("p.sys", 1, approver="maya", reason="seed")
    earlier = agent.system_prompt.mutate("candidate prompt", created_by="search")  # v2, same content
    await archive.put_artifact(earlier)
    repeat = agent.system_prompt.mutate("candidate prompt", created_by="search", version=3)
    online._candidate_in_flight = Variant(artifact=repeat, parent=agent.system_prompt, search_method="search")
    online._shadow_ref_scores, online._shadow_cand_scores = [0.2], [0.9]
    online._shadow_pairs = [(0.2, 0.9)]
    online.signal = SimpleNamespace(signal_id="score", signal_version=1)
    await online._finalize_shadow()
    assert (await archive.live_champion("p.sys")).version == 2
