"""TaskSuccessSignal: deterministic ground-truth averaging over scenarios."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from helix.artifact import Subtype, genesis
from helix.signal import SignalKind
from helix.signals.task_success import TaskSuccessSignal


@dataclass
class _Sc:
    id: str
    val: float


def _desc(content: str = "desc"):
    return genesis("prompt.tool.x.description", Subtype.TOOL_DESCRIPTION, content)


@pytest.mark.asyncio
async def test_averages_per_scenario_scores_and_counts_full_passes():
    async def scorer(candidate, sc):
        return sc.val, {"booked": sc.id}

    sig = TaskSuccessSignal([_Sc("s1", 1.0), _Sc("s2", 0.0), _Sc("s3", 0.5)], scorer)
    m = await sig.measure(_desc())

    assert sig.kind == SignalKind.GROUND_TRUTH
    assert abs(m.score - 0.5) < 1e-9          # (1.0 + 0.0 + 0.5) / 3
    assert m.metadata["n_scenarios"] == 3
    assert m.metadata["n_full_pass"] == 1     # only s1 fully satisfied
    assert m.signal_id.startswith("TaskSuccessSignal:")


@pytest.mark.asyncio
async def test_caches_per_artifact_version():
    calls = {"n": 0}

    async def counting(candidate, sc):
        calls["n"] += 1
        return 1.0, {}

    sig = TaskSuccessSignal([_Sc("s1", 1.0)], counting)
    art = _desc("d1")
    await sig.measure(art)
    await sig.measure(art)
    assert calls["n"] == 1                     # second call served from cache

    await sig.measure(art.mutate("d2", created_by="spo"))
    assert calls["n"] == 2                      # new version re-runs


@pytest.mark.asyncio
async def test_failing_scorer_scores_zero_without_crashing_the_round():
    async def boom(candidate, sc):
        raise RuntimeError("agent run failed")

    sig = TaskSuccessSignal([_Sc("s1", 1.0)], boom)
    m = await sig.measure(_desc())
    assert m.score == 0.0
    assert "error" in m.metadata["per_scenario"][0]


@pytest.mark.asyncio
async def test_max_scenarios_caps_the_run():
    async def scorer(candidate, sc):
        return 1.0, {}

    sig = TaskSuccessSignal([_Sc(f"s{i}", 1.0) for i in range(10)], scorer, max_scenarios=3)
    m = await sig.measure(_desc())
    assert m.metadata["n_scenarios"] == 3
