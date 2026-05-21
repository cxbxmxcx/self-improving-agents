"""GoldenSetCalibrator tests.

Use a stub agent factory + stub score_fn to test cache behavior, scoring,
and protocol conformance without LLM calls.
"""

from __future__ import annotations

import pytest

from helix.artifact import ArtifactKind, genesis
from helix.eval.dataset import EvalQuestion, EvalSet
from helix.signal import Preference, Signal
from helix.signals.golden_set import GoldenSetCalibrator
from helix.trajectory import Trajectory


def _make_golden() -> EvalSet:
    return EvalSet(questions=[
        EvalQuestion(id="G1", band=1, question="capital of france?", reference_answer="paris"),
        EvalQuestion(id="G2", band=1, question="capital of germany?", reference_answer="berlin"),
    ])


class _StubAgent:
    def __init__(self, prompt, answers: dict[str, str]):
        self.prompt = prompt
        self._answers = answers

    async def run(self, task: str):
        for key, value in self._answers.items():
            if key in task:
                t = Trajectory(task=task)
                t.complete(value)
                return value, t
        t = Trajectory(task=task)
        t.complete("don't know")
        return "don't know", t


def _make_factory(answers: dict[str, str]):
    async def factory(art):
        return _StubAgent(art, answers)
    return factory


@pytest.mark.asyncio
async def test_calibrator_requires_nonempty_golden_set():
    with pytest.raises(ValueError):
        GoldenSetCalibrator(
            golden_set=EvalSet(questions=[]),
            agent_factory=_make_factory({}),
        )


@pytest.mark.asyncio
async def test_perfect_answers_score_one():
    cal = GoldenSetCalibrator(
        golden_set=_make_golden(),
        agent_factory=_make_factory({"france": "paris is the capital", "germany": "berlin is the capital"}),
    )
    art = genesis("p", ArtifactKind.PROMPT, "v1")
    m = await cal.measure(candidate=art)
    assert m.score == 1.0
    assert m.metadata["n_correct"] == 2
    assert m.preference == Preference.NONE


@pytest.mark.asyncio
async def test_failed_answers_score_zero():
    cal = GoldenSetCalibrator(
        golden_set=_make_golden(),
        agent_factory=_make_factory({"france": "berlin", "germany": "paris"}),
    )
    art = genesis("p", ArtifactKind.PROMPT, "v1")
    m = await cal.measure(candidate=art)
    assert m.score == 0.0


@pytest.mark.asyncio
async def test_partial_correctness():
    cal = GoldenSetCalibrator(
        golden_set=_make_golden(),
        agent_factory=_make_factory({"france": "paris is the capital"}),
    )
    art = genesis("p", ArtifactKind.PROMPT, "v1")
    m = await cal.measure(candidate=art)
    assert m.score == 0.5
    assert m.metadata["n_correct"] == 1
    assert m.metadata["n_questions"] == 2


@pytest.mark.asyncio
async def test_cache_returns_quickly_on_second_call():
    cal = GoldenSetCalibrator(
        golden_set=_make_golden(),
        agent_factory=_make_factory({"france": "paris", "germany": "berlin"}),
        cache_ttl_sec=3600,
    )
    art = genesis("p", ArtifactKind.PROMPT, "v1")
    m1 = await cal.measure(candidate=art)
    m2 = await cal.measure(candidate=art)
    assert m1.score == m2.score
    # Second call hits the cache
    assert m2.metadata["cached"] is True
    assert m2.cost.tokens == 0


@pytest.mark.asyncio
async def test_cache_busts_on_new_artifact_version():
    cal = GoldenSetCalibrator(
        golden_set=_make_golden(),
        agent_factory=_make_factory({"france": "paris", "germany": "berlin"}),
        cache_ttl_sec=3600,
    )
    v1 = genesis("p", ArtifactKind.PROMPT, "v1")
    v2 = v1.mutate("v2", created_by="test")
    m1 = await cal.measure(candidate=v1)
    m2 = await cal.measure(candidate=v2)
    assert m2.metadata["cached"] is False


def test_calibrator_satisfies_signal_protocol():
    cal = GoldenSetCalibrator(
        golden_set=_make_golden(),
        agent_factory=_make_factory({}),
    )
    assert isinstance(cal, Signal)


@pytest.mark.asyncio
async def test_clear_cache_drops_entries():
    cal = GoldenSetCalibrator(
        golden_set=_make_golden(),
        agent_factory=_make_factory({"france": "paris", "germany": "berlin"}),
    )
    art = genesis("p", ArtifactKind.PROMPT, "v1")
    await cal.measure(candidate=art)
    assert cal.status()["cache_entries"] == 1
    cal.clear_cache()
    assert cal.status()["cache_entries"] == 0
