"""SwapAndAgree position-bias wrapper behavior. SPEC §3.3, §8.3."""

from __future__ import annotations

import pytest

from helix.artifact import Subtype, genesis
from helix.signal import Cost, GapMeasurement, Preference, SignalKind
from helix.signals.pairwise_judge import SwapAndAgree


class _RecordingJudge:
    """Inner judge that records each call's (candidate id, trajectory)."""

    def __init__(self, pref: Preference = Preference.LEFT) -> None:
        self.calls: list[tuple[str, object]] = []
        self._pref = pref
        self.signal_id = "rec"
        self.signal_version = 1

    @property
    def kind(self):
        return SignalKind.LLM_JUDGE_PAIRWISE

    @property
    def cost_estimate(self):
        return Cost()

    async def measure(self, candidate, trajectory=None, reference=None, ground_truth=None):
        self.calls.append((candidate.id, trajectory))
        return GapMeasurement(score=1.0, preference=self._pref, confidence=1.0, cost=Cost())


class _ContentJudge:
    """Position-immune: prefers whichever slot holds the 'cand' artifact."""

    signal_id = "content"
    signal_version = 1

    @property
    def kind(self):
        return SignalKind.LLM_JUDGE_PAIRWISE

    @property
    def cost_estimate(self):
        return Cost()

    async def measure(self, candidate, trajectory=None, reference=None, ground_truth=None):
        pref = Preference.LEFT if candidate.id == "cand" else Preference.RIGHT
        return GapMeasurement(score=1.0, preference=pref, confidence=1.0, cost=Cost())


def _arts():
    return genesis("cand", Subtype.PROMPT, "C"), genesis("ref", Subtype.PROMPT, "R")


@pytest.mark.asyncio
async def test_swap_and_agree_forwards_reference_trajectory_on_reverse_pass():
    """On the reverse pass the reference takes the candidate slot, so its
    trajectory must ride the positional arg, not None (fix #10)."""
    cand, ref = _arts()
    judge = _RecordingJudge()
    gt = {"candidate_trajectory": "TC", "reference_trajectory": "TR"}

    await SwapAndAgree(judge).measure(candidate=cand, trajectory="TC", reference=ref, ground_truth=gt)

    assert len(judge.calls) == 2
    assert judge.calls[0] == ("cand", "TC")   # forward: candidate + its trajectory
    assert judge.calls[1] == ("ref", "TR")    # reverse: reference + its trajectory


@pytest.mark.asyncio
async def test_swap_and_agree_confirms_a_position_immune_winner():
    cand, ref = _arts()
    verdict = await SwapAndAgree(_ContentJudge()).measure(
        candidate=cand, reference=ref, ground_truth={}
    )
    assert verdict.preference == Preference.LEFT  # both passes agree the candidate wins


@pytest.mark.asyncio
async def test_swap_and_agree_collapses_position_bias_to_tie():
    """A judge that always prefers the LEFT slot disagrees with itself once the
    sides are swapped, so the wrapper returns TIE."""
    cand, ref = _arts()
    verdict = await SwapAndAgree(_RecordingJudge(pref=Preference.LEFT)).measure(
        candidate=cand, reference=ref, ground_truth={}
    )
    assert verdict.preference == Preference.TIE
    assert verdict.confidence == 0.0
