"""LLM-as-Judge pairwise signal.

A model picks between two candidates given the same task. The judge sees the
task context, the two answers, and the rubric, and returns LEFT/RIGHT/TIE plus
optional differential feedback.

Position bias is the canonical problem with pairwise judging: judges prefer
whichever answer is shown first. The SwapAndAgree wrapper runs the judge twice
with positions swapped and requires both runs to agree; on disagreement, the
verdict is TIE. SPEC section 8.3.

This implementation expects candidate and reference artifacts to be prompts
(text content), and uses the trajectory's final_output as the answer to judge.
Callers that want to judge other artifact kinds wire their own answer-extractor
function via the `extract_answer` parameter.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field

from helix._caching import cacheable_system_message
from helix.llm_call import acompletion as helix_acompletion
from helix.artifact import Artifact
from helix.signal import (
    Cost,
    GapMeasurement,
    Preference,
    Signal,
    SignalKind,
    derive_signal_id,
)
from helix.trajectory import Trajectory


class _PairwiseVerdict(BaseModel):
    """Structured judge output."""
    winner: str = Field(
        description="Which candidate wins. Must be exactly 'LEFT', 'RIGHT', or 'TIE'.",
    )
    feedback: str = Field(
        description="A short explanation, including the axis along which the winner is better and where the loser fails.",
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="How confident you are in this verdict, from 0 (coin flip) to 1 (certain).",
    )


DEFAULT_RUBRIC = """You are judging which of two candidate prompts produced the better answer to a research question.

Score on these axes in this order of importance:
1. Factual accuracy against the reference answer.
2. Whether the candidate correctly retrieved from the corpus or correctly recognized that the answer is not in the corpus.
3. Clarity and directness.
4. Citation of source documents when appropriate.

If both answers are equally good (or equally bad), reply TIE."""


AnswerExtractor = Callable[[Trajectory | None], str]


def _default_extract_answer(trajectory: Trajectory | None) -> str:
    if trajectory is None or trajectory.final_output is None:
        return "(no answer produced)"
    return str(trajectory.final_output)


class PairwiseJudge:
    """LLM-as-Judge pairwise Signal implementation.

    Usage:
        judge = PairwiseJudge(model="gpt-4o", rubric=DEFAULT_RUBRIC)
        m = await judge.measure(
            candidate=candidate_artifact,
            reference=current_best_artifact,
            trajectory=candidate_trajectory,
            ground_truth={"question": q, "reference_answer": ref, "candidate_trajectory": ct, "reference_trajectory": rt},
        )
        m.preference  # Preference.LEFT means the candidate beat the reference

    The `ground_truth` dict is the conventional way to pass auxiliary context
    (the question, the reference answer, and both trajectories) without
    extending the Signal protocol. Callers can pass anything they want here;
    this judge looks for "question", "reference_answer", and the two
    trajectories under "candidate_trajectory" and "reference_trajectory".
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        rubric: str = DEFAULT_RUBRIC,
        rubric_id: tuple[str, int] | None = None,
        extract_answer: AnswerExtractor = _default_extract_answer,
        version: int = 1,
    ) -> None:
        self.model = model
        self.rubric = rubric
        self._rubric_id = rubric_id
        self.extract_answer = extract_answer
        self._version = version

    @property
    def kind(self) -> SignalKind:
        return SignalKind.LLM_JUDGE_PAIRWISE

    @property
    def cost_estimate(self) -> Cost:
        # Rough estimate: one model call, ~1k tokens.
        return Cost(tokens=1000, dollars=0.005, wall_clock_sec=3.0)

    @property
    def signal_id(self) -> str:
        return derive_signal_id(
            "PairwiseJudge",
            {"model": self.model, "rubric": self.rubric, "rubric_id": self._rubric_id},
        )

    @property
    def signal_version(self) -> int:
        return self._version

    async def measure(
        self,
        candidate: Artifact,
        trajectory: Trajectory | None = None,
        reference: Artifact | None = None,
        ground_truth: Any | None = None,
    ) -> GapMeasurement:
        if reference is None:
            raise ValueError("PairwiseJudge requires a reference artifact to compare against")

        ctx = ground_truth or {}
        question = ctx.get("question", "(no question provided)")
        reference_answer = ctx.get("reference_answer", "(no reference answer provided)")
        cand_traj = ctx.get("candidate_trajectory", trajectory)
        ref_traj = ctx.get("reference_trajectory")

        cand_answer = self.extract_answer(cand_traj)
        ref_answer = self.extract_answer(ref_traj)

        verdict, cost = await self._judge_once(
            question=question,
            reference_answer=reference_answer,
            left=cand_answer,
            right=ref_answer,
        )

        preference = self._verdict_to_preference(verdict.winner)
        score = 1.0 if preference == Preference.LEFT else (0.0 if preference == Preference.RIGHT else 0.5)

        return GapMeasurement(
            score=score,
            preference=preference,
            feedback=verdict.feedback,
            confidence=verdict.confidence,
            rubric_id=self._rubric_id,
            signal_id=self.signal_id,
            signal_version=self.signal_version,
            cost=cost,
            metadata={
                "judge_model": self.model,
                "candidate_answer_excerpt": cand_answer[:300],
                "reference_answer_excerpt": ref_answer[:300],
            },
        )

    async def _judge_once(
        self,
        question: str,
        reference_answer: str,
        left: str,
        right: str,
    ) -> tuple[_PairwiseVerdict, Cost]:
        prompt = self._render_prompt(question, reference_answer, left, right)
        # The rubric is identical across the 20 judge calls in one round, so
        # marking it cacheable saves 70-90% on input tokens for Anthropic.
        response = await helix_acompletion(
            model=self.model,
            messages=[
                cacheable_system_message(self.rubric, self.model),
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content or "{}"
        try:
            data = json.loads(raw)
            verdict = _PairwiseVerdict(**data)
        except Exception:
            verdict = _PairwiseVerdict(winner="TIE", feedback="judge output malformed", confidence=0.0)

        usage = getattr(response, "usage", None)
        cost = Cost(
            tokens=getattr(usage, "total_tokens", 0) if usage else 0,
            dollars=0.0,  # callers can layer their own cost model
            wall_clock_sec=0.0,
        )
        return verdict, cost

    def _render_prompt(
        self,
        question: str,
        reference_answer: str,
        left: str,
        right: str,
    ) -> str:
        return f"""Question:
{question}

Reference answer (the ground truth target):
{reference_answer}

Candidate LEFT:
{left}

Candidate RIGHT:
{right}

Reply as JSON with three fields:
  "winner": "LEFT" or "RIGHT" or "TIE"
  "feedback": a short string explaining the verdict
  "confidence": a float between 0 and 1"""

    @staticmethod
    def _verdict_to_preference(winner: str) -> Preference:
        w = winner.strip().upper()
        if w == "LEFT":
            return Preference.LEFT
        if w == "RIGHT":
            return Preference.RIGHT
        return Preference.TIE


# ---------------------------------------------------------------------------
# SwapAndAgree: position-bias mitigation wrapper. SPEC section 8.3.
# ---------------------------------------------------------------------------

class SwapAndAgree:
    """Wrap a pairwise Signal; run twice with positions swapped, require agreement.

    Position bias is the canonical pairwise-judge failure mode (judges prefer
    whichever answer is shown first). Running the same judge with the candidate
    and reference swapped, and requiring both runs to agree, suppresses it.
    On disagreement we return TIE.
    """

    def __init__(self, inner: Signal) -> None:
        self.inner = inner

    @property
    def kind(self) -> SignalKind:
        return self.inner.kind

    @property
    def cost_estimate(self) -> Cost:
        # Two calls instead of one.
        inner_cost = self.inner.cost_estimate
        return Cost(
            tokens=inner_cost.tokens * 2,
            dollars=inner_cost.dollars * 2,
            wall_clock_sec=inner_cost.wall_clock_sec * 2,
        )

    @property
    def signal_id(self) -> str:
        return derive_signal_id(
            "SwapAndAgree",
            {"inner": getattr(self.inner, "signal_id", type(self.inner).__name__)},
        )

    @property
    def signal_version(self) -> int:
        return getattr(self.inner, "signal_version", 1)

    async def measure(
        self,
        candidate: Artifact,
        trajectory: Trajectory | None = None,
        reference: Artifact | None = None,
        ground_truth: Any | None = None,
    ) -> GapMeasurement:
        forward = await self.inner.measure(candidate, trajectory, reference, ground_truth)

        swapped_ctx: dict[str, Any] = dict(ground_truth or {})
        swapped_ctx["candidate_trajectory"], swapped_ctx["reference_trajectory"] = (
            swapped_ctx.get("reference_trajectory"),
            swapped_ctx.get("candidate_trajectory", trajectory),
        )
        reversed_pref_map = {
            Preference.LEFT: Preference.RIGHT,
            Preference.RIGHT: Preference.LEFT,
            Preference.TIE: Preference.TIE,
            Preference.NONE: Preference.NONE,
        }
        reverse = await self.inner.measure(reference, None, candidate, swapped_ctx)
        # After swap, LEFT-pref means the original reference won; map back.
        reverse_pref = reversed_pref_map[reverse.preference]

        if forward.preference == reverse_pref and forward.preference != Preference.TIE:
            agreed = forward.preference
            confidence = min(forward.confidence, reverse.confidence)
        else:
            agreed = Preference.TIE
            confidence = 0.0

        score = 1.0 if agreed == Preference.LEFT else (0.0 if agreed == Preference.RIGHT else 0.5)

        return GapMeasurement(
            score=score,
            preference=agreed,
            feedback=(forward.feedback or "") + "\n---\n" + (reverse.feedback or ""),
            confidence=confidence,
            rubric_id=forward.rubric_id,
            signal_id=self.signal_id,
            signal_version=self.signal_version,
            cost=forward.cost + reverse.cost,
            metadata={
                "forward": forward.to_dict(),
                "reverse": reverse.to_dict(),
                "agreed": agreed.value,
            },
        )
