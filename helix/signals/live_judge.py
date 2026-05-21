"""LiveTrajectoryJudge: rubric-based absolute scoring on a single trajectory.

Unlike PairwiseJudge (which compares two candidates against each other) or
ground-truth signals (which check against a reference answer), this judge
operates on a trajectory in isolation. It reads:

  - The user's question
  - The retrieved passages the agent surfaced
  - The agent's final answer
  - The trajectory step pattern (retrieve called? abstain reached?)

...and rates it 0-1 against a rubric. The rubric is the artifact under
improvement at this layer; the book's Ch 8 thesis ("rubrics decay, calibrate
periodically") becomes concrete here.

Use this signal when you cannot get pairwise comparisons (live production
traffic with no second-prompt to compare against) and have no ground-truth
answer (the user's question is novel). Combine with GoldenSetCalibrator in
a CompositeSignal so the rubric stays anchored.

Failure mode: Huang ICLR 2024. Without an external oracle, the rubric drifts
and the scores compound their own drift. The mitigation is the calibration
signal, which periodically scores the champion against a golden set.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field

from helix._caching import cacheable_system_message
from helix.artifact import Artifact
from helix.llm_call import acompletion as helix_acompletion
from helix.signal import (
    Cost,
    GapMeasurement,
    Preference,
    SignalKind,
)
from helix.trajectory import StepKind, Trajectory


class _LiveVerdict(BaseModel):
    score: float = Field(ge=0.0, le=1.0, description="0=poor, 1=excellent")
    reasoning: str = Field(description="One paragraph explanation of the score.")
    confidence: float = Field(ge=0.0, le=1.0)
    flags: list[str] = Field(
        default_factory=list,
        description=(
            "Short tags for systemic issues: 'no_retrieval', 'hallucinated', "
            "'abstained_correctly', 'wrong_citation', 'verbose', 'concise', etc."
        ),
    )


DEFAULT_LIVE_RUBRIC = """You are an evaluator of a research assistant's response to a user question.

You will see:
- The user's question
- The retrieval calls the agent made (queries + retrieved sources)
- The agent's final answer

Rate the response from 0 to 1 along these dimensions, weighted in this order:

1. GROUNDING (most important): did the agent actually use retrieved passages
   to answer? Did it cite specific sources? Or did it answer from prior
   knowledge / hallucinate?

2. CORRECTNESS / FAITHFULNESS: does the answer reflect what the retrieved
   passages actually say? Be strict about misattribution.

3. ABSTENTION DISCIPLINE: if the corpus doesn't contain the answer, did the
   agent say so? Inventing an answer for a trap question scores low even if
   it sounds plausible.

4. CONCISION: a tight, direct answer scores higher than a verbose one of
   equal correctness.

Score the response 0-1 holistically. Reply as JSON:
  "score": float
  "reasoning": one paragraph
  "confidence": float
  "flags": list of short tags (e.g. ["no_retrieval", "hallucinated"]
                                or ["abstained_correctly", "concise"])"""


class LiveTrajectoryJudge:
    """Absolute LLM-as-Judge for a single live trajectory.

    Conforms to the Signal protocol via measure(). Fills GapMeasurement.score
    in [0,1], preference=NONE (this is absolute, not pairwise), feedback is
    the judge's reasoning + flags string.
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        rubric: str = DEFAULT_LIVE_RUBRIC,
        rubric_id: tuple[str, int] | None = None,
    ) -> None:
        self.model = model
        self.rubric = rubric
        self._rubric_id = rubric_id

    @property
    def kind(self) -> SignalKind:
        return SignalKind.LLM_JUDGE_ABSOLUTE

    @property
    def cost_estimate(self) -> Cost:
        # Rubric + question + trajectory excerpt: ~2-3k tokens typical.
        return Cost(tokens=2500, dollars=0.012, wall_clock_sec=3.0)

    async def measure(
        self,
        candidate: Artifact,
        trajectory: Trajectory | None = None,
        reference: Artifact | None = None,  # unused
        ground_truth: Any | None = None,     # unused
    ) -> GapMeasurement:
        if trajectory is None:
            raise ValueError("LiveTrajectoryJudge requires a trajectory to evaluate")

        rendered_trajectory = self._render_trajectory(trajectory)
        user_prompt = f"""User question:
{trajectory.task}

Trajectory steps the agent took:
{rendered_trajectory}

Agent's final answer:
{trajectory.final_output or '(no final output)'}

Rate this response per the rubric. Reply as JSON."""

        response = await helix_acompletion(
            model=self.model,
            messages=[
                cacheable_system_message(self.rubric, self.model),
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content or "{}"
        try:
            data = json.loads(raw)
            verdict = _LiveVerdict(**data)
        except Exception:
            verdict = _LiveVerdict(
                score=0.5,
                reasoning="(judge output malformed; defaulted to neutral)",
                confidence=0.0,
            )

        usage = getattr(response, "usage", None)
        cost = Cost(tokens=getattr(usage, "total_tokens", 0) if usage else 0)

        feedback = verdict.reasoning
        if verdict.flags:
            feedback += f"\n\nFlags: {', '.join(verdict.flags)}"

        return GapMeasurement(
            score=verdict.score,
            preference=Preference.NONE,  # absolute, not pairwise
            feedback=feedback,
            confidence=verdict.confidence,
            rubric_id=self._rubric_id,
            cost=cost,
            metadata={
                "judge_kind": "live_trajectory",
                "judge_model": self.model,
                "flags": verdict.flags,
                "trajectory_id": trajectory.id,
            },
        )

    @staticmethod
    def _render_trajectory(trajectory: Trajectory) -> str:
        """Compact summary of retrieval + tool activity. Skips raw model output
        because the judge has the final answer separately."""
        lines: list[str] = []
        for s in trajectory.steps:
            if s.kind == StepKind.TOOL_CALL:
                name = s.payload.get("name", "?")
                args = s.payload.get("arguments", {})
                arg_str = ", ".join(f"{k}={str(v)[:80]}" for k, v in (args or {}).items())
                lines.append(f"  [{s.index}] tool_call: {name}({arg_str})")
            elif s.kind == StepKind.TOOL_RESULT:
                result = s.payload.get("result")
                if isinstance(result, list) and result and isinstance(result[0], dict) and "source" in result[0]:
                    sources = [
                        f"{h.get('source', '?')}:p{h.get('page', '?')}"
                        for h in result[:5]
                    ]
                    lines.append(f"  [{s.index}] tool_result: {len(result)} hits from {sources}")
                else:
                    lines.append(f"  [{s.index}] tool_result: {str(result)[:150]}")
        return "\n".join(lines) if lines else "  (no tool activity)"
