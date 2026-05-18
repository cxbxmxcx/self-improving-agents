"""Reflection signal.

A model critiques a trajectory and produces textual feedback. Reflection is a
Signal in the framework, not a Search method. SPEC section 3.3 locks this in:
the architectural payoff is that GEPA's reflective mutation, Reflexion's
verbal reinforcement, and the metacognitive Reflector of Chapter 5 all consume
the same Signal output.

This implementation reads the candidate Artifact and its produced Trajectory,
asks the reflector model what went wrong (or right), and fills the `feedback`
field of the GapMeasurement. The `score` and `preference` fields stay None /
NONE because reflection is purely textual.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from helix.llm_call import acompletion as helix_acompletion

from helix.artifact import Artifact
from helix.signal import (
    Cost,
    GapMeasurement,
    Preference,
    SignalKind,
)
from helix.trajectory import Trajectory


class _ReflectionOutput(BaseModel):
    feedback: str = Field(
        description="Concrete, actionable critique that a mutation operator can act on."
    )
    confidence: float = Field(ge=0.0, le=1.0)


DEFAULT_REFLECTION_PROMPT = """You are critiquing an agent's behavior on one task.

You will see the agent's system prompt (the candidate artifact), the user's
question, and the trajectory of steps the agent took. Read the trajectory and
identify what specifically the candidate prompt caused the agent to do well or
poorly. Be concrete: name retrieval queries that missed, citations the agent
should have made, hallucinations to call out, abstentions that were warranted.

Your output is consumed by a mutation operator that will edit the candidate
prompt based on your feedback, so be directly actionable. Do not produce vague
generalities like 'the agent should be more careful.' Instead say things like
'the agent answered without retrieving; the prompt should require a retrieve
call before any factual claim.'"""


class Reflection:
    """Reflection Signal: textual critique of a Trajectory."""

    def __init__(
        self,
        model: str = "gpt-4o",
        prompt: str = DEFAULT_REFLECTION_PROMPT,
    ) -> None:
        self.model = model
        self.prompt = prompt

    @property
    def kind(self) -> SignalKind:
        return SignalKind.REFLECTION

    @property
    def cost_estimate(self) -> Cost:
        return Cost(tokens=1500, dollars=0.008, wall_clock_sec=4.0)

    async def measure(
        self,
        candidate: Artifact,
        trajectory: Trajectory | None = None,
        reference: Artifact | None = None,
        ground_truth: Any | None = None,
    ) -> GapMeasurement:
        if trajectory is None:
            raise ValueError("Reflection signal requires a trajectory to critique")

        traj_summary = self._render_trajectory(trajectory)
        context = ground_truth or {}
        question = context.get("question", trajectory.task)

        user_prompt = f"""Candidate prompt under critique:
---
{candidate.content if isinstance(candidate.content, str) else str(candidate.content)}
---

User question:
{question}

Trajectory (numbered steps):
{traj_summary}

Final agent answer:
{trajectory.final_output if trajectory.final_output else '(no final output)'}

What went well, what went poorly, and what concrete edits should be made to
the candidate prompt? Reply as JSON with two fields:
  "feedback": the critique string
  "confidence": float in [0, 1]"""

        response = await helix_acompletion(
            model=self.model,
            messages=[
                {"role": "system", "content": self.prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content or "{}"
        try:
            import json as _json
            data = _json.loads(raw)
            out = _ReflectionOutput(**data)
        except Exception:
            out = _ReflectionOutput(feedback="(reflection output malformed)", confidence=0.0)

        usage = getattr(response, "usage", None)
        cost = Cost(tokens=getattr(usage, "total_tokens", 0) if usage else 0)

        return GapMeasurement(
            score=None,
            preference=Preference.NONE,
            feedback=out.feedback,
            confidence=out.confidence,
            cost=cost,
            metadata={"reflector_model": self.model},
        )

    @staticmethod
    def _render_trajectory(trajectory: Trajectory) -> str:
        """Render the trajectory as numbered steps for the reflector to read."""
        lines = []
        for s in trajectory.steps:
            payload_excerpt = str(s.payload)[:300]
            lines.append(f"  [{s.index}] {s.kind.value}: {payload_excerpt}")
        return "\n".join(lines) if lines else "(no steps)"
