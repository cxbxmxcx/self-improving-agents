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

import json as _json
import re as _re
from typing import Any

from pydantic import BaseModel, Field

from helix.llm_call import acompletion as helix_acompletion

from helix.artifact import Artifact
from helix.signal import (
    Cost,
    GapMeasurement,
    Preference,
    SignalKind,
    derive_signal_id,
)
from helix.trajectory import Trajectory


class _ReflectionOutput(BaseModel):
    feedback: str = Field(
        description="Concrete, actionable critique that a mutation operator can act on."
    )
    confidence: float = Field(ge=0.0, le=1.0)


def _strip_code_fences(raw: str) -> str:
    """Anthropic JSON-mode responses often arrive wrapped in a ```json fence.
    Strip it so json.loads sees clean JSON."""
    s = raw.strip()
    if s.startswith("```"):
        s = _re.sub(r"^```[a-zA-Z]*\n?", "", s)
        s = _re.sub(r"\n?```$", "", s)
    return s.strip()


def _coerce_feedback(value: Any) -> str:
    """Reflection feedback must be a string, but models sometimes return a nested
    object or list. Flatten those into one readable critique rather than discarding
    the diagnosis the reflector actually produced."""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return "; ".join(_coerce_feedback(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return "; ".join(_coerce_feedback(v) for v in value)
    return str(value)


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
        version: int = 1,
    ) -> None:
        self.model = model
        self.prompt = prompt
        self._version = version

    @property
    def kind(self) -> SignalKind:
        return SignalKind.REFLECTION

    @property
    def cost_estimate(self) -> Cost:
        return Cost(tokens=1500, dollars=0.008, wall_clock_sec=4.0)

    @property
    def signal_id(self) -> str:
        return derive_signal_id(
            "Reflection",
            {"model": self.model, "prompt": self.prompt},
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
the candidate prompt? Reply with a brief plain-text critique."""

        # Reflection is a textual signal, so we ask for prose rather than forcing
        # JSON mode (which is not honored uniformly across providers and makes some
        # models emit prose plus a stray JSON block). The parse below still accepts
        # a JSON object if a model returns one.
        response = await helix_acompletion(
            model=self.model,
            messages=[
                {"role": "system", "content": self.prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        raw = response.choices[0].message.content or ""
        cleaned = _strip_code_fences(raw)
        try:
            data = _json.loads(cleaned)
            out = _ReflectionOutput(
                feedback=_coerce_feedback(data.get("feedback", "")),
                confidence=float(data.get("confidence", 0.5)),
            )
        except Exception:
            # JSON mode is not guaranteed on every provider, and the model often
            # returns a plain-prose critique instead. For a reflection signal that
            # prose IS the feedback, so use it directly rather than losing the
            # diagnosis the reflector actually produced.
            out = _ReflectionOutput(
                feedback=cleaned.strip() or "(reflection produced no output)",
                confidence=0.5,
            )

        usage = getattr(response, "usage", None)
        cost = Cost(tokens=getattr(usage, "total_tokens", 0) if usage else 0)

        return GapMeasurement(
            score=None,
            preference=Preference.NONE,
            feedback=out.feedback,
            confidence=out.confidence,
            signal_id=self.signal_id,
            signal_version=self.signal_version,
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
