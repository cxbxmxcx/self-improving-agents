"""A judge: a model that reads a run and says whether it met a rubric.

Most deployments have no programmatic checker, so people are the only other
source of verdicts, and they are slow and few. A judge fills the gap, with
limits set by the plan's decision 6:

- its verdicts carry source "judge" and are recorded with the run, where
  learners can read them;
- the detector ignores them unless "judge" is added to `gating_sources`;
- learned code never installs on a judge's word alone: when the verifier is a
  judge, approval defaults to "human".

    meta = Metacognition([...], verifier=Judge("The claim is the policy total, in US dollars."))

The judge sees the request, each tool call with what it returned (trimmed),
and the final answer, and replies in JSON. A reply it can't read leaves the
run without a verdict rather than counting it as rejected.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from metacog.meta.parts import Verdict

JUDGE_PROMPT = """You judge whether an assistant handled a request correctly. You will see
the request, every tool the assistant called with what the tool returned, and
its final answer. Judge the outcome against this rubric:

{rubric}

Reply with JSON only: {{"accepted": true or false, "reason": "one sentence"}}."""


def _trim(value: Any, limit: int) -> str:
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    return text if len(text) <= limit else text[:limit] + " ..."


def render_run(request: str, result: Any, limit: int = 1500) -> str:
    """The run as the judge reads it: request, tool calls with results, final answer."""
    from metacog.items import ToolCallItem, ToolCallOutputItem

    lines = [f"Request: {request}"]
    for item in getattr(result, "new_items", []):
        if isinstance(item, ToolCallItem):
            lines.append(f"- called {item.name}({json.dumps(item.arguments, default=str)})")
        elif isinstance(item, ToolCallOutputItem):
            lines.append(f"  returned {_trim(item.output, limit)}")
    lines.append(f"Final answer: {_trim(getattr(result, 'final_output', ''), limit)}")
    return "\n".join(lines)


@dataclass
class Judge:
    """A verifier that asks a model whether a run met `rubric`. It needs no checker, and it is not one."""

    rubric: str
    model: str | None = None  # default: the agent's own model
    params: dict[str, Any] = field(default_factory=dict)
    limit: int = 1500  # characters of each tool result and the answer the judge reads

    async def __call__(self, ctx: Any, result: Any) -> Verdict | None:
        from helix import llm_call

        from metacog.model_settings import DEFAULT_MODEL

        model = self.model or getattr(getattr(result, "last_agent", None), "model", None) or DEFAULT_MODEL
        request = result.input if isinstance(result.input, str) else str(result.input[-1].get("content", ""))
        messages = [{"role": "system", "content": JUDGE_PROMPT.format(rubric=self.rubric)},
                    {"role": "user", "content": render_run(request, result, self.limit)}]
        response = await llm_call.acompletion(model=model, messages=messages, **self.params)
        return self.read(response.choices[0].message.content or "")

    @staticmethod
    def read(reply: str) -> Verdict | None:
        """The verdict in a judge's reply, or None when the reply holds none."""
        found = re.search(r"\{.*\}", reply, re.DOTALL)
        try:
            data = json.loads(found.group(0)) if found else None
        except ValueError:
            data = None
        if not isinstance(data, dict) or not isinstance(data.get("accepted"), bool):
            return None
        return Verdict(data["accepted"], str(data.get("reason", "")), "judge")
