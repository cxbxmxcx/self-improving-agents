"""Jev, TypeSafe's System One decision model, behind the `Decider` interface.

Wire format (docs.typesafe.ai/api.md): a map of typed questions keyed by
name, answered under the same keys. A probability question travels as a
Noul; a score question's levels travel as its criteria. Score answers carry
per-level probabilities keyed by level index, which place them on [0, 1].
"""

from __future__ import annotations

from typing import Any

from helix.deciders.base import (
    Answer,
    Choice,
    HostedDecider,
    Probability,
    Question,
    Score,
    score_position,
)

# Dollars per input token; TypeSafe does not charge for output. Check the
# provider's pricing page before quoting a cost in the book.
JEV_PRICE_PER_INPUT_TOKEN = 42 / 1e9


class TypeSafeDecider(HostedDecider):
    url = "https://api.typesafe.ai/v1/systemone"
    # TypeSafe's SDK reads TYPESAFE_API_KEY; JEV_API_KEY is accepted as well.
    key_envs = ("TYPESAFE_API_KEY", "JEV_API_KEY")
    price_per_input_token = JEV_PRICE_PER_INPUT_TOKEN

    def __init__(self, model: str = "jev-latest", **kwargs: Any) -> None:
        super().__init__(model, **kwargs)
        self.model = model

    def _request(self, state: str | dict[str, Any], questions: dict[str, Question]) -> dict[str, Any]:
        asked: dict[str, Any] = {}
        for name, q in questions.items():
            if isinstance(q, Choice):
                asked[name] = {"type": "choice", "instructions": q.instructions, "criteria": dict(q.criteria)}
            elif isinstance(q, Score):
                asked[name] = {"type": "score", "instructions": q.instructions, "criteria": list(q.levels)}
            elif isinstance(q, Probability):
                asked[name] = {"type": "noul", "instructions": q.instructions}
                if q.criteria:
                    asked[name]["criteria"] = dict(q.criteria)
            else:
                raise TypeError(f"not a decision question: {q!r}")
        return {"model": self.model, "state": state, "questions": asked}

    def _answers(self, reply: dict[str, Any], questions: dict[str, Question]) -> dict[str, Answer]:
        out: dict[str, Answer] = {}
        for name, a in reply["answers"].items():
            probabilities = {str(k): float(v) for k, v in (a.get("probabilities") or {}).items()}
            if a["type"] == "choice":
                out[name] = Answer("choice", a["choice"], a.get("confidence"), probabilities)
            elif a["type"] == "score":
                q = questions.get(name)
                levels = list(q.levels) if isinstance(q, Score) else []
                legend = [str(k) for k in (a.get("legend") or {})]
                out[name] = Answer("score", float(a["score"]), a.get("confidence"), probabilities,
                                   score_position(a["score"], levels, probabilities, legend or None))
            else:
                out[name] = Answer("probability", float(a["noul"]))
        return out
