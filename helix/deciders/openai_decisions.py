"""OpenAI's Decisions API behind the `Decider` interface.

Wire format: a list of named questions with the state sent as text input.
A probability question travels as a predicate, a choice's options as
described values, and a score's levels as labels. Answers come back as a
list keyed by name; the adapter returns them in the same shape as Jev's.
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
    state_text,
)

# Dollars per input token for the Decisions API. Check the provider's pricing
# page before quoting a cost in the book.
OPENAI_DECISIONS_PRICE_PER_INPUT_TOKEN = 0.10 / 1e6


class OpenAIDecisionsDecider(HostedDecider):
    url = "https://api.openai.com/v1/decisions"
    key_envs = ("OPENAI_API_KEY",)
    price_per_input_token = OPENAI_DECISIONS_PRICE_PER_INPUT_TOKEN

    def __init__(self, model: str = "gpt-6-luna", **kwargs: Any) -> None:
        super().__init__(f"openai/{model}", **kwargs)
        self.model = model

    def _request(self, state: str | dict[str, Any], questions: dict[str, Question]) -> dict[str, Any]:
        asked: list[dict[str, Any]] = []
        for name, q in questions.items():
            if isinstance(q, Choice):
                asked.append({"type": "choice", "name": name, "instructions": q.instructions,
                              "choices": [{"value": k, "description": v or k} for k, v in q.criteria.items()]})
            elif isinstance(q, Score):
                asked.append({"type": "score", "name": name, "instructions": q.instructions,
                              "levels": [{"label": level} for level in q.levels]})
            elif isinstance(q, Probability):
                asked.append({"type": "predicate", "name": name, "instructions": q.instructions})
            else:
                raise TypeError(f"not a decision question: {q!r}")
        return {"model": self.model, "input": state_text(state), "questions": asked}

    def _answers(self, reply: dict[str, Any], questions: dict[str, Question]) -> dict[str, Answer]:
        out: dict[str, Answer] = {}
        for a in reply["answers"]:
            name = a["name"]
            probabilities = {str(p.get("value")): float(p["probability"]) for p in a.get("probabilities") or []}
            if a["type"] == "choice":
                out[name] = Answer("choice", a["choice"], a.get("confidence"), probabilities)
            elif a["type"] == "score":
                q = questions.get(name)
                levels = list(q.levels) if isinstance(q, Score) else []
                out[name] = Answer("score", float(a["score"]), a.get("confidence"), probabilities,
                                   score_position(a["score"], levels, probabilities))
            else:
                out[name] = Answer("probability", float(a["probability"]))
        return out
