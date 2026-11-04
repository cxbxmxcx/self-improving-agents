"""An LLM answering typed questions without reasoning, as a decision model.

The fallback when no hosted decision model is available: the same typed
questions go to any LiteLLM model, which returns probabilities as JSON. It
is usually slower and costlier than Jev or the Decisions API, which
chapter 6 measures rather than assumes.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import replace
from typing import Any

from helix.deciders.base import (
    Answer,
    Choice,
    DecisionUsage,
    Probability,
    Question,
    Score,
    question_to_dict,
)
from helix.llm_call import acompletion

SYSTEM_PROMPT = """You are a fast decision model. Read the state and answer every question with numbers only.
Return one JSON object: {"answers": {"<question name>": {...}}}.
- choice question: {"probabilities": {"<option>": p, ...}} over its options, summing to 1.
- score question: {"probabilities": {"0": p, "1": p, ...}} over its levels by index, 0 being the lowest, summing to 1.
- probability question: {"probability": p}, the chance that the statement holds.
Give no explanations."""


def _parse_json(text: str) -> dict[str, Any]:
    """The first JSON object in a reply; models sometimes wrap it in prose or fences."""
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        match = re.search(r"\{.*\}", text or "", re.DOTALL)
        if not match:
            return {}
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}


def _normalized(probabilities: dict[str, Any], keys: list[str]) -> dict[str, float]:
    """Probabilities over exactly `keys`, renormalized; uniform when the reply gave none."""
    clean = {k: max(0.0, float(probabilities.get(k, 0.0) or 0.0)) for k in keys}
    total = sum(clean.values())
    if total <= 0:
        return {k: 1.0 / len(keys) for k in keys}
    return {k: v / total for k, v in clean.items()}


def _answer(question: Question, raw: dict[str, Any]) -> Answer:
    if isinstance(question, Choice):
        probs = _normalized(raw.get("probabilities") or {}, list(question.criteria))
        best = max(probs, key=probs.get)
        return Answer("choice", best, probs[best], probs)
    if isinstance(question, Score):
        n = len(question.levels)
        given = raw.get("probabilities") or {}
        # Accept level labels as well as indices; models drift between the two.
        by_index = {str(question.levels.index(k)) if k in question.levels else str(k): v for k, v in given.items()}
        probs = _normalized(by_index, [str(i) for i in range(n)])
        score = sum(int(i) * p for i, p in probs.items())
        return Answer("score", score, max(probs.values()), probs, score / (n - 1) if n > 1 else None)
    if isinstance(question, Probability):
        p = raw.get("probability")
        value = min(1.0, max(0.0, float(p))) if isinstance(p, (int, float)) else 0.5
        return Answer("probability", value)
    raise TypeError(f"not a decision question: {question!r}")


def _dollars(response: Any) -> float:
    try:
        import litellm

        return float(litellm.completion_cost(completion_response=response) or 0.0)
    except Exception:  # unknown model prices should not break a decision
        return 0.0


class LLMDecider:
    """Any LiteLLM model as a decision model. `params` passes through to the
    call, for example a minimal reasoning effort on a reasoning model."""

    def __init__(self, model: str, *, params: dict[str, Any] | None = None) -> None:
        self.model = model
        self.name = f"llm/{model}"
        self.params = dict(params or {})
        self._usage = DecisionUsage()

    async def ask(self, state: str | dict[str, Any], questions: dict[str, Question]) -> dict[str, Answer]:
        return (await self.ask_with_usage(state, questions))[0]

    async def ask_with_usage(self, state: str | dict[str, Any],
                             questions: dict[str, Question]) -> tuple[dict[str, Answer], DecisionUsage]:
        prompt = json.dumps({"state": state, "questions": {k: question_to_dict(q) for k, q in questions.items()}},
                            ensure_ascii=False)
        started = time.perf_counter()
        response = await acompletion(
            model=self.model,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
            **self.params,
        )
        tokens = getattr(getattr(response, "usage", None), "prompt_tokens", None)
        call = DecisionUsage(calls=1, input_tokens=tokens if isinstance(tokens, int) else 0,
                             dollars=_dollars(response), seconds=time.perf_counter() - started, version=self.model)
        self._usage.add(call)
        reply = _parse_json(response.choices[0].message.content or "")
        answers = reply.get("answers") or {}
        return {name: _answer(q, answers.get(name) or {}) for name, q in questions.items()}, call

    def usage(self) -> DecisionUsage:
        return replace(self._usage)
