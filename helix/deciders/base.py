"""Decision models: fast, typed judgments that hook code acts on.

A decision model reads a state and answers named questions with a choice
among described options, a position on ordered levels, or the probability
that a statement holds. It writes no prose, so its answers can be compared
with thresholds in code.

Every provider sits behind the same `Decider` interface, so the harness
never binds to one: Jev (TypeSafe), OpenAI's Decisions API, an LLM answering
without reasoning, and a replay of recorded answers all answer the same
questions in the same shape.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Protocol, Union, runtime_checkable

import httpx

from helix.signal import Cost


@dataclass(frozen=True)
class Choice:
    """Pick one option. `criteria` maps each option to what it covers, or
    to None when the option's name says enough."""

    instructions: str
    criteria: dict[str, str | None]


@dataclass(frozen=True)
class Score:
    """A position on ordered levels, lowest first."""

    instructions: str
    levels: list[str]


@dataclass(frozen=True)
class Probability:
    """The probability that a statement holds: near 1 yes, near 0 no.

    Jev calls this question a Noul and OpenAI a predicate. The optional
    `criteria` ({"true": ..., "false": ...}) say what yes and no mean; only
    providers that accept them receive them.
    """

    instructions: str
    criteria: dict[str, str] | None = None


Question = Union[Choice, Score, Probability]


class MissingAnswer(KeyError):
    """A replay met a question its recording has no answer for."""


@dataclass(frozen=True)
class Answer:
    """One typed answer.

    `value` is the chosen option, the provider's own score, or the
    probability. `position` places a score answer on [0, 1] across its
    levels, so a threshold means the same thing whichever provider answered.
    """

    kind: str  # "choice", "score", or "probability"
    value: str | float
    confidence: float | None = None
    probabilities: dict[str, float] = field(default_factory=dict)
    position: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Answer:
        return cls(**data)


@dataclass
class DecisionUsage:
    """What a decision model has cost so far: calls, input tokens, dollars,
    seconds, and the model version that answered last."""

    calls: int = 0
    input_tokens: int = 0
    dollars: float = 0.0
    seconds: float = 0.0
    version: str | None = None

    def as_cost(self) -> Cost:
        return Cost(tokens=self.input_tokens, dollars=self.dollars, wall_clock_sec=self.seconds)

    def add(self, call: DecisionUsage) -> None:
        self.calls += call.calls
        self.input_tokens += call.input_tokens
        self.dollars += call.dollars
        self.seconds += call.seconds
        self.version = call.version or self.version

    def to_dict(self) -> dict[str, Any]:
        return {"calls": self.calls, "input_tokens": self.input_tokens, "dollars": self.dollars,
                "seconds": round(self.seconds, 4), "version": self.version}


@runtime_checkable
class Decider(Protocol):
    """The interface every decision model implements."""

    name: str

    async def ask(self, state: str | dict[str, Any], questions: dict[str, Question]) -> dict[str, Answer]:
        """Each question's answer, under the name it was asked with."""
        ...

    async def ask_with_usage(self, state: str | dict[str, Any],
                             questions: dict[str, Question]) -> tuple[dict[str, Answer], DecisionUsage]:
        """The answers and what this one call cost. Hooks use it so a run's
        cost stays exact even when several runs share one decision model."""
        ...

    def usage(self) -> DecisionUsage:
        """A snapshot of what this decision model has cost so far."""
        ...


async def ask_measured(decider: Any, state: str | dict[str, Any],
                       questions: dict[str, Question]) -> tuple[dict[str, Answer], DecisionUsage]:
    """`ask_with_usage` when the decision model offers it; otherwise `ask`,
    timed here, so third-party decision models still work."""
    method = getattr(decider, "ask_with_usage", None)
    if callable(method):
        return await method(state, questions)
    started = time.perf_counter()
    answers = await decider.ask(state, questions)
    return answers, DecisionUsage(calls=1, seconds=time.perf_counter() - started,
                                  version=getattr(decider, "name", None))


def question_to_dict(question: Question) -> dict[str, Any]:
    """A provider-neutral description of a question, used for hashing and recording."""
    if isinstance(question, Choice):
        return {"type": "choice", "instructions": question.instructions, "criteria": dict(question.criteria)}
    if isinstance(question, Score):
        return {"type": "score", "instructions": question.instructions, "levels": list(question.levels)}
    if isinstance(question, Probability):
        out: dict[str, Any] = {"type": "probability", "instructions": question.instructions}
        if question.criteria:
            out["criteria"] = dict(question.criteria)
        return out
    raise TypeError(f"not a decision question: {question!r}")


def question_from_dict(data: dict[str, Any]) -> Question:
    kind = data["type"]
    if kind == "choice":
        return Choice(data["instructions"], dict(data["criteria"]))
    if kind == "score":
        return Score(data["instructions"], list(data["levels"]))
    if kind == "probability":
        return Probability(data["instructions"], data.get("criteria"))
    raise ValueError(f"unknown question type: {kind}")


def question_key(state: str | dict[str, Any], questions: dict[str, Question]) -> str:
    """A stable key for one exchange, so recorded answers can be found again."""
    canonical = json.dumps(
        {"state": state, "questions": {k: question_to_dict(q) for k, q in questions.items()}},
        sort_keys=True, ensure_ascii=False, separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def state_text(state: str | dict[str, Any]) -> str:
    return state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)


def score_position(raw: float, levels: list[str], probabilities: dict[str, float],
                   index_hint: list[str] | None = None, raw_scale: str = "index") -> float | None:
    """Place a score answer on [0, 1].

    Per-level probabilities are preferred because they say which level each
    weight belongs to, whether a provider keys them by label or by index and
    whether it counts from 0 or 1. Without them, `raw_scale` says how to read
    the raw score: "index" for a 0-based level index, "unit" for [0, 1].
    """
    n = len(levels)
    if n < 2:
        return None
    if probabilities:
        total = sum(probabilities.values()) or 1.0
        if all(k in levels for k in probabilities):
            weighted = sum(p * levels.index(k) for k, p in probabilities.items()) / total
            return min(1.0, max(0.0, weighted / (n - 1)))
        if all(k.lstrip("-").isdigit() for k in probabilities):
            indices = sorted(int(k) for k in (index_hint or probabilities))
            base = 1 if indices and indices[0] >= 1 and indices[-1] >= n else 0
            weighted = sum(p * (int(k) - base) for k, p in probabilities.items()) / total
            return min(1.0, max(0.0, weighted / (n - 1)))
    if raw_scale == "unit":
        return min(1.0, max(0.0, float(raw)))
    return min(1.0, max(0.0, float(raw) / (n - 1)))


class HostedDecider:
    """Shared plumbing for hosted decision models: auth, retries on 429 and
    529, and usage accounting. Subclasses supply the wire format."""

    url: str = ""
    # Environment variables that may hold the key, preferred first.
    key_envs: tuple[str, ...] = ()
    price_per_input_token: float = 0.0
    retry_statuses = (429, 529)

    def __init__(self, name: str, *, api_key: str | None = None,
                 transport: httpx.AsyncBaseTransport | None = None,
                 timeout: float = 30.0, max_attempts: int = 6) -> None:
        key = api_key or next((os.environ[v] for v in self.key_envs if os.environ.get(v)), None)
        if not key:
            raise RuntimeError(f"{' or '.join(self.key_envs)} is not set; add it to the repository's .env file")
        self.name = name
        self.max_attempts = max_attempts
        self._client = httpx.AsyncClient(timeout=timeout, transport=transport,
                                         headers={"Authorization": f"Bearer {key}"})
        self._usage = DecisionUsage()

    async def ask(self, state: str | dict[str, Any], questions: dict[str, Question]) -> dict[str, Answer]:
        return (await self.ask_with_usage(state, questions))[0]

    async def ask_with_usage(self, state: str | dict[str, Any],
                             questions: dict[str, Question]) -> tuple[dict[str, Answer], DecisionUsage]:
        body = self._request(state, questions)
        # Seconds include any retries: that is the latency a hook waits for.
        started = time.perf_counter()
        for attempt in range(self.max_attempts):
            response = await self._client.post(self.url, json=body)
            if response.status_code in self.retry_statuses:
                await asyncio.sleep(0.5 * 2 ** attempt)
                continue
            response.raise_for_status()
            reply = response.json()
            tokens = int((reply.get("usage") or {}).get("input_tokens") or 0)
            call = DecisionUsage(calls=1, input_tokens=tokens, dollars=tokens * self.price_per_input_token,
                                 seconds=time.perf_counter() - started, version=reply.get("model") or self.name)
            self._usage.add(call)
            return self._answers(reply, questions), call
        raise RuntimeError(f"{self.name} kept answering {self.retry_statuses}")

    def usage(self) -> DecisionUsage:
        return replace(self._usage)

    async def aclose(self) -> None:
        await self._client.aclose()

    def _request(self, state: str | dict[str, Any], questions: dict[str, Question]) -> dict[str, Any]:
        raise NotImplementedError

    def _answers(self, reply: dict[str, Any], questions: dict[str, Question]) -> dict[str, Answer]:
        raise NotImplementedError
