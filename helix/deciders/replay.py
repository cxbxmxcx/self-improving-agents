"""Record a decision model's answers once, then replay them without a key.

The book's measured runs use live decision models; readers replay the
recorded answers, so every script runs without an API key and reproduces
the same decisions. A replay also reports the recorded tokens, dollars,
and seconds, so cost and latency comparisons survive the replay.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from helix.deciders.base import (
    Answer,
    Decider,
    DecisionUsage,
    MissingAnswer,
    Question,
    ask_measured,
    question_key,
    question_to_dict,
)


class RecordingDecider:
    """Wrap a live decision model and append every exchange to a JSONL file."""

    def __init__(self, inner: Decider, path: Path | str) -> None:
        self.inner = inner
        self.path = Path(path)
        self.name = inner.name

    async def ask(self, state: str | dict[str, Any], questions: dict[str, Question]) -> dict[str, Answer]:
        return (await self.ask_with_usage(state, questions))[0]

    async def ask_with_usage(self, state: str | dict[str, Any],
                             questions: dict[str, Question]) -> tuple[dict[str, Answer], DecisionUsage]:
        answers, call = await ask_measured(self.inner, state, questions)
        record = {
            "key": question_key(state, questions),
            "decider": self.inner.name,
            "version": call.version,
            "state": state,
            "questions": {k: question_to_dict(q) for k, q in questions.items()},
            "answers": {k: a.to_dict() for k, a in answers.items()},
            "input_tokens": call.input_tokens,
            "dollars": call.dollars,
            "seconds": round(call.seconds, 4),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        return answers, call

    def usage(self) -> DecisionUsage:
        return self.inner.usage()

    def live_usage(self) -> DecisionUsage:
        """What the calls that actually reached the model cost."""
        return self.inner.usage()


class ReplayDecider:
    """Serve recorded answers by the hash of the state and the questions.

    A miss raises in strict mode, so a changed question cannot silently reuse
    a stale answer; with a `fallback` decision model the miss is answered live.
    """

    def __init__(self, path: Path | str, *, strict: bool = True, fallback: Decider | None = None,
                 name: str | None = None) -> None:
        self.path = Path(path)
        self.strict = strict
        self.fallback = fallback
        self._records: dict[str, dict[str, Any]] = {}
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    record = json.loads(line)
                    self._records[record["key"]] = record
        recorded = {r["decider"] for r in self._records.values()}
        self.name = name or (f"replay/{next(iter(recorded))}" if len(recorded) == 1 else "replay")
        self._usage = DecisionUsage()

    def __len__(self) -> int:
        return len(self._records)

    async def ask(self, state: str | dict[str, Any], questions: dict[str, Question]) -> dict[str, Answer]:
        return (await self.ask_with_usage(state, questions))[0]

    async def ask_with_usage(self, state: str | dict[str, Any],
                             questions: dict[str, Question]) -> tuple[dict[str, Answer], DecisionUsage]:
        key = question_key(state, questions)
        record = self._records.get(key)
        if record is None:
            if self.fallback is not None:
                answers, call = await ask_measured(self.fallback, state, questions)
                # Report the seconds as a recording keeps them, so a later
                # replay of this run sees exactly the numbers it saw.
                call = replace(call, seconds=round(call.seconds, 4))
                # A decision model may answer a repeated question a little
                # differently; keep the first answer so one run sees one.
                self._records[key] = {"key": key, "answers": {k: a.to_dict() for k, a in answers.items()},
                                      "input_tokens": call.input_tokens, "dollars": call.dollars,
                                      "seconds": call.seconds, "version": call.version}
                self._usage.add(call)
                return answers, call
            if self.strict:
                raise MissingAnswer(f"no recorded answer for this state and these questions (key {key[:12]})")
            return {}, DecisionUsage()
        call = DecisionUsage(calls=1, input_tokens=int(record.get("input_tokens") or 0),
                             dollars=float(record.get("dollars") or 0.0),
                             seconds=float(record.get("seconds") or 0.0), version=record.get("version"))
        self._usage.add(call)
        return {k: Answer.from_dict(v) for k, v in record["answers"].items()}, call

    def usage(self) -> DecisionUsage:
        return replace(self._usage)

    def live_usage(self) -> DecisionUsage:
        """What the calls that actually reached the model cost: the misses a fallback answered."""
        return self.fallback.usage() if self.fallback is not None else DecisionUsage()
