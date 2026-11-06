"""Choosing a learner by measurement: metacognition about learning (A5, level 2, in ARCHITECTURE.md).

When learning keeps failing for a procedure, the loop holds a tournament. Each candidate mechanism learns
from the same judged runs, and the function it writes is scored on the gate's held-out cases, the way the
gate itself would try it. The winner becomes that procedure's learner and is kept in the store, so every
process sees the choice. Its function is that attempt's candidate, so it goes through the gate's decision
like any other.

    meta = Metacognition([...], selection=LearnerSelection())        # the five seeds compete
    meta = Metacognition([...], selection=LearnerSelection(candidates=["skill_synthesis", designed]))

The tournament is opt-in, and every trial is a real learning attempt with its own writer calls, so its cost
counts against the learning budget as one attempt.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from helix.memory.learning_mechanisms import Contract, Episode, Writer

from metacog.meta.mechanisms import Mechanism, spent, writer_cost
from metacog.meta.parts import Practice
from metacog.meta.sources import SEEDS


@dataclass
class LearnerSelection:
    """When the loop holds a learner tournament for a procedure, and which mechanisms compete.

    candidates          mechanism names (chapter 7's seeds run as source) or instances, such as a
                        SourceLearner a search designed
    after_refusals      learned versions refused by the gate in a row that start a tournament
    quarantined_within  a learned version quarantined within this many runs of being trusted starts
                        one too (None: never)
    """

    candidates: Sequence[str | Mechanism] = SEEDS
    after_refusals: int = 2
    quarantined_within: int | None = 8


@dataclass
class LearningProblem:
    """What a learner faces: a procedure's contract, the judged runs to learn from, how a candidate is
    tried while learning (`practice`), and the held-out cases a learned function is scored on (`score`)."""

    procedure: str
    contract: Contract
    episodes: list[Episode]
    practice: Practice
    cases: list[str]
    score: Practice


@dataclass
class LearnerTrial:
    """One mechanism's turn in a tournament."""

    mechanism: str
    accepted: int
    total: int
    writer_calls: int
    dollars: float
    source: str | None
    failures: list[str] = field(default_factory=list)
    note: str = ""
    learner: Mechanism | None = field(default=None, repr=False, compare=False)

    def to_dict(self) -> dict[str, Any]:
        return {"mechanism": self.mechanism, "accepted": self.accepted, "total": self.total,
                "writer_calls": self.writer_calls, "dollars": round(self.dollars, 6), "note": self.note}

    def line(self) -> str:
        return f"{self.mechanism} {self.accepted}/{self.total} ${self.dollars:.4f}"


@dataclass
class LearnerChoice:
    """A tournament's result: the winner (most held-out cases accepted, then the cheapest, then the first
    named) and every trial."""

    winner: LearnerTrial
    trials: list[LearnerTrial]

    @property
    def writer_calls(self) -> int:
        return sum(t.writer_calls for t in self.trials)

    @property
    def dollars(self) -> float:
        return sum(t.dollars for t in self.trials)

    def to_dict(self) -> dict[str, Any]:
        return {"winner": self.winner.mechanism, "trials": [t.to_dict() for t in self.trials]}

    def line(self) -> str:
        return ", ".join(t.line() for t in self.trials)


async def run_tournament(problem: LearningProblem, candidates: Sequence[Mechanism], *, writer_model: str,
                         writer_params: dict[str, Any] | None = None,
                         prices: dict[str, tuple[float, ...]] | None = None) -> LearnerChoice:
    """Let each candidate learn the problem with its own writer, score each function on the held-out cases,
    and pick the winner. A candidate that fails or runs out of budget scores nothing but keeps its spend."""
    if not candidates:
        raise ValueError("a tournament needs at least one candidate")
    trials: list[LearnerTrial] = []
    for learner in candidates:
        writer = Writer(model=writer_model, params=dict(writer_params or {}))
        learned, ended, note = None, None, ""
        try:
            learned = await learner.learn(problem.episodes, problem.practice, writer, problem.contract)
        except Exception as exc:  # noqa: BLE001 - a candidate's failure is its score, not the tournament's
            ended, note = exc, f"{type(exc).__name__}: {exc}"
        accepted, failures = 0, []
        if learned is not None:
            for case_id in problem.cases:
                reward, reason, _ = await problem.score(learned.source, case_id)
                if reward >= 1.0:
                    accepted += 1
                else:
                    failures.append(f"{case_id}: {reason}")
        calls, usage = spent(learned or ended, writer)
        trials.append(LearnerTrial(learner.name, accepted, len(problem.cases), calls,
                                   writer_cost(usage, writer_model, prices), learned.source if learned else None,
                                   failures, note, learner))
    best = max(range(len(trials)), key=lambda i: (trials[i].accepted, -trials[i].dollars, -i))
    return LearnerChoice(trials[best], trials)
