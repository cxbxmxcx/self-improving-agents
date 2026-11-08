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

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from helix.memory.learning_mechanisms import Contract, Episode, Writer
from helix.memory.mechanism_runtime import MechanismBudget

from helix.metacognition.mechanisms import Mechanism, spent, writer_cost
from helix.metacognition.parts import Practice
from helix.metacognition.sources import SEEDS

# A designed mechanism's standing after the generalization gate, best first: it beat the best seed on test
# problems from task classes it never saw; it was judged on its own problem only (or installed by a person);
# it only tied the seed on the test problems (accepted only when ties are allowed). Designs rank in this
# order, then by test and training scores, wherever they are listed or compete.
STANDINGS = {"transfers": 0, "untested": 1, "installed": 1, "ties": 2}


def design_rank(design: dict[str, Any]) -> tuple[float, float, float]:
    """Sort key for designed mechanisms, as kept in a store's "mechanisms" setting: the best first."""
    return (STANDINGS.get(design.get("standing") or "untested", 3), -(design.get("test") or 0.0),
            -(design.get("train") or 0.0))


@dataclass
class DesignPolicy:
    """When no mechanism can learn a procedure, the worker designs one (A5, level 3, in the loop).

    A tournament whose winner still fails the gate shows that no candidate learns the procedure. The worker
    (never a serving process) then runs a design search on that procedure's own learning problem. A design
    that beats the best seed is registered as proposed: a person approves it before it competes.

    tests           async () -> learning problems from other task classes, for the generalization gate; None
                    judges a design on its own problem only, and the proposal says so
    population, generations, budget (per problem), designer_model, designer_params
                    the search's settings
    fixed_effort    designs may not set the writer's effort (they must learn better, not buy a better writer)
    """

    tests: Callable[[], Awaitable[Sequence[Any]]] | None = None
    population: int = 4
    generations: int = 2
    budget: MechanismBudget | None = None
    designer_model: str | None = None
    designer_params: dict[str, Any] | None = None
    fixed_effort: bool = True


@dataclass
class LearnerSelection:
    """When the loop holds a learner tournament for a procedure, and which mechanisms compete.

    candidates          mechanism names (chapter 7's seeds run as source) or instances, such as a
                        SourceLearner a search designed
    after_refusals      learned versions refused by the gate in a row that start a tournament
    quarantined_within  a learned version quarantined within this many runs of being trusted starts
                        one too (None: never)
    designed            also let the designed mechanisms a person approved compete (helix.metacognition.design)
    design              a DesignPolicy: when a tournament's winner still fails the gate, the worker designs
                        a new mechanism (learning="worker" only; searches never run in a serving process)
    """

    candidates: Sequence[str | Mechanism] = SEEDS
    after_refusals: int = 2
    quarantined_within: int | None = 8
    designed: bool = True
    design: DesignPolicy | None = None


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
