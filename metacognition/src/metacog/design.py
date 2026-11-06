"""Design learning mechanisms by search: chapter 7's step, level 3 of A5 (ARCHITECTURE.md).

A learning mechanism is source: constants, helpers, and `async def learn(lab)` (metacog.meta.sources). So
designing a better learner is a code task, and a search can do it. `design_learners` runs helix's GEPA over
mechanism source: a designer model rewrites the best seed, and each design is scored by how well the functions
it learns do on held-out cases, across several learning problems.

A design must generalize, so the search has a gate for learners too:
- it must beat the best seed on the training problems;
- it must not lose to that seed on test problems from task classes it never saw.

A design that passes is registered in a deployment's store as proposed. Once a person approves it
(`approve_mechanism`, or `metacog approve-mechanism`), the loop's learner tournaments (LearnerSelection) let it
compete with the seeds. Designed source runs under helix's mechanism guard, like the seeds. That guard catches
honest mistakes and is not a security sandbox, which is one reason a person approves a design first.

    design = await design_learners(train=[d1_problem, d2_problem], test=[d3_problem], rng=random.Random(7))
    if design.accepted:
        register_mechanism(store, design)
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from helix.archive import SQLiteArchive
from helix.artifact import ArtifactKind, genesis
from helix.memory.mechanism_runtime import MechanismBudget, UnsafeMechanism, check_mechanism
from helix.search.base import SearchBudget
from helix.search.gepa import GEPA
from helix.signal import GapMeasurement

from metacog.meta.prompts import GENERIC, PromptPack
from metacog.meta.selection import LearningProblem, run_tournament
from metacog.meta.sources import SEEDS, SourceLearner, seed_source, source_id
from metacog.model_settings import DEFAULT_MODEL

LAB = """A learning mechanism is Python source that teaches an AI assistant a procedure. Given runs the
assistant already did, each judged by a checker, it asks a writer model for a Python function the assistant
can call instead, and returns that function's source. A mechanism is scored by how well the functions it
learns do on held-out cases across several different procedures, so it must work for procedures it has never
seen, and nothing in it may assume one domain.

Its form:
- top-level constants (prompts are plain strings with {{placeholder}} fields) and helper functions;
- one entry point, `async def learn(lab)`, returning the learned function's source, or a dict with "source"
  and optional "values";
- no imports, no names that start with an underscore, no bare `except`, and no str.format; fill a template
  with `lab.fill(TEMPLATE, name=value, ...)`.

What `lab` offers, every call metered (a mechanism over budget stops and scores nothing):
- `lab.episodes`: the judged runs, each with .task_id, .request, .steps, .success and .reason;
- `lab.render(episode)`: a run as text; `lab.contract`: the function's contract as text;
- `await lab.write(prompt, effort=None)`: the writer's function for a prompt, already checked by the code
  guard (it raises lab.Refused for unsafe code); effort is "none", "low" or "medium";
- `await lab.ask(prompt, effort=None)`: the writer's text reply;
- `await lab.practice(source, task_id)`: try a function on a run's case, returning (reward, reason,
  observations);
- `lab.stream()`: the distinct task ids in order; `lab.request(task_id)`: a task's request;
- `lab.note(text)`: keep a note.

Writer calls are the cost: the budget is {writer_calls} per procedure."""

DESIGN_MUTATION_PROMPT = """You are improving a learning mechanism by reflective mutation in a
population-based search.

""" + LAB + """

You will see the current mechanism, feedback from scoring it on each training procedure (cases accepted,
writer calls, and why its functions failed), and sometimes a sibling mechanism. Produce an improved mechanism
that fixes what the feedback shows and keeps what works.

Output only the mechanism's source."""

DESIGN_CROSSOVER_PROMPT = """You are combining two learning mechanisms into one offspring.

""" + LAB + """

Take what each parent does well and drop what either does poorly.

Output only the mechanism's source."""


class _NoQuestions:
    """GEPA asks an evaluation source for questions; the problems bring their own cases."""

    label = "learning problems, each scored on its held-out cases"

    async def get_questions(self, n: int | None = None) -> list[Any]:
        return []


@dataclass
class ProblemScore:
    """How the function a mechanism learned did on one learning problem's held-out cases."""

    problem: str
    accepted: int
    total: int
    writer_calls: int
    dollars: float
    note: str = ""

    @property
    def rate(self) -> float:
        return self.accepted / self.total if self.total else 0.0


async def score_mechanism(source: str, problems: Sequence[LearningProblem], *, name: str = "candidate",
                          prompts: PromptPack | str = GENERIC, writer_model: str | None = None,
                          writer_params: dict[str, Any] | None = None, budget: MechanismBudget | None = None,
                          prices: dict[str, tuple[float, ...]] | None = None) -> list[ProblemScore]:
    """Learn each problem with the mechanism, then score the learned function on the problem's held-out
    cases. Unsafe source raises UnsafeMechanism; a mechanism that fails or runs out of budget scores 0."""
    learner = SourceLearner(source, name, prompts, budget)
    scores = []
    for problem in problems:
        choice = await run_tournament(problem, [learner], writer_model=writer_model or DEFAULT_MODEL,
                                      writer_params=writer_params, prices=prices)
        trial = choice.winner
        scores.append(ProblemScore(problem.procedure, trial.accepted, trial.total, trial.writer_calls,
                                   trial.dollars, trial.note or "; ".join(trial.failures[:3])))
    return scores


def mean_rate(scores: Sequence[ProblemScore]) -> float:
    return sum(s.rate for s in scores) / len(scores) if scores else 0.0


def _feedback(scores: Sequence[ProblemScore]) -> str:
    return "\n".join(f"- {s.problem}: {s.accepted}/{s.total} held-out cases accepted, {s.writer_calls} writer "
                     f"calls, ${s.dollars:.4f}" + (f". {s.note}" if s.note else "") for s in scores)


@dataclass
class MechanismDesign:
    """A design search's result: the best design, how it and the seeds scored, and the verdict."""

    name: str
    source: str
    train: float
    test: float | None
    best_seed: str
    seeds: dict[str, dict[str, float | None]]
    accepted: bool
    reason: str
    dollars: float
    lineage: list[dict[str, Any]] = field(default_factory=list)
    scores: dict[str, list[dict[str, Any]]] = field(default_factory=dict)


async def design_learners(train: Sequence[LearningProblem], test: Sequence[LearningProblem], *,
                          seeds: Sequence[str] = SEEDS, prompts: PromptPack | str = GENERIC,
                          designer_model: str | None = None, designer_params: dict[str, Any] | None = None,
                          writer_model: str | None = None, writer_params: dict[str, Any] | None = None,
                          budget: MechanismBudget | None = None, population: int = 4, generations: int = 3,
                          rng: random.Random | None = None, prices: dict[str, tuple[float, ...]] | None = None,
                          archive: str | Path = ":memory:") -> MechanismDesign:
    """Score the seeds, search from the best one with GEPA, and judge the best design on the test problems.

    `budget` is the regime every mechanism learns under, seeds included (default: 12 writer calls and 15
    minutes per problem). Pass a seeded `rng` so a replayed search makes the recorded choices.
    """
    if not train or not test:
        raise ValueError("design_learners needs training problems and test problems from other task classes")
    budget = budget or MechanismBudget(writer_calls=12, seconds=900)
    spend = {"dollars": 0.0}
    common = dict(prompts=prompts, writer_model=writer_model, writer_params=writer_params, budget=budget,
                  prices=prices)

    async def scored(source: str, problems: Sequence[LearningProblem], name: str) -> list[ProblemScore]:
        result = await score_mechanism(source, problems, name=name, **common)
        spend["dollars"] += sum(s.dollars for s in result)
        return result

    seed_scores = {name: await scored(seed_source(name, prompts), train, name) for name in seeds}
    best_seed = max(seeds, key=lambda n: (mean_rate(seed_scores[n]), -seeds.index(n)))

    async def evaluate(candidate: Any, questions: Any = None) -> GapMeasurement:
        source = candidate.content if hasattr(candidate, "content") else candidate
        try:
            result = await scored(source, train, "design")
        except UnsafeMechanism as exc:
            return GapMeasurement(score=0.0, signal_id="learning_acceptance", signal_version=1, triggered=True,
                                  raw_value=0.0, feedback=f"refused by the mechanism guard: {exc}", metadata={})
        rate = mean_rate(result)
        return GapMeasurement(score=rate, signal_id="learning_acceptance", signal_version=1, triggered=rate < 1.0,
                              raw_value=rate, feedback=_feedback(result),
                              metadata={"problems": [asdict(s) for s in result]})

    def validate(source: str) -> str | None:
        try:
            check_mechanism(source)
        except UnsafeMechanism as exc:
            return str(exc)
        return None

    writer_calls = budget.writer_calls if budget.writer_calls is not None else "unlimited"
    search = GEPA(proposer_model=designer_model or writer_model or DEFAULT_MODEL, eval_source=_NoQuestions(),
                  population_size=population, generations=generations,
                  mutation_prompt=DESIGN_MUTATION_PROMPT.format(writer_calls=writer_calls),
                  crossover_prompt=DESIGN_CROSSOVER_PROMPT.format(writer_calls=writer_calls),
                  evaluator=evaluate, validate=validate,
                  proposer_params=dict(designer_params or {"reasoning_effort": "medium"}), rng=rng)
    store = SQLiteArchive(archive)
    seed = genesis("learning_mechanism", ArtifactKind.CODE, seed_source(best_seed, prompts), created_by=best_seed)
    async for _ in search.propose(seed, None, store, SearchBudget(max_candidates=population * generations)):
        pass
    measured = [v for v in await store.all_artifacts_with_measurements() if v.measurement is not None]
    store.close()
    lineage = [{"version": v.artifact.version, "parent": v.artifact.parent_id[1] if v.artifact.parent_id else None,
                "created_by": v.artifact.created_by, "score": v.measurement.score or 0.0} for v in
               sorted(measured, key=lambda v: v.artifact.version)]
    best = max(measured, key=lambda v: ((v.measurement.score or 0.0), -v.artifact.version), default=None)
    source = best.artifact.content if best is not None else seed_source(best_seed, prompts)
    train_rate = (best.measurement.score or 0.0) if best is not None else mean_rate(seed_scores[best_seed])
    seed_train = mean_rate(seed_scores[best_seed])

    seed_test = mean_rate(await scored(seed_source(best_seed, prompts), test, best_seed))
    seeds_report = {name: {"train": mean_rate(seed_scores[name]), "test": seed_test if name == best_seed else None}
                    for name in seeds}
    if train_rate <= seed_train:
        return MechanismDesign(f"designed_{source_id(source)}", source, train_rate, None, best_seed, seeds_report,
                               False, f"no design beat {best_seed} on the training problems ({train_rate:.2f} vs "
                               f"{seed_train:.2f})", spend["dollars"], lineage)
    test_scores = await scored(source, test, "design")
    test_rate = mean_rate(test_scores)
    accepted = test_rate >= seed_test
    reason = (f"beats {best_seed} on training ({train_rate:.2f} vs {seed_train:.2f}) and holds on the test problems "
              f"({test_rate:.2f} vs {seed_test:.2f})" if accepted else
              f"beats {best_seed} on training ({train_rate:.2f} vs {seed_train:.2f}) but loses on the test problems "
              f"({test_rate:.2f} vs {seed_test:.2f})")
    return MechanismDesign(f"designed_{source_id(source)}", source, train_rate, test_rate, best_seed, seeds_report,
                           accepted, reason, spend["dollars"], lineage,
                           {"test": [asdict(s) for s in test_scores]})


# ------------------------------------------------------------------ the registry of designed mechanisms

def register_mechanism(store: Any, design: MechanismDesign, *, state: str = "proposed") -> str:
    """Keep a design in the deployment's store, proposed for a person to approve. Returns its name."""
    if not design.accepted:
        raise ValueError(f"{design.name} was not accepted: {design.reason}")
    store = getattr(store, "store", store)
    mechanisms = dict(store.setting("mechanisms") or {})
    mechanisms[design.name] = {"source": design.source, "state": state, "parent": design.best_seed,
                               "train": design.train, "test": design.test, "reason": design.reason,
                               "created_at": datetime.now(UTC).isoformat()}
    store.put_setting("mechanisms", mechanisms)
    return design.name


def approve_mechanism(store: Any, name: str) -> None:
    """Let the loop's learner tournaments use a designed mechanism."""
    store = getattr(store, "store", store)
    mechanisms = dict(store.setting("mechanisms") or {})
    if name not in mechanisms:
        raise KeyError(f"no designed mechanism {name!r}")
    check_mechanism(mechanisms[name]["source"])
    mechanisms[name] = {**mechanisms[name], "state": "approved"}
    store.put_setting("mechanisms", mechanisms)
