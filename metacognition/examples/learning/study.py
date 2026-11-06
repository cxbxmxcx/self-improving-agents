"""A5 live: learner tournaments (L2) and a learner design search (L3) on learning problems from the domains.

    python -m examples.learning.study                                   # replay the committed recordings
    python -m examples.learning.study --mode record --budget 10         # paid calls only for missing replies

Phases (choose with --phases):
  problems     build each learning problem (examples.learning.problems) and show what it holds
  tournaments  L2: chapter 7's five seeds compete on each training problem, as the loop's tournament would
  design       L3: GEPA designs a mechanism on the training problems (D1 and D2). The best design then faces
               the best seed on chapter 6's two calendar classes, which no design has seen

Every learner learns under one regime (--regime): chapter 7's "low_writer" (runs at low effort, a writer at low
effort, 12 writer calls a problem) or "weak_writer" (the same, with a writer that does no reasoning).
A design may not set the writer's effort itself: it must learn better under the regime's writer.
--budget is a hard cap on new spending across every recording this touches.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from examples.learning.problems import REGIMES, Regime, calendar_problem, expenses_problem, refunds_problem
from metacog import Recording
from metacog.design import design_learners
from metacog.meta import GENERIC, SEEDS, LearningProblem, SourceLearner, run_tournament
from metacog.pricing import PRICES

HERE = Path(__file__).parent
EXAMPLES = HERE.parent
WRITER_MODEL = "gpt-5.4-mini"
TRAIN = ("expenses", "refunds")
TEST = ("multi_attendee", "same_time")


class BudgetSpent(BaseException):
    """New spending passed the cap. A BaseException, so no learner's `except Exception` swallows it."""


class Spend:
    """One cap across several recordings."""

    def __init__(self, limit: float) -> None:
        self.limit = limit
        self.recordings: list[Recording] = []

    @property
    def total(self) -> float:
        return sum(r.spent(PRICES) for r in self.recordings)


class CappedRecording(Recording):
    """A recording that stops the study once new calls, across every capped recording, pass the budget."""

    def __init__(self, path: Path, *, mode: str, spend: Spend) -> None:
        super().__init__(path, mode=mode)
        self.spend = spend
        spend.recordings.append(self)

    async def __call__(self, **kwargs: Any) -> Any:
        reply = await super().__call__(**kwargs)
        if self.spend.total > self.spend.limit:
            raise BudgetSpent(f"${self.spend.total:.2f} of new calls is over the ${self.spend.limit:.2f} budget")
        return reply


async def build_problems(args: argparse.Namespace, spend: Spend) -> dict[str, LearningProblem]:
    regime = REGIMES[args.regime]
    with Recording(EXAMPLES / "expenses" / "recorded" / "study.jsonl").active():  # D1's runs, all recorded
        problems = {"expenses": await expenses_problem(regime)}
    refunds = CappedRecording(EXAMPLES / "refunds" / "recorded" / "study.jsonl", mode=args.mode, spend=spend)
    with refunds.active():
        problems["refunds"] = await refunds_problem(regime)
    for task_class in TEST:  # chapter 6's runs, from chapter 6's recordings
        problems[task_class] = await calendar_problem(task_class, regime)
    print(f"Learning problems, regime {args.regime}: runs at {', '.join(regime.efforts)}, writer at "
          f"{regime.writer_effort}, {regime.writer_calls} writer calls")
    for name, problem in problems.items():
        role = "train" if name in TRAIN else "test"
        print(f"  {name:<15} {role:<5} {len(problem.episodes)} runs, {sum(e.success for e in problem.episodes)} "
              f"accepted; scored on {len(problem.cases)} held-out cases")
    return problems


def seeds(regime: Regime) -> list[SourceLearner]:
    return [SourceLearner.seed(name, GENERIC, budget=regime.budget) for name in SEEDS]


async def phase_tournaments(problems: dict[str, LearningProblem], args: argparse.Namespace) -> dict[str, Any]:
    regime = REGIMES[args.regime]
    report = {}
    for name in TRAIN:
        choice = await run_tournament(problems[name], seeds(regime), writer_model=WRITER_MODEL,
                                      writer_params=regime.writer_params, prices=PRICES)
        print(f"\nTournament on {name} (held-out cases accepted, writer calls, dollars)")
        for trial in choice.trials:
            stopped = f"  stopped: {trial.note}" if trial.note else ""
            print(f"  {trial.mechanism:<24} {trial.accepted:>2}/{trial.total:<2} {trial.writer_calls:>3} "
                  f"${trial.dollars:.4f}{stopped}")
        print(f"  winner: {choice.winner.mechanism}")
        report[name] = {**choice.to_dict(), "sources": {t.mechanism: t.source for t in choice.trials}}
    (args.output / f"tournaments_{args.regime}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


async def phase_design(problems: dict[str, LearningProblem], args: argparse.Namespace) -> dict[str, Any]:
    regime = REGIMES[args.regime]
    design = await design_learners([problems[n] for n in TRAIN], [problems[n] for n in TEST], seeds=SEEDS,
                                   writer_model=WRITER_MODEL, writer_params=regime.writer_params,
                                   designer_model=WRITER_MODEL, designer_params={"reasoning_effort": "medium"},
                                   budget=regime.budget, fixed_effort=True, population=args.population,
                                   generations=args.generations, rng=random.Random(args.seed), prices=PRICES)
    print(f"\nDesign search: GEPA, population {args.population}, {args.generations} generations, from "
          f"{design.best_seed}")
    for row in design.lineage:
        print(f"  v{row['version']:<3} from {row['parent'] or '-':<5} {row['created_by']:<24} train {row['score']:.3f}")
    print("  seeds on training: " + ", ".join(f"{n} {s['train']:.3f}" for n, s in design.seeds.items()))
    test = "-" if design.test is None else f"{design.test:.3f}"
    print(f"  best design: train {design.train:.3f}, test {test}; {'ACCEPTED' if design.accepted else 'refused'}: "
          f"{design.reason}; search spent ${design.dollars:.4f}")
    (args.output / f"{design.name}.py").write_text(design.source, encoding="utf-8")
    if design.accepted and args.mode != "replay":  # a replay reproduces the kept file; it never rewrites it
        kept = HERE / "designed" / f"{design.name}.py"
        kept.parent.mkdir(exist_ok=True)
        kept.write_text(design.source, encoding="utf-8", newline="\n")
    report = {key: value for key, value in asdict(design).items() if key != "source"}
    (args.output / f"design_{args.regime}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


async def main(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    if args.mode != "replay":
        from dotenv import load_dotenv

        load_dotenv(Path.cwd() / ".env")
    spend = Spend(args.budget)
    try:
        problems = await build_problems(args, spend)
        learning = CappedRecording(args.recording, mode=args.mode, spend=spend)
        with learning.active():
            if "tournaments" in args.phases:
                await phase_tournaments(problems, args)
            if "design" in args.phases:
                await phase_design(problems, args)
    finally:
        if args.mode != "replay":
            calls = sum(r.live_calls for r in spend.recordings)
            print(f"\n{calls} live calls, ${spend.total:.4f}, saved to {args.recording} (runs: the refunds recording)")


def parse(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--phases", nargs="+", default=["tournaments", "design"], choices=["tournaments", "design"])
    parser.add_argument("--regime", default="low_writer", choices=list(REGIMES))
    parser.add_argument("--mode", default="replay", choices=["replay", "record", "live"])
    parser.add_argument("--recording", type=Path, default=HERE / "recorded" / "learning.jsonl")
    parser.add_argument("--output", type=Path, default=HERE / "runs")
    parser.add_argument("--population", type=int, default=4)
    parser.add_argument("--generations", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7, help="GEPA's random choices, so a replay matches")
    parser.add_argument("--budget", type=float, default=10.0, help="hard cap on new spending, in dollars")
    return parser.parse_args(argv)


if __name__ == "__main__":
    try:
        asyncio.run(main(parse()))
    except BudgetSpent as exc:
        sys.exit(f"stopped: {exc}")
