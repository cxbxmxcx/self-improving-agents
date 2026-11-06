"""The expense study: metacog's second task class, end to end on the SDK (HARNESS_PLAN.md M5).

    python -m examples.expenses.study                                   # replay the committed recording
    python -m examples.expenses.study --mode record --budget 3          # paid calls only for missing replies

Phases (choose with --phases):
  floors   the no-memory reasoning floors on the development trips (the study's E1 baseline)
  learn    value learning with the domain-neutral prompts, from the medium and high development runs
           (replayed from the floors phase), checked on the validation trips
  memory   the floors with a procedure: the learned one, or the hand-written one with --memory reference
  online   the online loop through the policy change: 6 test trips, then 14 drift trips

Every model call goes through a metacog Recording, so a recorded study replays without an API key.
--repeats N runs each case N times (one at a time, each repeat its own reply) to see variance.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from helix.memory.learning_mechanisms import Writer

from examples.expenses.agent import MODEL, REFERENCE, assistant, check, practice, reimbursable_total, run_on_case
from examples.expenses.world import ExpenseWorld, task_set
from metacog import ModelSettings, Reasoning, Recording, RunConfig, Runner
from metacog.meta import GENERIC, Controller, Gate, Metacognition, ValueLearning
from metacog.meta.mechanisms import writer_cost
from metacog.meta.report import render
from metacog.offline import Case, Floors, calibrate
from metacog.pricing import PRICES

HERE = Path(__file__).parent
EFFORTS = ("none", "low", "medium", "high")


class BudgetSpent(Exception):
    pass


def guard(recording: Recording, budget: float) -> None:
    spent = recording.spent(PRICES)
    if spent > budget:
        raise BudgetSpent(f"${spent:.2f} of live calls is over the ${budget:.2f} budget")


def show(floors: Floors, title: str) -> None:
    print(f"\n{title}")
    print(f"  {'arm':<10} {'effort':<7} {'solved':>7} {'reasoning':>9} {'dollars':>8} {'seconds':>7}")
    for row in floors.rows():
        print(f"  {row['arm']:<10} {row['effort']:<7} {row['solved']:>3}/{row['total']:<3} "
              f"{row['reasoning_tokens']:>9.0f} {row['dollars']:>8.4f} {row['seconds']:>7.1f}")
    for arm in floors.sweeps:
        solid, adequate = floors.floor(arm) or "not reached", floors.floor(arm, 7 / 8) or "not reached"
        print(f"  {arm}: solid floor {solid}, adequate {adequate}")


def cases(split: str, n: int) -> list[Case]:
    return [Case(t["id"], t["text"], t) for t in task_set(split)[:n]]


async def phase_floors(args, recording) -> Floors:
    floors = await calibrate(assistant(), cases("dev", args.tasks), verifier=check, world=ExpenseWorld,
                             efforts=args.efforts, arms=("no_memory",), repeats=args.repeats,
                             concurrency=1 if args.repeats > 1 else args.concurrency, max_turns=12,
                             run_config=RunConfig(max_tool_calls=24), prices=PRICES)
    show(floors, "Reasoning floors without memory (development trips)")
    floors.save(args.output / "floors_no_memory.json")
    guard(recording, args.budget)
    return floors


async def phase_learn(args, recording, learner: Any = None) -> str:
    meta = Metacognition([reimbursable_total], verifier=check, learning="inline", relearn_after=10**9, recall_k=1)
    agent = assistant(metacognition=meta)
    for task in task_set("dev")[: args.tasks]:
        for effort in ("medium", "high"):
            config = RunConfig(model_settings=ModelSettings(reasoning=Reasoning(effort)), max_tool_calls=24,
                               case_id=task["id"], case_data=task)
            await Runner.run(agent, task["text"], context=ExpenseWorld(task), max_turns=12, run_config=config)
            guard(recording, args.budget)
    episodes = meta.store.episodes()
    print(f"\nExperience: {len(episodes)} runs, {sum(e.success for e in episodes)} accepted")
    writer = Writer(model=MODEL, params={"reasoning_effort": "medium"})
    learner = learner or ValueLearning(GENERIC)
    learned = await learner.learn(episodes, practice, writer, reimbursable_total.contract)
    validation = task_set("validation")[: args.tasks]
    accepted = [(await practice(learned.source, t["id"]))[0] for t in validation]
    reference = [await run_on_case(REFERENCE, t) for t in validation]
    outputs = [await run_on_case(learned.source, t) for t in validation]
    matched = sum(out == ref for out, ref in zip(outputs, reference, strict=True))
    spent = writer_cost(learned.usage or writer.usage, MODEL, PRICES)
    print(f"Learned by {learned.mechanism.replace('_', ' ')}: {learned.writer_calls} writer calls, "
          f"${spent:.4f}; validation {sum(accepted):.0f}/{len(validation)} "
          f"accepted, {matched}/{len(validation)} match the hand-written reference")
    (args.output / "learned_reimbursable_total.py").write_text(learned.source, encoding="utf-8")
    return learned.source


def procedure_source(args) -> tuple[str, str]:
    if args.memory == "reference":
        return REFERENCE, "human"
    path = args.output / "learned_reimbursable_total.py"
    if not path.exists():
        raise SystemExit("no learned procedure yet: run the learn phase first, or pass --memory reference")
    return path.read_text(encoding="utf-8"), "value_learning"


async def phase_memory(args, recording) -> Floors:
    source, by = procedure_source(args)
    meta = Metacognition([reimbursable_total], verifier=check, recall_k=1, learning="inline")
    meta.install("reimbursable_total", source, created_by=by)
    split = "dev" if args.memory == "reference" else "test"
    floors = await calibrate(assistant(metacognition=meta), cases(split, args.tasks), verifier=check,
                             world=ExpenseWorld, efforts=args.efforts, arms=("memory",), repeats=args.repeats,
                             concurrency=1 if args.repeats > 1 else args.concurrency, max_turns=12,
                             run_config=RunConfig(max_tool_calls=24), prices=PRICES)
    show(floors, f"Reasoning floors with the {'hand-written' if by == 'human' else 'learned'} procedure "
                 f"({split} trips)")
    floors.save(args.output / f"floors_memory_{args.memory}.json")
    guard(recording, args.budget)
    return floors


async def phase_online(args, recording) -> None:
    source, by = procedure_source(args)
    # The study's settings until this model is calibrated; measured floors, kept in the store, come first.
    meta = Metacognition([reimbursable_total], verifier=check, practice=practice,
                         gate=Gate(cases=[t["id"] for t in task_set("gate")]),
                         controller=Controller({"memory": "none", "no_memory": "high"}), recall_k=1, relearn_after=4,
                         learning="inline")
    for name in ("floors_no_memory.json", f"floors_memory_{args.memory}.json"):
        if (args.output / name).exists():
            floors = Floors.load(args.output / name)
            floors.model = floors.model or MODEL  # floors saved before they named their model
            floors.record(meta)
    meta.install("reimbursable_total", source, created_by=by)
    agent = assistant(metacognition=meta)
    with_memory, without = (meta.setting("reimbursable_total", trusted, MODEL) for trusted in (True, False))
    print(f"\nOnline loop: {with_memory} with a trusted procedure, {without} without")
    print(f"{'#':>3} {'case':<5} {'offered':<9} {'effort':<7} verdict")
    stream = task_set("test")[:6] + task_set("stream")[:14]
    for n, task in enumerate(stream, 1):
        result = await Runner.run(agent, task["text"], context=ExpenseWorld(task), max_turns=12,
                                  run_config=RunConfig(max_tool_calls=24, case_id=task["id"], case_data=task))
        report = result.metacognition
        verdict = "accepted" if report.verdict.accepted else f"rejected: {report.verdict.reason}"
        print(f"{n:>3} {task['id']:<5} {'yes' if report.offered else 'no':<9} {report.effort or '-':<7} {verdict}")
        for event in report.events:
            print(f"      {event}")
        guard(recording, args.budget)
    print("\n" + render(meta.summary()))
    (args.output / "online_summary.json").write_text(json.dumps(meta.summary(), indent=2, default=str),
                                                     encoding="utf-8")


async def main(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    recording = Recording(args.recording, mode=args.mode, distinct_repeats=args.repeats > 1)
    if args.mode != "replay":
        from dotenv import load_dotenv

        load_dotenv(Path.cwd() / ".env")
    try:
        with recording.active():
            for phase in args.phases:
                await {"floors": phase_floors, "learn": phase_learn, "memory": phase_memory,
                       "online": phase_online}[phase](args, recording)
    finally:
        if args.mode != "replay":
            print(f"\n{recording.live_calls} live calls, ${recording.spent(PRICES):.4f}, saved to {recording.path}")


def parse(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--phases", nargs="+", default=["floors", "learn", "memory", "online"],
                        choices=["floors", "learn", "memory", "online"])
    parser.add_argument("--mode", default="replay", choices=["replay", "record", "live"])
    parser.add_argument("--recording", type=Path, default=HERE / "recorded" / "study.jsonl")
    parser.add_argument("--output", type=Path, default=HERE / "runs")
    parser.add_argument("--tasks", type=int, default=16, help="cases per split")
    parser.add_argument("--efforts", nargs="+", default=list(EFFORTS), choices=list(EFFORTS))
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--memory", default="learned", choices=["learned", "reference"])
    parser.add_argument("--budget", type=float, default=1.0, help="dollars of live calls before stopping")
    return parser.parse_args(argv)


if __name__ == "__main__":
    try:
        asyncio.run(main(parse()))
    except BudgetSpent as exc:
        sys.exit(f"stopped: {exc}")
    except KeyError as exc:
        sys.exit(f"{exc.args[0]}")
