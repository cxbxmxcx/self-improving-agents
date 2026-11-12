"""Measure the reasoning floor and consistency with and without the starting memory.

Run the development tasks at each selected effort setting in two arms. The
memory arm recalls the starting memory, the function chapter 4's context
search found, while the baseline has no procedural memory. Compare their
solid and adequate floors with reported reasoning tokens, API cost, and
model-call time.

With --repeats, serve every task again at the setting the controller would
pick for each arm and count how often the same request gets the same
booking and verdict. The first run replays the sweep's own reply; each
repeat after it is kept in a recording of its own, because a recording
replays one reply per request.

Replay uses saved replies and requires no API key. The --record option adds
paid calls only when a reply is missing. Add --arms text_procedure to measure
the optional instruction-based memory, which may require a new recording.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chapters.ch06.assistant.generator import task_set
from chapters.ch06.assistant.harness import (
    HAND_WRITTEN_MEMORY,
    PRICES,
    STARTING_MEMORY,
    TEXT_PROCEDURE,
    install_memory,
    run_task,
)
from chapters.ch06.assistant.recording import Recording
from helix.eval.effort import (
    ADEQUATE,
    EFFORTS,
    SOLID,
    ReasoningFloorSignal,
    sweep,
)

HERE = Path(__file__).parent
RECORDED = HERE / "assistant" / "recorded" / "reasoning_floor.jsonl"
# The text procedure is kept for comparison; the starting memory is the
# memory the loop begins with.
ARMS = {
    "no_memory": None,
    "starting_memory": STARTING_MEMORY,
    "hand_written": HAND_WRITTEN_MEMORY,
    "text_procedure": TEXT_PROCEDURE,
}
DEFAULT_ARMS = ["no_memory", "starting_memory"]
# The most one run is expected to cost at a setting, for --budget.
RUN_MARGIN = {"none": 0.02, "low": 0.04, "medium": 0.06, "high": 0.10}


class BudgetSpent(Exception):
    """The next run could cross the budget for live calls."""


async def measure(
    arm: str, tasks: list[dict], efforts: list[str], concurrency: int
):
    memory = await install_memory(ARMS[arm])

    async def run(task: dict, effort: str):
        return await run_task(task, effort, memory=memory)

    return await sweep(
        arm,
        tasks,
        run,
        efforts=efforts,
        concurrency=concurrency,
        prices=PRICES,
    )


def report(result) -> None:
    print(f"\n{result.arm}")
    print(
        f"  {'effort':<7} {'solved':>6} {'reasoning':>9} {'dollars':>8}"
        f" {'seconds':>7}"
    )
    for row in result.rows():
        print(
            f"  {row['effort']:<7} {row['solved']:>3}/{row['total']:<2}"
            f" {row['reasoning_tokens']:>9.0f} {row['dollars']:>8.4f}"
            f" {row['seconds']:>7.1f}"
        )
    print(
        "  floor: solid (every task)"
        f" {result.floor(SOLID) or 'not reached'}, adequate (7 in 8)"
        f" {result.floor(ADEQUATE) or 'not reached'}"
    )


def controller_setting(result) -> str:
    """The setting the online controller picks for an arm: its adequate floor, else its best record."""
    measured = [e for e in EFFORTS if e in result.cells]
    return result.floor(ADEQUATE) or max(
        measured, key=lambda e: (result.cells[e].rate, EFFORTS.index(e))
    )


async def consistency(
    arm: str, tasks: list[dict], effort: str, args: argparse.Namespace,
    spent: list[float],
) -> dict:
    """Serve each task args.repeats times at one setting; compare the runs."""
    memory = await install_memory(ARMS[arm])
    runs = {task["id"]: [] for task in tasks}
    for repeat in range(1, args.repeats + 1):
        path = args.recording if repeat == 1 else args.recording.with_name(
            f"{args.recording.stem}.repeat{repeat}.jsonl"
        )
        # The first run is the sweep's own and is always replayed.
        recording = Recording(
            path, mode="record" if args.record and repeat > 1 else "replay"
        )
        with recording.active():
            for task in tasks:
                if args.record and (
                    sum(spent) + recording.spent(PRICES) + RUN_MARGIN[effort]
                    > args.budget
                ):
                    spent.append(recording.spent(PRICES))
                    raise BudgetSpent(
                        f"${sum(spent):.4f} of live calls spent; the next run"
                        f" could cross the ${args.budget:.2f} budget"
                    )
                try:
                    solved, _, trajectory = await run_task(
                        task, effort, memory=memory
                    )
                    booked = tuple(trajectory.metadata["booked"])
                except KeyError:
                    raise  # a reply the recording lacks
                except Exception as exc:  # noqa: BLE001 - a crashed run counts as rejected
                    solved, booked = False, (f"crashed: {type(exc).__name__}",)
                runs[task["id"]].append((solved, booked))
        spent.append(recording.spent(PRICES))
    return {
        "arm": arm,
        "effort": effort,
        "repeats": args.repeats,
        "tasks": len(tasks),
        "same_booking": sum(len(set(r)) == 1 for r in runs.values()),
        "accepted_every_time": sum(
            all(ok for ok, _ in r) for r in runs.values()
        ),
        "accepted_runs": sum(ok for r in runs.values() for ok, _ in r),
        "runs": {task_id: [[ok, list(booked)] for ok, booked in r]
                 for task_id, r in runs.items()},
    }


async def main(args: argparse.Namespace) -> None:
    tasks = task_set(args.split)
    recording = Recording(
        args.recording, mode="record" if args.record else "replay"
    )
    # The signal's bar is every task solved; seven in eight is reported beside it.
    signal = ReasoningFloorSignal(target="low", threshold=SOLID)
    results = []
    with recording.active():
        for arm in args.arms:
            result = await measure(
                arm, tasks, args.efforts, args.concurrency
            )
            results.append(result)
            report(result)
            gap = signal.measure_sweep(result)
            print(f"  signal: score {gap.score:.2f}, {gap.feedback}")
    if args.record:
        print(
            f"\n{recording.live_calls} live model calls added to"
            f" {recording.path.name}"
        )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps([r.to_dict() for r in results], indent=2),
            encoding="utf-8",
        )
    if args.repeats > 1:
        print(f"\nConsistency over {args.repeats} runs of each request,"
              " at the setting the controller picks")
        spent: list[float] = [recording.spent(PRICES)]
        rows = []
        try:
            for result in results:
                effort = controller_setting(result)
                row = await consistency(result.arm, tasks, effort, args, spent)
                rows.append(row)
                print(
                    f"  {row['arm']:<16} at {effort:<6} same booking"
                    f" {row['same_booking']:>2}/{row['tasks']},"
                    f" accepted every time"
                    f" {row['accepted_every_time']:>2}/{row['tasks']}"
                    f" ({row['accepted_runs']} of"
                    f" {row['repeats'] * row['tasks']} runs)"
                )
        except BudgetSpent as exc:
            print(f"  stopped: {exc}")
        if args.record:
            print(f"  live calls cost ${sum(spent):.4f}")
        out = args.output.with_name("consistency.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--split", default="dev", choices=("dev", "validation", "test")
    )
    parser.add_argument(
        "--arms", nargs="+", default=DEFAULT_ARMS, choices=list(ARMS)
    )
    parser.add_argument(
        "--efforts",
        nargs="+",
        default=list(EFFORTS),
        choices=list(EFFORTS),
    )
    parser.add_argument(
        "--repeats", type=int, default=1,
        help="Runs of each request for the consistency check (1 skips it).",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="Call gpt-5.4-mini for calls the recording lacks.",
    )
    parser.add_argument("--recording", type=Path, default=RECORDED)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument(
        "--budget", type=float, default=0.75,
        help="Dollars of live calls the consistency runs may add.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=HERE / "runs" / "reasoning_floor.json",
    )
    args = parser.parse_args()
    if args.record:
        from helix.env import load_env

        load_env()
    try:
        asyncio.run(main(args))
    except KeyError as exc:
        sys.exit(
            f"{exc.args[0]}. The recording has no runs for these settings"
            " yet; record them with --record (paid calls to"
            " gpt-5.4-mini)."
        )
