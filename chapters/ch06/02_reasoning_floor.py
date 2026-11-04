"""Measure the reasoning floor with and without bootstrapped memory.

Run the development tasks at each selected effort setting in two arms.
The reference arm recalls the function listing 6.1's context search found,
while the baseline has no procedural memory. Compare their solid and
adequate floors with reported reasoning tokens, API cost, and model-call time.

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
    BOOTSTRAPPED_MEMORY,
    HAND_WRITTEN_MEMORY,
    PRICES,
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
# The text procedure is kept for comparison; the bootstrapped memory is
# the measure of good.
ARMS = {
    "no_memory": None,
    "bootstrapped_memory": BOOTSTRAPPED_MEMORY,
    "hand_written": HAND_WRITTEN_MEMORY,
    "text_procedure": TEXT_PROCEDURE,
}
DEFAULT_ARMS = ["no_memory", "bootstrapped_memory"]


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


async def main(args: argparse.Namespace) -> None:
    tasks = task_set(args.split)
    recording = Recording(
        args.recording, mode="record" if args.record else "replay"
    )
    # The gate is solving every task; seven in eight is reported beside it.
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
        "--record",
        action="store_true",
        help="Call gpt-5.4-mini for calls the recording lacks.",
    )
    parser.add_argument("--recording", type=Path, default=RECORDED)
    parser.add_argument("--concurrency", type=int, default=4)
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
