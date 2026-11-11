"""Serve requests with a learning component written by hand.

A learning component stores memories and returns a router. The router picks
the memory that fits a request, and the memory's route says what serves it:
a decision program with no language model, the agent at a reasoning level, or
the agent read-only. This one was written by hand from the development
requests, as the reference a search has to match.

No language model runs here. The router and the program ask Jev, a small
model that returns typed judgments, and code does the rest. A request the
program cannot finish is counted as sent to the agent. Replay uses saved
answers, and --record asks Jev for the answers the recording lacks.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chapters.ch07.assistant import requests
from chapters.ch07.assistant.components import (
    hand_written_program,
    memories,
)
from chapters.ch07.assistant.desk import desk
from chapters.ch07.assistant.harness import (
    kept_procedure,
    kind_of,
    recordings,
    run,
)
from helix.metacognition.decide import check_program, serve


def route_of(memory: dict) -> str:
    if memory["route"] == "program":
        return "decision program"
    level = f"agent at {memory['route']}"
    return f"{level}, read-only" if memory["read_only"] else level


def what_happened(served) -> str:
    if served.handled:
        return "booked right" if served.right else "booked wrong"
    if served.read_only:
        return "to the agent, read-only"
    return f"to the agent at {served.route}"


def show_one(task: dict, served) -> None:
    """One request from words to booking: what Jev judged, and the cost."""
    print(f"\nOne request: {task['text']}")
    for question, answer in served.answers.items():
        value = answer.get("choice") or f"{answer.get('noul', 0):.2f}"
        print(f"   Jev, {question:<8} {value}")
    print(f"   memory {served.served_by}: {what_happened(served)}"
          f" ({served.reason})")
    print(f"   {served.decider_calls} Jev calls,"
          f" {served.decider_input_tokens} tokens,"
          f" ${served.dollars:.6f}, {served.seconds:.2f} s")


async def main(args: argparse.Namespace) -> None:
    _, jev = recordings(args.record, budget=0.0)
    program = hand_written_program()
    check_program(program)  # the guard every decision program passes
    print("Memories of the hand-written component")
    for memory in memories():
        print(f"   {memory['name']:<17} {route_of(memory):<22}"
              f" {memory['description']}")

    procedure, _ = kept_procedure()
    the_desk = desk(procedure)
    outcomes, shown = Counter(), False
    cost = seconds = 0.0
    tasks = requests.load(args.requests)
    for task in tasks:
        served = await serve(program, task, the_desk, jev)
        outcomes[(kind_of(task), what_happened(served))] += 1
        cost, seconds = cost + served.dollars, seconds + served.seconds
        if served.handled and served.right and not shown:
            show_one(task, served)
            shown = True

    print(f"\n{len(tasks)} {args.requests} requests")
    for (kind, outcome), count in sorted(outcomes.items()):
        print(f"   {kind:<12} {outcome:<24} {count:>3}")
    print(f"   ${cost / len(tasks):.6f} and {seconds / len(tasks):.2f} s"
          " a request, with no language model")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--record", action="store_true",
        help="Ask Jev for the answers the recording lacks.",
    )
    parser.add_argument(
        "--requests", default="experience",
        choices=["experience", "search"],
        help="The set to serve; the gate's and the held-out sets stay unread.",
    )
    run(main, parser.parse_args())
