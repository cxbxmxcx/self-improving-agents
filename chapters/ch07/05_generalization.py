"""Apply the component the search found to a second kind of scheduling work.

Listing 7.3's search wrote a learning component from requests for one
meeting. Here the assistant also takes a second class from Chapter 6's code:
the same meeting at the same time on several days. Each of those requests needs one
booking per day, and a decision program takes one action, so that work
cannot leave the language model. What a component can still decide is the
reasoning level it runs at and the memory it reads first.

No search runs. The found component is the assistant's one candidate. It
runs overnight on a day of both kinds of request, with test requests of both
kinds to practice on, and the gate judges what it wrote on held-back
requests of both kinds. The held-out requests then show what serves each
kind of work, and where reasoning stays.
Replay uses saved replies; --record adds paid calls up to --budget.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chapters.ch07.assistant import requests
from chapters.ch07.assistant.desk import contract
from chapters.ch07.assistant.harness import (
    PRICES,
    RUNS,
    decision_tier,
    harness,
    measure,
    night,
    recordings,
    run,
    save,
    serve,
    summarize,
)


async def main(args: argparse.Namespace) -> None:
    if not args.component.exists():
        sys.exit(f"{args.component} is missing; run listing 7.3 first.")
    found = args.component.read_text(encoding="utf-8")
    models, jev = recordings(args.record, args.budget)

    # The code listing 7.3 found, unchanged, with no search behind it.
    # The gate and the job description now cover both kinds of request.
    tier = decision_tier(jev, [("found_component", found)],
                         seen="second_search", gate="second_gate",
                         contract=contract(second_class=True))
    agent, meta = harness(tier, second_class=True)
    floor = meta.controller.table["memory"]
    held_out = requests.load("second_held_out")
    log: list[dict] = []
    events: list[str] = []
    try:
        with models.active():
            await measure(agent, meta, held_out, floor,
                          f"model at {floor}", log)
            day = requests.load("second_experience")
            await serve(agent, day, "experience", log)
            # Overnight the component learns from the day it has seen.
            events = await night(meta)
            if meta.programs.current() is not None:
                await serve(agent, held_out, "found component", log)
    finally:
        for event in events:
            print(f"   {event}")
        print()
        save(args.output, "generalization", log, meta,
             summary=summarize(log), events=events)
        if args.record:
            usage = jev.live_usage()
            print(f"\n{models.live_calls} model calls,"
                  f" ${models.spent(PRICES):.2f};"
                  f" {usage.calls} Jev calls, ${usage.dollars:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--record", action="store_true",
        help="Call the models for replies the recordings lack.",
    )
    parser.add_argument(
        "--component", type=Path,
        default=RUNS / "code_search" / "component_dgm.py",
        help="The learning component listing 7.3 saved.",
    )
    parser.add_argument(
        "--budget", type=float, default=3.0,
        help="Dollars of new model calls before the listing stops.",
    )
    parser.add_argument(
        "--output", type=Path, default=RUNS / "generalization"
    )
    run(main, parser.parse_args())
