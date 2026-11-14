"""Measure the service gap of the assistant with and without a component.

Good is every request right, with no wrong action and no language model
doing work that code could do. Each request sits some distance from good: 0
when a decision program did the work or a request that should not be acted
on went read-only, 0.4 when the language model got it right, 1 when it was
not done, and 3 for a wrong action. The service gap is the mean distance.

The held-out requests are measured three ways: Chapter 6's assistant at the
floor listing 6.1 measured, the same assistant at high reasoning, and the
assistant with the hand-written component of listing 7.1 beside it. The gate
admits the component on requests of its own before it serves any.
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
from chapters.ch07.assistant.components import hand_written_component
from chapters.ch07.assistant.harness import (
    PRICES,
    RUNS,
    decision_tier,
    harness,
    measure,
    recordings,
    run,
    save,
    serve,
    summarize,
)


async def main(args: argparse.Namespace) -> None:
    models, jev = recordings(args.record, args.budget)
    held_out = requests.load("held_out")
    log: list[dict] = []

    # The component is one candidate, and no search rewrites it.
    tier = decision_tier(jev, [("hand_written", hand_written_component())])
    agent, meta = harness(tier)
    floor = meta.controller.table["memory"]
    try:
        with models.active():
            # Chapter 6's assistant, with nothing in front of the model.
            await measure(agent, meta, held_out, floor,
                          f"model at {floor}", log)
            await measure(agent, meta, held_out, "high",
                          "model at high", log)

            # Run the component, and install what it wrote if the gate
            # admits it on requests held back from whoever wrote it.
            events: list[str] = []
            await meta.programs.improve(events)
            for event in events:
                print(f"   {event}")
            if meta.programs.current() is not None:
                await serve(agent, held_out, "hand-written component", log)
    finally:
        print()
        save(args.output, "service_gap", log, meta,
             summary=summarize(log))
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
        "--budget", type=float, default=1.0,
        help="Dollars of new model calls before the listing stops.",
    )
    parser.add_argument("--output", type=Path, default=RUNS / "service_gap")
    run(main, parser.parse_args())
