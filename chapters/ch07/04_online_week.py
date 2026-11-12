"""Serve a week of requests by day and improve the decision tier by night.

By day the assistant serves requests, and Chapter 6's loop keeps it right:
it records each run, quarantines a function that starts failing, raises
reasoning and relearns. By night, with nobody waiting, the decision tier
looks at the day. If the language model still does work that code could do,
the search of listing 7.3 runs. If a decision program was switched off after
wrong actions, it is tested again and switched back on only if it passes.

On Wednesday Dana rewords her note, as in listing 6.2. The function the
programs call is stale, so bookings go wrong until the loop relearns it.
From that day the gate judges on cases held back from the changed weeks,
because cases from before the change no longer describe the user's world.
While the function is quarantined, a program's route cannot take a request
below the reasoning the controller raised for it. --routes-lower-reasoning
replays the first recording, where a route could lower it.
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
from chapters.ch07.assistant.harness import (
    PRICES,
    RUNS,
    code_search,
    decision_tier,
    function_gate,
    harness,
    night,
    recordings,
    run,
    save,
    seed,
    serve,
    summarize,
)


async def main(args: argparse.Namespace) -> None:
    models, jev = recordings(args.record, args.budget)
    if args.output is None:
        name = "online_week_routes_lower" if args.routes_lower_reasoning else "online_week"
        args.output = RUNS / name
    args.output.mkdir(parents=True, exist_ok=True)
    archive = args.output / "archive.sqlite"
    archive.unlink(missing_ok=True)

    # The same seed and search as listing 7.3, so Monday night replays it.
    tier = decision_tier(jev, [seed("component_llm")],
                         search=code_search("dgm", archive),
                         keep_raised_effort=not args.routes_lower_reasoning)
    agent, meta = harness(tier)
    log: list[dict] = []
    nights: dict[str, list[str]] = {}
    try:
        with models.active():
            for day, tasks in requests.week().items():
                print(f"\n{day}: {len(tasks)} requests")
                if day == "Wednesday":
                    # Dana's note changes today, and the gate's held-back
                    # cases change with the world they describe.
                    changed = requests.load("changed_gate")
                    tier.labeled = {r["id"]: r for r in changed}
                    meta.gate = function_gate(changed=True)
                await serve(agent, tasks, day, log)
                # Nobody is asking now, so the tier may improve.
                nights[day] = await night(meta)
                for event in nights[day] or ["nothing to improve"]:
                    if not event.startswith("search: design "):
                        print(f"   night: {event}")
    finally:
        print()
        save(args.output, "online_week", log, meta, nights=nights,
             summary=summarize(log), learning=meta.store.learning(),
             search=meta.programs.search_log)
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
        "--budget", type=float, default=5.0,
        help="Dollars of new model calls before the listing stops.",
    )
    parser.add_argument(
        "--routes-lower-reasoning", action="store_true",
        help="Let a route lower the reasoning the controller raised while a"
             " function is quarantined, as the first recording did.",
    )
    parser.add_argument("--output", type=Path, default=None)
    run(main, parser.parse_args())
