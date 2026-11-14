"""Search for a learning component, starting from one that only uses the model.

Four choices set the search. The artifact is a learning component, code
with a fixed entry point. The signal is the service gap of listing 7.2. The
method is the Darwin Gödel Machine's archive from Chapter 4, or GEPA with
--method gepa. The builder is the model that rewrites the code.

The assistant first serves a day of requests with the language model, and
those runs are the experience a component learns from. Overnight the search
measures each rewrite on requests it may read. A rewrite reaches the gate
only by clearing the bar there, and the builder never sees the gate's
requests. The held-out requests are served before and after.
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
    BUILDER,
    PRICES,
    RUNS,
    code_search,
    decision_tier,
    harness,
    measure,
    night,
    recordings,
    run,
    save,
    seed,
    serve,
    summarize,
)


def show_search(tried: list[dict]) -> None:
    """One line per rewrite the search measured or refused."""
    print(f"\n{'design':<10} {'outcome':<26} {'own requests':<13}"
          f" {'gate':<8} wrong")
    for row in tried:
        print(f"{str(row['design'])[:8]:<10} {row['outcome']:<26}"
              f" {row.get('seen', ''):<13} {row.get('right', ''):<8}"
              f" {row.get('wrong', '')}")


async def main(args: argparse.Namespace) -> None:
    models, jev = recordings(args.record, args.budget)
    args.output.mkdir(parents=True, exist_ok=True)
    archive = args.output / f"archive_{args.method}.sqlite"
    archive.unlink(missing_ok=True)  # every run starts a new archive

    # The seed keeps every request on the language model. The search
    # rewrites it with the builder, six measured rewrites at most.
    search = code_search(args.method, archive, population=args.population,
                         generations=args.generations)
    tier = decision_tier(jev, [seed("component_llm")], search=search)
    agent, meta = harness(tier)
    floor = meta.controller.table["memory"]
    held_out = requests.load("held_out")
    log: list[dict] = []
    events: list[str] = []
    print(f"Method {search.method}, builder {BUILDER},"
          f" up to {search.population * search.generations} rewrites")
    try:
        with models.active():
            await measure(agent, meta, held_out, floor,
                          f"model at {floor}", log)
            # A day of requests, served by the language model.
            day = requests.week()["Monday"]
            await serve(agent, day, "experience", log)
            # The night: the gap is open, so the search runs.
            events = await night(meta)
            if meta.programs.current() is not None:
                await serve(agent, held_out, "searched component", log)
    finally:
        for event in events:
            if not event.startswith("search: design "):
                print(f"   {event}")
        show_search(meta.programs.search_log)
        print()
        found = meta.store.setting("program_mechanisms") or []
        for component in found:
            path = args.output / f"component_{args.method}.py"
            path.write_text(component["source"], encoding="utf-8",
                            newline="\n")
        calls, dollars = models.used(BUILDER, PRICES)
        save(args.output, f"code_search_{args.method}", log, meta,
             summary=summarize(log), events=events,
             search=meta.programs.search_log,
             builder={"calls": calls, "dollars": round(dollars, 6)})
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
        "--method", default="dgm", choices=["dgm", "gepa"],
        help="The search method; the artifact and the signal stay the same.",
    )
    parser.add_argument(
        "--population", type=int, default=6,
        help="Rewrites per generation; DGM stops at the first one admitted.",
    )
    parser.add_argument(
        "--generations", type=int, default=1,
        help="Generations; GEPA needs two before it crosses two rewrites.",
    )
    parser.add_argument(
        "--budget", type=float, default=5.0,
        help="Dollars of new model calls before the listing stops.",
    )
    parser.add_argument("--output", type=Path, default=RUNS / "code_search")
    run(main, parser.parse_args())
