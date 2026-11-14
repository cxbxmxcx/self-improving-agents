"""A live check of the wiring before the calendar desk's paid runs: a few cents, and none of the stream's requests.

    python -m examples.scale.desk_check --record --budget 0.5

It answers two questions that a scripted soak cannot. Do the keys, the agent's model and the decision model work
through the same code the runs use? And does the decision model send a request to the right function, when the two
kinds of booking are worded by the stream's own writers? The weeks and words here come from seeds the stream never
uses, so nothing the test measures is seen.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections import Counter
from pathlib import Path
from typing import Any

from chapters.ch07.assistant.desk import FIND_COMMON_STARTS, FIND_SHARED_STARTS
from helix.deciders.base import Choice, ask_measured
from helix.metacognition.recording import BudgetSpent

from examples.scale import desk_run, desk_stream

HERE = Path(__file__).parent
FIRST_SEED = 700_000  # the stream's own seeds end below this


def labeled(stream: desk_stream.Stream, count: int) -> list[tuple[str, str]]:
    """Requests of both kinds of booking in every writer's words, each with the function it needs."""
    out = []
    for n in range(count):
        week = stream.weeks.get(FIRST_SEED + n, "one", desk_stream.FIFTEEN, False)
        group = desk_stream.GROUPS[n % len(desk_stream.GROUPS)]
        bank = stream.banks[group]
        filled = desk_stream.fields(week, bank)
        turn = n // len(desk_stream.GROUPS)
        out.append((bank["single"][turn % 12].format(**filled), FIND_SHARED_STARTS.name))
        out.append((bank["double"][turn % 6].format(**filled), FIND_SHARED_STARTS.name))
        out.append((bank["same_time"][turn % 6].format(**filled), FIND_COMMON_STARTS.name))
        out.append((bank["same_time"][(turn + 3) % 6].format(**filled), FIND_COMMON_STARTS.name))
    return out


async def main(args: argparse.Namespace) -> dict[str, Any]:
    stream = desk_stream.Stream(weeks=desk_stream.Weeks(args.weeks))
    transports = desk_run.Transports(args, "check")
    models, jev = transports.open("check")
    out: dict[str, Any] = {"stopped": ""}
    try:
        # Which function does a request need? The same question the loop asks once it has two.
        descriptions = {p.name: p.description for p in (FIND_SHARED_STARTS, FIND_COMMON_STARTS)}
        question = Choice("Which of these does `request` need?", dict(descriptions))
        tally: Counter[tuple[str, str]] = Counter()
        missed = []
        for text, needs in labeled(stream, args.requests):
            answers, _ = await ask_measured(jev, {"request": text}, {"procedure": question})
            chosen = str(answers["procedure"].value)
            tally[needs, chosen] += 1
            if chosen != needs:
                missed.append({"needs": needs, "chosen": chosen, "request": text})
        right = sum(count for (needs, chosen), count in tally.items() if needs == chosen)
        by_kind = {f"{needs} -> {chosen}": count for (needs, chosen), count in sorted(tally.items())}
        out["routing"] = {"right": right, "requests": sum(tally.values()), "by_kind": by_kind, "missed": missed[:12]}
        print(f"routing: {right}/{sum(tally.values())} requests sent to the function they need; "
              f"{out['routing']['by_kind']}", flush=True)
        # A few requests through the agent with Chapter 6's loop, as the runs serve them.
        agent, _ = desk_run.build("loop", stream, jev, args, transports.folder)
        kinds = ["one", "two", "near", "out"]
        stream.exam()  # the stream's own requests are worded first, so these take none of their words
        specs = stream._specs(kinds, stream.days[0], FIRST_SEED + 500, ("g1", "g2", "g3", "g4"))
        tasks = [stream._request(spec, f"check-{n}", n) for n, spec in enumerate(specs)]
        with models.active():
            rows = [await desk_run.serve_one("check", "check", agent, {**task, "day": 0, "seq": n})
                    for n, task in enumerate(tasks)]
        out["served"] = [{k: row[k] for k in ("asked", "right", "outcome", "effort", "dollars", "seconds", "failed")}
                         for row in rows]
        for row in out["served"]:
            print(f"   {row['asked']:<5} {'right' if row['right'] else 'WRONG':<6} {row['outcome']:<10} "
                  f"effort {row['effort']}, ${row['dollars']:.5f}, {row['seconds']:.2f} s {row['failed']}", flush=True)
    except BudgetSpent as exc:
        out["stopped"] = f"budget: {exc}"
        print(f"Stopped: {exc}")
    out["spent"] = transports.spent()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "check.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    return out


def parse(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--record", action="store_true", help="call the models for replies the recordings lack")
    parser.add_argument("--budget", type=float, default=0.5)
    parser.add_argument("--requests", type=int, default=18, help="weeks to word four ways for the routing check")
    parser.add_argument("--weeks", type=Path, default=desk_stream.CACHE)
    parser.add_argument("--recorded", type=Path, default=HERE / "runs" / "desk_calendar" / "recorded")
    parser.add_argument("--output", type=Path, default=HERE / "runs" / "desk_calendar" / "results")
    args = parser.parse_args(argv)
    args.soak = False
    return args


if __name__ == "__main__":
    asyncio.run(main(parse()))
