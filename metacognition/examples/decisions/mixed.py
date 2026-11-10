"""A desk where part of the work can lose the LLM and part can't: does the loop know which is which?

    python -m examples.decisions.mixed --mode record --budget 8
    python -m examples.decisions.mixed --domain d7 --mode record --budget 9

D5 is an on-call desk. Changing one shift needs only a name and a day, and planning a week brings new constraints each time,
in forms the gate and the held-out requests don't share with the search. D7 is a support desk. Assigning a ticket is routine,
and triaging one means weighing what its thread really establishes. The agent starts at high reasoning on everything. The same held-out requests are measured three ways before any
learning, at each reasoning setting, and once more after the loop has run:

    start       the LLM alone at each setting, frozen so it teaches nothing
    experience  the agent serves development requests at high reasoning until the cost gap opens, and a search rewrites
                an LLM-only learning component
    program     whatever the gate admitted serves the held-out requests
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from helix.deciders import recorded_decider
from helix.metacognition import Controller
from helix.metacognition.decide import JEV_MODEL
from helix.metacognition.pricing import PRICES

from examples.decisions import continuous, llm
from examples.decisions import loop as decision_loop
from examples.decisions.wordings import load
from examples.learning.study import BudgetSpent, CappedRecording, Spend

HERE = Path(__file__).parent


def kind_of(row: dict[str, Any]) -> str:
    return "near miss" if row["style"] in ("near_miss", "out_of_scope") else str(row["kind"])


def show(rows: list[dict[str, Any]]) -> None:
    """Each held-out stage by kind of request: what was right, what served it, and what a request cost."""
    for name in dict.fromkeys(r["phase"] for r in rows if r["phase"] != "experience"):
        part = [r for r in rows if r["phase"] == name]
        print(f"{name}: right {sum(r['right'] for r in part)}/{len(part)}, "
              f"${sum(r['dollars'] for r in part) / len(part):.5f} and {sum(r['seconds'] for r in part) / len(part):.2f} s")
        for kind in dict.fromkeys(kind_of(r) for r in sorted(part, key=kind_of)):
            group = [r for r in part if kind_of(r) == kind]
            if not group:
                continue
            served = Counter("program" if r["tier"] == "program" else "read-only" if r["read_only"] else
                             f"LLM at {r.get('effort')}" for r in group)
            print(f"  {kind:<9} {sum(r['right'] for r in group):>2}/{len(group):<2} "
                  f"${sum(r['dollars'] for r in group) / len(group):.5f} "
                  f"{sum(r['seconds'] for r in group) / len(group):5.2f} s  {dict(served)}")


async def main(args: argparse.Namespace) -> None:
    if args.mode != "replay":
        from dotenv import load_dotenv

        load_dotenv(Path.cwd() / ".env")
        os.environ.setdefault("TYPESAFE_API_KEY", os.environ.get("JEV_API_KEY", ""))
    args.output.mkdir(parents=True, exist_ok=True)
    jev = recorded_decider(args.jev_recording, mode=args.mode, model=JEV_MODEL)
    flags = ["--domain", args.domain, "--mode", args.mode, "--mechanism", "component_llm", "--search-method", args.method,
             "--search-target", "mechanism", "--population", str(args.population), "--generations", str(args.generations),
             "--seed", str(args.seed), "--label", args.label, "--max-wrong", "0", "--stop-when-admitted",
             "--designer-model", args.designer_model, "--writer-model", args.writer_model, "--archive", str(args.archive),
             "--output", str(args.output)]
    tier = decision_loop.tier_for(args.domain, jev, decision_loop.parse(flags))
    tier.search.lower_reasoning = args.lower_reasoning
    agent, meta, make_world = llm._setup(args.domain, llm.MODEL, programs=tier)
    meta.controller = Controller({"memory": args.start, "no_memory": args.start})

    def tried_on(task: dict[str, Any]) -> Any:
        world = make_world(task)
        world.style = task.get("style", "plain")
        return world

    tier.world = tried_on
    stream = load(args.domain, "dev+near")
    random.Random(11).shuffle(stream)
    _, held = decision_loop.split_test(args.domain)
    held_out = [*held, *load(args.domain, "stress2")]
    random.Random(12).shuffle(held_out)
    recording = CappedRecording(args.recording, mode=args.mode, spend=Spend(args.budget))
    rows: list[dict[str, Any]] = []
    try:
        with recording.active():
            for effort in dict.fromkeys((args.start, "low", "none")):
                await continuous.frozen_phase(f"start: LLM alone at {effort}", held_out, agent, meta, effort, rows,
                                              world=tried_on)
            show(rows)
            await decision_loop.phase("experience", stream, agent, meta, tried_on, rows, quiet=args.quiet)
            if meta.programs.current() is not None:
                meta.programs.paused = True  # the held-out requests measure the loop; they must not teach it
                await decision_loop.phase("program", held_out, agent, meta, tried_on, rows, quiet=args.quiet)
                meta.programs.paused = False
                efforts = {run["case_id"]: run["effort"] for run in meta.store.runs()[-len(held_out):]}
                for row in rows:
                    if row["phase"] == "program":
                        row["effort"] = efforts.get(row["id"])
            else:
                print("the search installed no program, so there is no program stage")
    finally:
        print()
        show(rows)
        (args.output / f"mixed_{args.domain}.json").write_text(json.dumps(
            {"rows": rows, "learning": meta.store.learning(decision_loop.DECISION), "signal": meta.programs.signal(),
             "search": meta.programs.search_log}, indent=1, default=str), encoding="utf-8")
        current = meta.programs.current()
        if current is not None:
            (args.output / f"mixed_{args.domain}_program.py").write_text(current["source"], encoding="utf-8", newline="\n")
            (args.output / f"mixed_{args.domain}_memories.json").write_text(json.dumps(meta.programs._stored(), indent=1),
                                                                            encoding="utf-8")
        for entry in meta.store.learning(decision_loop.DECISION):
            print(f"  learning: {entry['mechanism'][:24]:<24} {entry['outcome']:<20} ${entry['dollars'] or 0:.4f}")
        if args.mode != "replay":
            print(f"models: {recording.live_calls} live calls, ${recording.spent(PRICES):.4f}; Jev: "
                  f"{jev.live_usage().calls} live calls, ${jev.live_usage().dollars:.4f}")


def parse(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--domain", default="d5", choices=["d5", "d7"])
    parser.add_argument("--start", default="high", choices=["low", "medium", "high"], help="the reasoning it starts at")
    parser.add_argument("--designer-model", default=llm.BUILDER, help="the model that rewrites the code in a search")
    parser.add_argument("--writer-model", default=llm.BUILDER, help="the model a learning component asks to write code")
    parser.add_argument("--method", default="dgm", choices=["dgm", "gepa"])
    parser.add_argument("--population", type=int, default=6, help="rewrites a search may try, per generation")
    parser.add_argument("--lower-reasoning", action="store_true",
                        help="also admit a variant that stays on the LLM, if it matches what's running now and costs less")
    parser.add_argument("--generations", type=int, default=1)
    parser.add_argument("--seed", type=int, default=51)
    parser.add_argument("--label", default="mixed")
    parser.add_argument("--mode", default="replay", choices=["replay", "record", "live"])
    parser.add_argument("--recording", type=Path, default=HERE / "recorded" / "llm.jsonl")
    parser.add_argument("--jev-recording", type=Path, default=HERE / "recorded" / "jev.jsonl")
    parser.add_argument("--archive", type=Path, default=Path(":memory:"), help="a path keeps the search's history")
    parser.add_argument("--output", type=Path, default=HERE / "learned")
    parser.add_argument("--budget", type=float, default=8.0, help="dollars of live model calls; it stops before passing this")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    try:
        asyncio.run(main(parse()))
    except BudgetSpent as exc:
        sys.exit(f"stopped: {exc}")
