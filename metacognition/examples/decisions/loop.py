"""The decision loop, live: signal, gap, learn, gate, search, and a per-task switch between a program and the LLM.

    python -m examples.decisions.loop --domain d3 --mode record --budget 5

The agent starts with its trusted procedures and no program, and the only mechanism it has is program synthesis,
which says nothing about Jev. The stream has three parts:

    experience   the development wordings and near misses, served by the LLM, until the gap opens
    held out     wordings and near misses no part of the loop has seen, served by whatever the loop installed
    drift        (D3) a vendor changes its invoice layout, the parser goes stale, intake is switched off and
                 payments stay on the program; a corrected parser arrives and the loop relearns and switches it back

A search sees half the gate's wordings and the first near-miss set. The gate holds the rest of the wordings and a
near-miss set of its own, so a program that only knows the words a search showed it can't pass. The held-out stream is
the other half of the test wordings and the second near-miss set, worded apart from both.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
from pathlib import Path
from typing import Any

import helix
from helix.deciders import Decider, recorded_decider
from helix.metacognition.decide import JEV_MODEL
from helix.metacognition.decisions import DECISION, RUNG, ProgramSearch, ProgramTier
from helix.metacognition.pricing import PRICES

from examples.decisions import llm, study
from examples.decisions.learn import contract
from examples.decisions.wordings import load
from examples.learning.study import BudgetSpent, CappedRecording, Spend
from metacog import RunConfig, Runner

HERE = Path(__file__).parent
SEEDS = Path(helix.__file__).parent / "metacognition" / "programs"
METHODS = ("dgm", "gepa")


def split_test(domain: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Test requests by base request: the first half for the gate, the second for the held-out stream."""
    tasks = load(domain, "test")
    bases = list(dict.fromkeys(t["id"].split("~")[0] for t in tasks))
    first = set(bases[: len(bases) // 2])
    return [t for t in tasks if t["id"].split("~")[0] in first], [t for t in tasks if t["id"].split("~")[0] not in first]


def split_near(tasks: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Near misses by base request, alternating within each kind of work, so both sides hold every kind of near miss."""
    of_kind: dict[str, list[str]] = {}
    for t in tasks:
        bases = of_kind.setdefault(t.get("kind", ""), [])
        if t["id"].split("~")[0] not in bases:
            bases.append(t["id"].split("~")[0])
    first = {base for bases in of_kind.values() for base in bases[::2]}
    return [t for t in tasks if t["id"].split("~")[0] in first], [t for t in tasks if t["id"].split("~")[0] not in first]


def tier_for(domain: str, jev: Decider, args: argparse.Namespace) -> ProgramTier:
    desk, _, _ = study.domain(domain)
    gate, _ = split_test(domain)
    # A search and the gate must not share cases, or a search grades its own exam: half the bases are the search's to
    # see and the rest stay hidden behind the gate. The search sees every kind of near miss; the gate asks the same
    # kinds about other invoices in words of its own. The held-out stream's near misses are worded apart from both.
    bases = list(dict.fromkeys(t["id"].split("~")[0] for t in gate))
    for_search = set(bases[: len(bases) // 2])
    near_seen, _ = split_near(load(domain, "stress"))
    _, near_hidden = split_near(load(domain, "gate"))
    seen = [*[t for t in gate if t["id"].split("~")[0] in for_search], *near_seen]
    hidden = [*[t for t in gate if t["id"].split("~")[0] not in for_search], *near_hidden]
    labeled = {t["id"]: t for t in hidden}
    source = (SEEDS / f"{args.mechanism}.py").read_text(encoding="utf-8")
    search = ProgramSearch(method=args.search_method, target=args.search_target, archive=args.archive, population=args.population,
                           designer_model=args.designer_model, designer_params={"reasoning_effort": args.designer_effort},
                           share_all=args.share_all, stop_when_admitted=args.stop_when_admitted,
                           generations=args.generations, seed=args.seed, label=args.label) if args.search else None
    return ProgramTier(desk=desk, decider=jev, contract=contract(domain, routes=args.routes), labeled=labeled, mechanisms=[(args.mechanism, source)],
                       search=search, search_cases={t["id"]: t for t in seen}, min_runs=args.min_runs,
                       max_wrong=args.max_wrong, bar=args.bar, routed_bar=args.routed_bar, read_only_handoffs=args.routes,
                       writer_model=args.writer_model, retry_after=12, max_attempts=args.max_attempts, gating_sources=("verifier",))


async def phase(name: str, tasks: list[dict[str, Any]], agent: Any, meta: Any, world_of: Any, rows: list[dict[str, Any]],
                *, quiet: bool, on_request: Any = None) -> None:
    for n, task in enumerate(tasks):
        if on_request is not None:
            on_request(n, meta)
        world = world_of(task)
        world.style = task["style"]
        result = await Runner.run(agent, task["text"], context=world, max_turns=12,
                                  run_config=RunConfig(max_tool_calls=24, case_id=task["id"], case_data=task))
        report = result.metacognition
        served = report.effort == RUNG
        handed = any(e.startswith("program: handed off") for e in report.events)
        routed = any(e.startswith("program: routed") for e in report.events)
        read_only = any(e.startswith("program: handed off read-only") for e in report.events)
        memory = next((e.split("program: used memory ", 1)[1] for e in report.events if e.startswith("program: used memory ")), None)
        rows.append({"phase": name, "id": task["id"], "style": task["style"], "kind": task.get("kind"),
                     "tier": "program" if served else "llm", "handed_off": handed, "read_only": read_only, "routed": routed, "memory": memory,
                     "right": bool(report.verdict.accepted), "reply": str(result.final_output)[:400] if read_only else None,
                     "dollars": float(result.cost.dollars or 0.0), "seconds": float(result.cost.seconds or 0.0)})
        for event in report.events:
            if not event.startswith(("program: handed off", "program: routed", "program: used memory")) and not quiet:
                print(f"  [{name} #{n + 1} {task['id']}] {event}")


def report(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out = {}
    for name in dict.fromkeys(r["phase"] for r in rows):
        part = [r for r in rows if r["phase"] == name]
        program = [r for r in part if r["tier"] == "program"]
        out[name] = {"requests": len(part), "program": len(program), "program_right": sum(r["right"] for r in program),
                     "handed_off": sum(r["handed_off"] for r in part), "read_only": sum(r["read_only"] for r in part),
                     "routed": sum(r["routed"] for r in part),
                     "right": sum(r["right"] for r in part),
                     "dollars_per_request": sum(r["dollars"] for r in part) / len(part),
                     "seconds_per_request": sum(r["seconds"] for r in part) / len(part)}
        c = out[name]
        print(f"{name:<11} {c['requests']:>3} requests: program served {c['program']:>3} ({c['program_right']} right), "
              f"routed {c['routed']:>3}, handed off {c['handed_off']:>3} ({c['read_only']} read-only); right {c['right']:>3}; ${c['dollars_per_request']:.5f} and "
              f"{c['seconds_per_request']:.2f} s a request")
    return out


async def main(args: argparse.Namespace) -> dict[str, Any]:
    if args.mode != "replay":
        import os

        from dotenv import load_dotenv

        load_dotenv(Path.cwd() / ".env")
        os.environ.setdefault("TYPESAFE_API_KEY", os.environ.get("JEV_API_KEY", ""))
    jev = recorded_decider(args.jev_recording, mode=args.mode, model=JEV_MODEL)
    tier = tier_for(args.domain, jev, args)
    agent, meta, world_of = llm._setup(args.domain, llm.MODEL, programs=tier,
                                       stale_parser=args.domain == "d3")

    def tried_on(task: dict[str, Any]) -> Any:
        world = world_of(task)
        world.style = task["style"]
        return world

    tier.world = tried_on
    experience = load(args.domain, "dev+near")
    random.Random(11).shuffle(experience)
    _, held = split_test(args.domain)
    held_out = [*held, *load(args.domain, "stress2")]
    random.Random(12).shuffle(held_out)
    recording = CappedRecording(args.recording, mode=args.mode, spend=Spend(args.budget))
    rows: list[dict[str, Any]] = []
    try:
        with recording.active():
            await phase("experience", experience, agent, meta, world_of, rows, quiet=args.quiet)
            print(f"after experience: {meta.programs.signal()}")
            meta.programs.paused = True  # the held-out requests measure the loop; they must not teach it
            await phase("held_out", held_out, agent, meta, world_of, rows, quiet=args.quiet)
            meta.programs.paused = False
            if args.domain == "d3":
                from examples.invoices.agent import PARSER
                from examples.invoices.world import task_set

                drift = [{**t, "style": "drift", "expect": "handoff" if t["kind"] == "question" else "act"}
                         for t in task_set("stream")[:args.drift]]

                def fix_parser(n: int, meta: Any) -> None:
                    if n == args.fixed_at:
                        print(f"  [drift #{n + 1}] a person installs a corrected parser")
                        meta.install("parse_invoice", PARSER)
                await phase("drift", drift, agent, meta, world_of, rows, quiet=args.quiet, on_request=fix_parser)
    finally:
        summary = report(rows)
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / f"loop_{args.domain}.json").write_text(json.dumps(
            {"summary": summary, "rows": rows, "learning": meta.store.learning(DECISION),
             "signal": meta.programs.signal(), "search": meta.programs.search_log}, indent=1, default=str),
            encoding="utf-8")
        current = meta.programs.current()
        if current is not None:
            (args.output / f"loop_{args.domain}_program.py").write_text(current["source"], encoding="utf-8", newline="\n")
            (args.output / f"loop_{args.domain}_memories.json").write_text(json.dumps(meta.programs._stored(), indent=1), encoding="utf-8")
        found = meta.store.setting("program_mechanisms") or []
        for mechanism in found:
            (args.output / f"{mechanism['name']}.py").write_text(mechanism["source"], encoding="utf-8", newline="\n")
        for entry in meta.store.learning(DECISION):
            print(f"  learning: {entry['mechanism'][:24]:<24} {entry['outcome']:<20} ${entry['dollars'] or 0:.4f}")
        if args.mode != "replay":
            print(f"models: {recording.live_calls} live calls, ${recording.spent(PRICES):.4f}; Jev: "
                  f"{jev.live_usage().calls} live calls, ${jev.live_usage().dollars:.4f}")
    return {"summary": summary, "rows": rows}


def parse(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--domain", default="d3", choices=["d1", "d3", "d5"])
    parser.add_argument("--mechanism", default="program_synthesis")
    parser.add_argument("--no-search", dest="search", action="store_false")
    parser.add_argument("--no-routes", dest="routes", action="store_false", help="programs only act or hand off")
    parser.add_argument("--routed-bar", type=float, default=0.7, help="the LLM's right share on what a program routes to it")
    parser.add_argument("--search-method", default="dgm", choices=METHODS)
    parser.add_argument("--search-target", default="mechanism", choices=["mechanism", "program"],
                        help="rewrite the mechanism, or edit the program it wrote with what works frozen")
    parser.add_argument("--max-attempts", type=int, default=6, help="times the loop tries to close a gap")
    parser.add_argument("--designer-model", default=llm.BUILDER, help="the model that rewrites the code in a search")
    parser.add_argument("--writer-model", default=llm.BUILDER, help="the model a learning component asks to write code")
    parser.add_argument("--designer-effort", default="medium", help="that model's reasoning")
    parser.add_argument("--share-all", action="store_true", help="show the search every test run, the gate's too")
    parser.add_argument("--stop-when-admitted", action="store_true", help="end the search at the first variant the gate admits")
    parser.add_argument("--archive", default=":memory:", help="a path keeps the search's history between runs")
    parser.add_argument("--population", type=int, default=4)
    parser.add_argument("--generations", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--label", default=None)
    parser.add_argument("--min-runs", type=int, default=48)
    parser.add_argument("--max-wrong", type=int, default=1)
    parser.add_argument("--bar", type=float, default=0.9)
    parser.add_argument("--drift", type=int, default=60, help="drift requests (D3); the first drifted one is the 21st")
    parser.add_argument("--fixed-at", type=int, default=40, help="the drift request a corrected parser arrives at")
    parser.add_argument("--mode", default="replay", choices=["replay", "record", "live"])
    parser.add_argument("--recording", type=Path, default=HERE / "recorded" / "llm.jsonl")
    parser.add_argument("--jev-recording", type=Path, default=HERE / "recorded" / "jev.jsonl")
    parser.add_argument("--output", type=Path, default=HERE / "learned")
    parser.add_argument("--budget", type=float, default=5.0, help="dollars of live model calls before stopping")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    try:
        asyncio.run(main(parse()))
    except BudgetSpent as exc:
        sys.exit(f"stopped: {exc}")
