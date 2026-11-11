"""Serve the support inbox's stream with one of three assistants, as the calendar desk's driver does.

    python -m examples.inbox.run start --soak            learn the routing function every assistant starts with
    python -m examples.inbox.run full --soak             scripted models: no key, nothing spent
    python -m examples.inbox.run full --record --budget 19

The parts and their order are the calendar desk's: `start` first (forty messages before the stream, from which the
loop learns the routing function that all three assistants then start with), then `alone`, `loop`, `full` and
`searches`. examples/scale/desk_report.py turns the results into the nine pass criteria.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

from helix.metacognition import LearningBudget, MetaStore
from helix.metacognition.recording import BudgetSpent, MissingReply

from examples.inbox import scripted
from examples.inbox.desk import code_search, decision_tier, harness
from examples.inbox.stream import Data, Stream
from examples.inbox.world import Inbox
from examples.scale.desk_run import EVERY, Transports, _day_line, _say, freeze, serve_frozen, serve_one

HERE = Path(__file__).parent
RUNS = HERE.parents[0] / "scale" / "runs" / "desk_inbox"
ARMS = ("alone", "loop", "full")


def transports(args: argparse.Namespace, part: str) -> Transports:
    return Transports(args, part, scripted=(scripted.InboxModels(), scripted.InboxJev()) if args.soak else None)


def _stream(args: argparse.Namespace) -> Stream:
    return Stream(args.per_day, args.days, data=Data(args.data), held=args.held)


def learned_function(args: argparse.Namespace) -> str:
    path = args.output / "route_ticket_start.py"
    if not path.exists():
        sys.exit(f"Run the start part first: {path} holds the routing function every assistant starts with.")
    return path.read_text(encoding="utf-8")


def build(arm: str, stream: Stream, jev: Any, args: argparse.Namespace, folder: Path) -> tuple[Any, Any]:
    tier = None
    if arm == "full":
        archive = folder / "archive.sqlite"
        archive.unlink(missing_ok=True)
        tier = decision_tier(jev, stream, stream.days[0], search=code_search(archive), long_running=True,
                             retest_by_day=True)
    settings = {"budget": LearningBudget(attempts_per_day=3, dollars_per_day=1.5,
                                         runs_per_day=max(8, stream.per_day // EVERY[arm])),
                "experience_limit": 8, "learn_from_gating_runs": True, "confirm_quarantine": True,
                "executor": "guarded" if args.soak else "auto", "store": MetaStore(keep=2000)}
    agent, meta = harness(stream, tier, function=learned_function(args), **settings)
    if arm == "alone":
        meta.quarantine("route_ticket", "this assistant has no learned function")
    return agent, meta


def morning(day: Any, stream: Stream, meta: Any) -> None:
    if scripted.NOW["policy"] != day.policy.key:
        scripted.NOW["before"], scripted.NOW["policy"] = scripted.NOW["policy"], day.policy.key
    meta.gate = stream.function_gate(day)
    if meta.programs is not None:
        meta.programs.tier.labeled, meta.programs.tier.search_cases = stream.held_back(day)


def _settings(args: argparse.Namespace, arm: str) -> dict[str, Any]:
    return {"desk": "inbox", "per_day": args.per_day, "days": args.days, "soak": bool(args.soak),
            "serves_every": EVERY.get(arm, 1)}


async def run_start(args: argparse.Namespace) -> dict[str, Any]:
    """Serve forty messages with no routing function, and keep the function the loop learns from them."""
    stream = _stream(args)
    part = transports(args, "start")
    models, jev = part.open("start")
    scripted.NOW.update(policy=stream.days[0].policy.key, before=stream.days[0].policy.key)
    settings = {"executor": "guarded" if args.soak else "auto", "experience_limit": 8, "learn_from_gating_runs": True,
                "budget": LearningBudget(attempts_per_day=3, dollars_per_day=1.5, runs_per_day=40)}
    agent, meta = harness(stream, None, function=None, **settings)
    rows, out = [], {"arm": "start", "settings": _settings(args, "start"), "stopped": ""}
    try:
        with models.active():
            for task in stream.start():
                rows.append(await serve_one("start", "start", agent, {**task, "day": 0, "seq": len(rows)},
                                            world_for=Inbox))
                if meta.store.trusted("route_ticket") is not None:
                    break  # learned: the stream's assistants start from here
    except (BudgetSpent, MissingReply) as exc:
        out["stopped"] = f"{type(exc).__name__}: {exc}"
    learned = meta.store.trusted("route_ticket")
    out.update(rows=rows, learning=meta.store.learning(), spent=part.spent(),
               learned=None if learned is None else learned["source"])
    args.output.mkdir(parents=True, exist_ok=True)
    if learned is not None:
        (args.output / "route_ticket_start.py").write_text(learned["source"], encoding="utf-8", newline="\n")
    (args.output / "start.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    said = "learned and kept" if learned is not None else "not learned"
    _say(f"start: {sum(r['right'] for r in rows)}/{len(rows)} right; route_ticket {said}"
         + (f"; {out['stopped']}" if out["stopped"] else ""))
    return out


async def run_arm(arm: str, args: argparse.Namespace) -> dict[str, Any]:
    stream = _stream(args)
    part = transports(args, arm)
    folder = part.folder if not part.scripted else args.output
    folder.mkdir(parents=True, exist_ok=True)
    models, jev = part.open("stream")
    scripted.NOW.update(policy=stream.days[0].policy.key, before=stream.days[0].policy.key)
    agent, meta = build(arm, stream, jev, args, folder)
    rows: list[dict[str, Any]] = []
    nights: list[dict[str, Any]] = []
    out: dict[str, Any] = {"arm": arm, "settings": _settings(args, arm), "stopped": ""}
    _say(f"{arm}: {args.days} days of {args.per_day}, every {EVERY[arm]} message(s)"
         + ("" if part.spend is None else f"; ${part.already:.2f} already recorded, ${part.spend.limit:.2f} left"))
    started = time.perf_counter()
    try:
        with models.active():
            for day in stream.days:
                morning(day, stream, meta)
                mine = [task for task in stream.served(day) if task["seq"] % EVERY[arm] == 0]
                if arm == "alone":
                    today = await serve_frozen(arm, agent, meta, mine, "high", world_for=Inbox)
                else:
                    today = [await serve_one(arm, "stream", agent, task, world_for=Inbox) for task in mine]
                rows += today
                _say(_day_line(day, today, None if part.spend is None else part.spend.total, started))
                if arm == "full":
                    events = [e for e in await _night(meta) if not e.startswith("search: design ")]
                    nights.append({"day": day.number, "events": events})
                    for event in events:
                        _say(f"   night: {event[:200]}")
            if arm != "alone":
                rows += await exam(arm, agent, meta, stream, part, started)
    except BudgetSpent as exc:
        out["stopped"] = f"budget: {exc}"
        _say(f"Stopped: {exc}")
    except MissingReply as exc:
        out["stopped"] = f"replay: {exc.args[0]}"
        _say(f"Stopped: {exc.args[0]}. Replay reads saved replies only; --record adds them.")
    out.update(rows=rows, nights=nights, learning=meta.store.learning(), spent=part.spent(),
               search=meta.programs.search_log if meta.programs is not None else [],
               memories=meta.programs.memories() if meta.programs is not None else [],
               seconds=round(time.perf_counter() - started, 1))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / f"{arm}.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    stream_rows = [row for row in rows if row["phase"] == "stream"]
    _say(f"{arm}: {sum(r['right'] for r in stream_rows)}/{len(stream_rows)} right over the stream in "
         f"{out['seconds']:.0f} s" + (f"; {out['stopped']}" if out["stopped"] else ""))
    return out


async def _night(meta: Any) -> list[str]:
    events: list[str] = []
    meta.programs.paused = False
    try:
        await meta.programs.maybe_improve(events)
    finally:
        meta.programs.paused = True
    return events


async def exam(arm: str, agent: Any, meta: Any, stream: Stream, part: Transports, started: float
               ) -> list[dict[str, Any]]:
    tasks = stream.exam()
    freeze(meta)
    rows = [await serve_one(arm, "exam", agent, task, world_for=Inbox) for task in tasks]
    _say(f"exam: {sum(r['right'] for r in rows)}/{len(rows)} right, {time.perf_counter() - started:.0f} s")
    for number in (2, 3):
        models, jev = part.open(f"repeat{number}")
        if not part.scripted and meta.programs is not None:
            meta.programs.tier.decider = jev
        with models.active():
            again = [await serve_one(arm, f"repeat{number}", agent, task, world_for=Inbox) for task in tasks]
        rows += again
        _say(f"repeat {number}: {sum(r['right'] for r in again)}/{len(again)} right")
    return rows


async def run_searches(args: argparse.Namespace) -> dict[str, Any]:
    """Five independent searches, each from the component that keeps every message on the agent, after the same
    recorded day of experience, measured on the same 52 held-out messages."""
    stream = _stream(args)
    part = transports(args, "searches")
    first = stream.days[0]
    held_out = stream.held_out()
    out: dict[str, Any] = {"arm": "searches", "searches": [], "stopped": "", "settings": _settings(args, "searches")}
    floor_rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    try:
        for number in range(1, args.searches + 1):
            models, jev = _search_transports(part, number)
            folder = part.folder if not part.scripted else args.output
            archive = folder / f"archive_{number}.sqlite"
            archive.parent.mkdir(parents=True, exist_ok=True)
            archive.unlink(missing_ok=True)
            tier = decision_tier(jev, stream, first, search=code_search(archive, seed=100 + number,
                                                                        label=f"inbox, repeat {number}"))
            agent, meta = harness(stream, tier, function=learned_function(args),
                                  executor="guarded" if args.soak else "auto")
            scripted.NOW.update(policy=first.policy.key, before=first.policy.key)
            log: list[dict[str, Any]] = []
            with models.active():
                if number == 1:  # the agent at its floor with the function, frozen: nothing is recorded or learned
                    floor_rows = await serve_frozen("floor", agent, meta, held_out, "none", world_for=Inbox)
                for task in stream.served(first)[:80]:
                    log.append(await serve_one("experience", "experience", agent, task, world_for=Inbox))
                events = await _night(meta)
                admitted = meta.programs.current() is not None
                served = [await serve_one("held_out", "held_out", agent, task, world_for=Inbox)
                          for task in held_out] if admitted else []
            row = {"search": number, "admitted": admitted, "rewrites": len(meta.programs.search_log),
                   "held_out_right": sum(r["right"] for r in served), "held_out": len(served),
                   "dollars_per_request": sum(r["dollars"] for r in served) / len(served) if served else None,
                   "seconds_per_request": sum(r["seconds"] for r in served) / len(served) if served else None,
                   "served_by_programs": sum(r["outcome"] == "program" for r in served),
                   "wrong_program_actions": sum(r["outcome"] == "wrong" for r in served),
                   "night": [e for e in events if not e.startswith("search: design ")][:6]}
            out["searches"].append(row)
            _say(f"search {number}: {'admitted' if admitted else 'nothing admitted'} after {row['rewrites']} rewrites, "
                 f"held out {row['held_out_right']}/{row['held_out']}, {time.perf_counter() - started:.0f} s")
    except (BudgetSpent, MissingReply) as exc:
        out["stopped"] = f"{type(exc).__name__}: {exc}"
        _say(f"Stopped: {exc}")
    out.update(floor={"right": sum(r["right"] for r in floor_rows), "requests": len(floor_rows)},
               spent=part.spent(), seconds=round(time.perf_counter() - started, 1))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "searches.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    return out


def _search_transports(part: Transports, number: int) -> tuple[Any, Any]:
    """Every search after the first starts from the first one's recorded replies by the agent and the decision model,
    so the day of experience is paid for once. None of the builder's replies are kept, so each search asks afresh."""
    if part.scripted or number == 1:
        return part.open(f"search{number}") if not part.scripted else part.open()
    for kind in ("models", "jev"):
        source, target = part.folder / f"search1_{kind}.jsonl", part.folder / f"search{number}_{kind}.jsonl"
        if source.exists() and not target.exists():
            rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
            kept = [json.dumps({**row, "copied": True}) for row in rows if "claude" not in str(row.get("model"))]
            target.write_text("\n".join(kept) + "\n", encoding="utf-8", newline="\n")
    return part.open(f"search{number}")


def parse(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("part", choices=("start", *ARMS, "searches"))
    parser.add_argument("--soak", action="store_true")
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--budget", type=float, default=0.0)
    parser.add_argument("--per-day", type=int, default=200)
    parser.add_argument("--days", type=int, default=20)
    parser.add_argument("--held", type=int, default=16)
    parser.add_argument("--searches", type=int, default=5)
    parser.add_argument("--data", type=Path, default=HERE.parents[0] / "scale" / "runs" / "outside_data")
    parser.add_argument("--recorded", type=Path, default=RUNS / "recorded")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.record and args.soak:
        parser.error("--soak calls no model, so it has nothing to record")
    if args.record and args.budget <= 0:
        parser.error("--record needs --budget")
    if args.output is None:
        args.output = RUNS / ("soak" if args.soak else "results")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse(argv)
    if sys.platform == "win32":
        sys.stdout.reconfigure(errors="replace")  # type: ignore[union-attr]
    if args.part == "start":
        asyncio.run(run_start(args))
    elif args.part == "searches":
        asyncio.run(run_searches(args))
    else:
        asyncio.run(run_arm(args.part, args))


__all__ = ["main", "parse", "run_arm", "run_start", "run_searches", "Counter", "shutil"]

if __name__ == "__main__":
    main()
