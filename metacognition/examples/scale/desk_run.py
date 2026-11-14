"""Serve the calendar desk's stream with one of three assistants, and keep what every request did and cost.

    python -m examples.scale.desk_run full --soak                 scripted models: no key, nothing spent
    python -m examples.scale.desk_run full --record --budget 19   paid, and stopped before it could pass the budget
    python -m examples.scale.desk_run full                        replay what was recorded, with no key

The three assistants serve requests from the same stream (examples/scale/desk_stream.py).

    alone   the agent at high reasoning with no learned function: every 25th request, and nothing is learned
    loop    the agent with Chapter 6's loop: every fifth request, online
    full    that loop with Chapter 7's decision tier and a night after each day: every request, online

After the last night `loop` and `full` take the final exam three times, each from new recordings, with nothing
learned, quarantined or switched off in between, so one request's three verdicts differ only by what the models
answered. `searches` is a fourth part: five independent
searches on Chapter 7's own request sets, each from the same starting component.

A recording keeps every reply a model gave, so a run that stops can be started again and pays only for what it has
not yet asked. The budget counts what the part's recordings already hold.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

from chapters.ch07.assistant import requests as chapter_requests
from chapters.ch07.assistant.desk import FIND_COMMON_STARTS, contract
from chapters.ch07.assistant.harness import RECORDED, code_search, decision_tier, harness, measure, night, seed, serve
from helix.deciders import recorded_decider
from helix.metacognition import LearningBudget, MetaStore, RunConfig, Runner
from helix.metacognition.decide import JEV_MODEL
from helix.metacognition.exceptions import AgentsException
from helix.metacognition.gap import outcome_of
from helix.metacognition.parts import Verdict
from helix.metacognition.pricing import PRICES
from helix.metacognition.recording import BudgetSpent, CappedRecording, MissingReply, Spend
from helix.signals.cost import call_dollars

from examples.scale import desk_stream
from examples.scale.scripted import NOW, TASKS, Jev, Models, World

HERE = Path(__file__).parent
REPO_ROOT = HERE.parents[2]
ARMS = ("alone", "loop", "full")
EVERY = {"full": 1, "loop": 5, "alone": 25}  # the share of the stream each assistant serves
ROUTINE = ("program: used memory", "program: handed off", "program: routed")


# ---------------------------------------------------------------- models, and what they have cost


DECISION_MODEL = "decision model"


def file_spend(folder: Path) -> dict[str, dict[str, float]]:
    """Calls and dollars by model over every reply a part's recordings hold: what the part has cost so far. A reply
    copied in from the chapter's recordings was paid for before this test, and is left out."""
    out: dict[str, dict[str, float]] = {}
    for path in sorted(folder.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("copied"):
                continue
            if path.name.endswith("_jev.jsonl"):
                model, dollars = DECISION_MODEL, float(row.get("dollars") or 0.0)
            else:
                model = str(row.get("model"))
                dollars = call_dollars(row.get("model") or "", row.get("usage") or {}, PRICES) or 0.0
            cell = out.setdefault(model, {"calls": 0, "dollars": 0.0})
            cell["calls"] += 1
            cell["dollars"] += dollars
    return out


class Transports:
    """The language models and the decision model of one part: scripted for a soak, else recordings under a cap."""

    def __init__(self, args: argparse.Namespace, part: str, scripted: tuple[Any, Any] | None = None) -> None:
        self.args, self.part, self.folder = args, part, args.recorded / part
        self.scripted = bool(args.soak)
        self.spend: Spend | None = None
        if self.scripted:  # another desk passes its own scripted models
            self._models, self._jev = scripted or (Models(careless=0.2, struggles=0.3, stale=0.15),
                                                   Jev(misread=0.01, misroute=0.02))
            return
        if args.record:
            from dotenv import load_dotenv

            load_dotenv(REPO_ROOT / ".env")
            os.environ.setdefault("TYPESAFE_API_KEY", os.environ.get("JEV_API_KEY", ""))
        self.folder.mkdir(parents=True, exist_ok=True)
        # The cap stops the language models' calls. The decision model's are counted and too small to need it.
        already = sum(cell["dollars"] for model, cell in file_spend(self.folder).items() if model != DECISION_MODEL)
        self.spend = Spend(max(0.0, args.budget - already), PRICES)
        self.already = already

    def open(self, name: str = "stream", distinct_repeats: bool = False) -> tuple[Any, Any]:
        """The models and the decision model for one recording of this part."""
        if self.scripted:
            return self._models, self._jev
        mode = "record" if self.args.record else "replay"
        models = CappedRecording(self.folder / f"{name}_models.jsonl", mode=mode, spend=self.spend,
                                 distinct_repeats=distinct_repeats)
        return models, recorded_decider(self.folder / f"{name}_jev.jsonl", mode=mode, model=JEV_MODEL)

    def spent(self) -> dict[str, Any]:
        if self.scripted:
            return {"scripted": dict(self._models.calls), "decision_model_calls": self._jev.calls}
        return {"by_model": file_spend(self.folder)}


# ---------------------------------------------------------------- the assistants


def build(arm: str, stream: desk_stream.Stream, jev: Any, args: argparse.Namespace, folder: Path) -> tuple[Any, Any]:
    """One assistant. Each has Chapter 6's function and loop, set up for a deployment that serves for months."""
    first = stream.days[0]
    settings: dict[str, Any] = {
        "store": MetaStore(keep=2000), "practice": stream.practice, "gate": stream.function_gate(first),
        # Three attempts and $1.50 of learning per function in a day of the stream, so a learner that keeps failing
        # stops costing money until the next day.
        "budget": LearningBudget(attempts_per_day=3, dollars_per_day=1.5,
                                 runs_per_day=max(8, stream.per_day // EVERY[arm])),
        "experience_limit": 8, "learn_from_gating_runs": True, "recall_k": 1, "route_with": jev,
        "confirm_quarantine": True,
        "executor": "guarded" if args.soak else "auto"}
    tier = None
    if arm == "full":
        archive = folder / "archive.sqlite"
        archive.unlink(missing_ok=True)
        tier = decision_tier(jev, [seed("component_llm")], search=code_search("dgm", archive), long_running=True,
                             retest_by_day=True)
    agent, meta = harness(tier, **settings)
    if arm == "alone":
        meta.quarantine("find_shared_starts", "this assistant has no learned function")
    return agent, meta


def morning(day: desk_stream.Day, stream: desk_stream.Stream, meta: Any, learns: bool) -> None:
    """What a day starts with: the note in force, the requests held back from it, and any new kind of work."""
    if NOW["conditions"] != day.note.key:
        NOW["before"], NOW["conditions"] = NOW["conditions"], day.note.key  # what a scripted builder writes for
    meta.gate = stream.function_gate(day)
    if meta.programs is not None:
        meta.programs.tier.labeled, meta.programs.tier.search_cases = stream.held_back(day)
    if learns and day.second_class and FIND_COMMON_STARTS.name not in meta.procedures:
        # The team says what the new kind of work needs. No code comes with it: the loop has to learn the function.
        meta.add_procedure(FIND_COMMON_STARTS)
        if meta.programs is not None:
            meta.programs.tier.contract = contract(second_class=True)


def _row(arm: str, phase: str, task: dict[str, Any], *, right: bool, outcome: str, result: Any = None,
         report: Any = None, world: Any = None, failed: str = "") -> dict[str, Any]:
    cost = getattr(result, "cost", None)
    by_source = getattr(cost, "by_source", None) or {}
    events = [event for event in (report.events if report is not None else [])
              if not event.startswith(ROUTINE) or "failed" in event]
    return {"arm": arm, "phase": phase, "day": task.get("day"), "seq": task.get("seq"), "id": task["id"],
            "asked": task["asked"], "group": task.get("group", ""), "sensitive": bool(task.get("sensitive")),
            "outage": bool(task.get("outage")), "right": right, "outcome": outcome,
            "effort": getattr(report, "effort", None), "memory": getattr(report, "memory", None),
            "dollars": float(getattr(cost, "dollars", 0.0) or 0.0),
            "decider_dollars": float((by_source.get("decider") or {}).get("dollars", 0.0)),
            "seconds": float(getattr(cost, "seconds", 0.0) or 0.0),
            "booked": len(getattr(world, "booked", []) or []), "events": events, "failed": failed}


async def serve_one(arm: str, phase: str, agent: Any, task: dict[str, Any], world_for: Any = World
                    ) -> dict[str, Any]:
    """One request served online: the loop records it, judges it, and may quarantine, relearn or switch off."""
    world = world_for(task)
    config = RunConfig(max_tool_calls=24, case_id=task["id"], case_data=task)
    try:
        result = await Runner.run(agent, task["text"], context=world, max_turns=16, run_config=config)
    except AgentsException as exc:
        return _row(arm, phase, task, right=False, outcome="lost", world=world, failed=f"{type(exc).__name__}: {exc}")
    report = result.metacognition
    right = bool(report.verdict and report.verdict.accepted)
    outcome = outcome_of(right, program=report.program, should_act=task["expect"] == "act")
    return _row(arm, phase, task, right=right, outcome=outcome, result=result, report=report, world=world)


async def serve_frozen(arm: str, agent: Any, meta: Any, tasks: list[dict[str, Any]], effort: str,
                       at_once: int = 6, world_for: Any = World) -> list[dict[str, Any]]:
    """Requests served with nothing recorded and nothing learned, at one reasoning level."""
    slots = asyncio.Semaphore(at_once)

    async def one(task: dict[str, Any]) -> dict[str, Any]:
        world = world_for(task)
        config = RunConfig(metacognition_mode="frozen", effort=effort, max_tool_calls=24, case_id=task["id"],
                           case_data=task)
        async with slots:
            try:
                result = await Runner.run(agent, task["text"], context=world, max_turns=16, run_config=config)
            except AgentsException as exc:
                return _row(arm, "stream", task, right=False, outcome="lost", world=world,
                            failed=f"{type(exc).__name__}: {exc}")
        verdict = Verdict.coerce(meta.verifier(result.context_wrapper, result))
        right = bool(verdict and verdict.accepted)
        row = _row(arm, "stream", task, right=right, outcome="llm" if right else "llm_wrong", result=result,
                   world=world)
        return {**row, "effort": effort}

    return list(await asyncio.gather(*(one(task) for task in tasks)))


def _say(text: str) -> None:
    print(text, flush=True)


def _day_line(day: desk_stream.Day, rows: list[dict[str, Any]], spent: float | None, started: float) -> str:
    kinds = Counter(row["outcome"] for row in rows)
    right = sum(row["right"] for row in rows)
    dollars = sum(row["dollars"] for row in rows)
    notable = Counter(event.split(" ", 2)[0] + " " + event.split(" ", 2)[1] for row in rows for event in row["events"])
    cap = "" if spent is None else f", ${spent:.2f} of new calls so far"
    return (f"day {day.number:>2}: {right}/{len(rows)} right, {dict(kinds)}, ${dollars:.3f} served{cap}, "
            f"{time.perf_counter() - started:.0f} s; {dict(notable) if notable else 'no events'}"
            f"{'  <- ' + day.change if day.change else ''}")


# ---------------------------------------------------------------- one assistant over the stream


async def run_arm(arm: str, args: argparse.Namespace) -> dict[str, Any]:
    stream = desk_stream.Stream(args.per_day, args.days, weeks=desk_stream.Weeks(args.weeks), held=args.held)
    drawn = stream.prepare(args.workers)
    transports = Transports(args, arm)
    folder = transports.folder if not transports.scripted else args.output
    folder.mkdir(parents=True, exist_ok=True)
    models, jev = transports.open("stream")
    NOW.update(conditions=stream.days[0].note.key, before=stream.days[0].note.key)
    agent, meta = build(arm, stream, jev, args, folder)
    rows: list[dict[str, Any]] = []
    nights: list[dict[str, Any]] = []
    out: dict[str, Any] = {"arm": arm, "settings": {"per_day": args.per_day, "days": args.days, "soak": bool(args.soak),
                                                    "serves_every": EVERY[arm]}, "stopped": ""}
    _say(f"{arm}: {args.days} days of {args.per_day}, every {EVERY[arm]} request(s); {drawn} weeks drawn"
         + ("" if transports.spend is None else f"; ${transports.already:.2f} already recorded, "
            f"${transports.spend.limit:.2f} left under the budget"))
    started = time.perf_counter()
    try:
        with models.active():
            for day in stream.days:
                morning(day, stream, meta, learns=arm != "alone")
                mine = [task for task in stream.served(day) if task["seq"] % EVERY[arm] == 0]
                if arm == "alone":
                    today = await serve_frozen(arm, agent, meta, mine, "high")
                else:
                    today = [await serve_one(arm, "stream", agent, task) for task in mine]
                rows += today
                _say(_day_line(day, today, None if transports.spend is None else transports.spend.total, started))
                if arm == "full":
                    events = await night(meta)
                    shown = [event for event in events if not event.startswith("search: design ")]
                    nights.append({"day": day.number, "events": shown})
                    for event in shown:
                        _say(f"   night: {event[:200]}")
            if arm != "alone":
                rows += await exam(arm, agent, meta, stream, transports, started)
    except BudgetSpent as exc:
        out["stopped"] = f"budget: {exc}"
        _say(f"Stopped: {exc}")
    except MissingReply as exc:
        out["stopped"] = f"replay: {exc.args[0]}"
        _say(f"Stopped: {exc.args[0]}. Replay reads saved replies only; --record adds them.")
    out.update(rows=rows, nights=nights, learning=meta.store.learning(), spent=transports.spent(),
               search=meta.programs.search_log if meta.programs is not None else [],
               memories=meta.programs.memories() if meta.programs is not None else [],
               seconds=round(time.perf_counter() - started, 1))
    if not transports.scripted and args.record:
        usage = jev.live_usage()
        out["decision_model_live"] = {"calls": usage.calls, "dollars": usage.dollars}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / f"{arm}.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    right = sum(row["right"] for row in rows if row["phase"] == "stream")
    served = sum(row["phase"] == "stream" for row in rows)
    _say(f"{arm}: {right}/{served} right over the stream in {out['seconds']:.0f} s"
         + (f"; {out['stopped']}" if out["stopped"] else ""))
    return out


def freeze(meta: Any) -> None:
    """Keep the assistant as the last night left it: it serves and records, and nothing it sees changes it."""
    meta.learning = "worker"  # serving only records, so nothing is relearned
    meta.gating_sources = ()  # no verdict can quarantine a function
    if meta.programs is not None:
        meta.programs.paused = True  # no improvement
        meta.programs.tier.gating_sources = ()  # and no verdict can switch a program's action off


async def exam(arm: str, agent: Any, meta: Any, stream: desk_stream.Stream, transports: Transports,
               started: float) -> list[dict[str, Any]]:
    """The final exam, served three times by the assistant the last night left, each time from new recordings."""
    tasks = stream.exam()
    freeze(meta)
    rows = [await serve_one(arm, "exam", agent, task) for task in tasks]
    _say(f"exam: {sum(row['right'] for row in rows)}/{len(rows)} right, {time.perf_counter() - started:.0f} s")
    for number in (2, 3):
        models, jev = transports.open(f"repeat{number}")
        deciders = [jev] if transports.scripted else _swap_decider(meta, jev)
        with models.active():
            again = [await serve_one(arm, f"repeat{number}", agent, task) for task in tasks]
        rows += again
        _say(f"repeat {number}: {sum(row['right'] for row in again)}/{len(again)} right, "
             f"{time.perf_counter() - started:.0f} s ({len(deciders)} decision models)")
    return rows


def _swap_decider(meta: Any, jev: Any) -> list[Any]:
    """Give the assistant a decision model with a recording of its own, so a repeat asks its questions again."""
    meta.route_with = jev
    if meta.programs is not None:
        meta.programs.tier.decider = jev
    return [jev]


# ---------------------------------------------------------------- five searches from the same start


async def run_searches(args: argparse.Namespace) -> dict[str, Any]:
    """Five independent searches on Chapter 7's requests. Each starts from the component that keeps every request on
    the agent, with the day of experience the chapter recorded, and asks the builder afresh."""
    transports = Transports(args, "searches")
    floor_rows: list[dict[str, Any]] = []
    out: dict[str, Any] = {"arm": "searches", "searches": [], "stopped": ""}
    held_out = chapter_requests.load("held_out")
    if transports.scripted:  # the scripted models find a request's truth by its words
        for name in ("experience", "search", "gate", "held_out"):
            TASKS.update({request["text"]: request for request in chapter_requests.load(name)})
    started = time.perf_counter()
    try:
        for number in range(1, args.searches + 1):
            models, jev = _search_transports(transports, number)
            archive = (transports.folder if not transports.scripted else args.output) / f"archive_{number}.sqlite"
            archive.parent.mkdir(parents=True, exist_ok=True)
            archive.unlink(missing_ok=True)
            search = code_search("dgm", archive, seed=100 + number, label=f"calendar, repeat {number}")
            tier = decision_tier(jev, [seed("component_llm")], search=search)
            agent, meta = harness(tier, executor="guarded" if args.soak else "auto")
            log: list[dict[str, Any]] = []
            with models.active():
                if number == 1:
                    await measure(agent, meta, held_out, meta.controller.table["memory"], "floor", floor_rows)
                await serve(agent, chapter_requests.week()["Monday"], "experience", log)
                events = await night(meta)
                admitted = meta.programs.current() is not None
                if admitted:
                    await serve(agent, held_out, "searched component", log)
            served = [row for row in log if row["stage"] == "searched component"]
            tried = meta.programs.search_log
            row = {"search": number, "admitted": admitted, "rewrites": len(tried),
                   "outcomes": [str(t.get("outcome")) for t in tried],
                   "held_out_right": sum(r["right"] for r in served), "held_out": len(served),
                   "dollars_per_request": sum(r["dollars"] for r in served) / len(served) if served else None,
                   "seconds_per_request": sum(r["seconds"] for r in served) / len(served) if served else None,
                   "served_by_programs": sum(r["outcome"] == "program" for r in served),
                   "wrong_program_actions": sum(r["outcome"] == "wrong" for r in served),
                   "night": [e for e in events if not e.startswith("search: design ")][:6]}
            out["searches"].append(row)
            _say(f"search {number}: {'admitted' if admitted else 'nothing admitted'} after {row['rewrites']} "
                 f"rewrites, held out {row['held_out_right']}/{row['held_out']}, "
                 f"{time.perf_counter() - started:.0f} s"
                 + ("" if transports.spend is None else f", ${transports.spend.total:.2f} of new calls"))
    except BudgetSpent as exc:
        out["stopped"] = f"budget: {exc}"
        _say(f"Stopped: {exc}")
    except MissingReply as exc:
        out["stopped"] = f"replay: {exc.args[0]}"
        _say(f"Stopped: {exc.args[0]}. Replay reads saved replies only; --record adds them.")
    out.update(floor={"right": sum(r["right"] for r in floor_rows), "requests": len(floor_rows)},
               spent=transports.spent(), seconds=round(time.perf_counter() - started, 1))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "searches.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    return out


def _search_transports(transports: Transports, number: int) -> tuple[Any, Any]:
    """A search's recordings start from the chapter's replies by the agent and the decision model, so the day of
    experience costs nothing again. They hold none of the builder's replies, so every search asks the builder."""
    if transports.scripted:
        return transports.open()
    models_path = transports.folder / f"search{number}_models.jsonl"
    jev_path = transports.folder / f"search{number}_jev.jsonl"
    if not models_path.exists():
        for source, target in ((RECORDED / "models.jsonl", models_path), (RECORDED / "jev.jsonl", jev_path)):
            rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
            kept = [json.dumps({**row, "copied": True}) for row in rows if "claude" not in str(row.get("model"))]
            target.write_text("\n".join(kept) + "\n", encoding="utf-8", newline="\n")
    return transports.open(f"search{number}")


# ---------------------------------------------------------------- the command line


def parse(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("part", choices=(*ARMS, "searches"))
    parser.add_argument("--soak", action="store_true", help="scripted models that misbehave: no key and no spend")
    parser.add_argument("--record", action="store_true", help="call the models for replies the recordings lack")
    parser.add_argument("--budget", type=float, default=0.0, help="dollars this part may cost in all, with what its "
                                                                 "recordings already hold")
    parser.add_argument("--per-day", type=int, default=200)
    parser.add_argument("--days", type=int, default=20)
    parser.add_argument("--held", type=int, default=16, help="requests in each set held back from a day")
    parser.add_argument("--searches", type=int, default=5)
    parser.add_argument("--workers", type=int, default=0, help="processes that draw weeks; 0 uses every processor")
    parser.add_argument("--weeks", type=Path, default=desk_stream.CACHE, help="where drawn weeks are kept")
    parser.add_argument("--recorded", type=Path, default=HERE / "runs" / "desk_calendar" / "recorded")
    parser.add_argument("--output", type=Path, default=None, help="where results go")
    args = parser.parse_args(argv)
    if args.record and args.soak:
        parser.error("--soak calls no model, so it has nothing to record")
    if args.record and args.budget <= 0:
        parser.error("--record needs --budget, the dollars this part may cost in all")
    if args.output is None:
        args.output = HERE / "runs" / "desk_calendar" / ("soak" if args.soak else "results")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse(argv)
    if sys.platform == "win32":  # a recording's long runs print names the console's code page may lack
        sys.stdout.reconfigure(errors="replace")  # type: ignore[union-attr]
    asyncio.run(run_searches(args) if args.part == "searches" else run_arm(args.part, args))


if __name__ == "__main__":
    main()
