"""Serve a long scripted stream and say whether the loop held over its whole length.

    python -m examples.scale.long_run --requests 100000

No model is called, so a run costs nothing but time. The calendar desk serves days of requests with a night after
each, as listing 7.4 does, while conditions keep changing: Dana's note every other day, some days of garbled
requests, some stretches with the calendar down, and one long quiet stretch. The script reports four things.

    stalls      a day with nothing going on where programs serve far less than they usually do
    recovery    after each change of note, whether the function is trusted and the action on again by that night,
                and whether the next uneventful day is back to normal
    learning    builder calls, which should follow the changes and not the number of requests
    overhead    the harness's own time per request, each night's time, and the store's size, over the stream

`--lifetime-limits` runs the harness as it was before limits were counted for each gap, and `--keep 0` keeps every
episode, to show what each of those costs over a long stream. Learned functions run in this process, so the time
reported is the loop's own. `--executor subprocess` starts a new process for every call, as the chapters do; that
isolation costs tens of milliseconds a call, which hides the loop's few.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any

from chapters.ch07.assistant.harness import code_search, decision_tier, harness, night, seed
from helix.metacognition import Gate, LearningBudget, MetaStore, RunConfig, Runner
from helix.metacognition.exceptions import AgentsException
from helix.metacognition.gap import outcome_of

from examples.scale import stream
from examples.scale.scripted import NOW, Jev, Models, World, function_for

HERE = Path(__file__).parent
WATCHED = {"program: switched off": "switched_off", "detector: quarantined": "quarantined",
           "learner: skipped": "skipped"}


def build(args: argparse.Namespace, jev: Jev) -> tuple[Any, Any, Any]:
    """The calendar assistant of Chapter 7, set up for a stream that is long and keeps changing."""
    tier = decision_tier(jev, [seed("component_llm")], search=code_search("dgm"), long_running=not args.lifetime_limits)
    budget = (LearningBudget(attempts_per_day=2) if args.lifetime_limits
              else LearningBudget(attempts_per_day=3, runs_per_day=args.per_day))
    agent, meta = harness(tier, store=MetaStore(keep=args.keep or None), budget=budget, executor=args.executor,
                          experience_limit=None if args.lifetime_limits else 8)
    meta.install("find_shared_starts", function_for("ordinary"), created_by="scripted")
    return agent, meta, tier


async def a_day(day: stream.Day, tasks: list[dict[str, Any]], agent: Any) -> dict[str, Any]:
    """Serve one day and count what happened."""
    outcomes: Counter[str] = Counter()
    seen: Counter[str] = Counter()
    took = []
    for task in tasks:
        config = RunConfig(max_tool_calls=24, case_id=task["id"], case_data=task)
        started = time.perf_counter()
        try:
            result = await Runner.run(agent, task["text"], context=World(task), max_turns=16, run_config=config)
        except AgentsException:
            outcomes["failed"] += 1
            continue
        took.append(time.perf_counter() - started)
        report = result.metacognition
        right = bool(report.verdict and report.verdict.accepted)
        outcomes[outcome_of(right, program=report.program, should_act=task["expect"] == "act")] += 1
        outcomes["right"] += right
        for event in report.events:
            for start, name in WATCHED.items():
                seen[name] += event.startswith(start)
            seen["relearned"] += event.startswith("learner:") and "installed" in event
    took.sort()
    return {"day": day.number, "conditions": day.conditions, "changed": day.changed, "garbled": day.garbled,
            "outage": day.outage, "requests": len(tasks), **{k: outcomes[k] for k in (
                "right", "program", "wrong", "llm", "llm_wrong", "read_only", "lost", "failed")},
            **{k: seen[k] for k in ("switched_off", "quarantined", "relearned", "skipped")},
            "ms": round(1000 * statistics.fmean(took), 2) if took else 0.0,
            "ms_p95": round(1000 * took[int(0.95 * (len(took) - 1))], 2) if took else 0.0}


async def main(args: argparse.Namespace) -> dict[str, Any]:
    days = max(4, args.requests // args.per_day)
    quiet = (days // 2, days // 2 + args.quiet_days - 1) if args.quiet_days else (0, 0)
    calendar = stream.plan(days, change_every=args.change_every, quiet=quiet)
    weeks, jev, models = stream.Weeks(args.weeks), Jev(misread=args.misread), Models(careless=args.careless)
    agent, meta, tier = build(args, jev)
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    kept_days = (args.keep // args.per_day + 3) if args.keep else None
    with models.active():
        for day in calendar:
            NOW["conditions"] = day.conditions
            meta.gate = Gate(cases=stream.FUNCTION_WEEKS[day.conditions])  # today's weeks judge a relearned function
            tasks = stream.served(day, args.per_day, weeks)
            gate, search = stream.held_back(day, weeks)
            builder_before = models.calls["builder"]
            row = await a_day(day, tasks, agent)
            tier.labeled, tier.search_cases = gate, search  # held back from today: tonight's gate and search
            night_started = time.perf_counter()
            events = await night(meta)
            loop = meta.store.state("decision:loop")
            row.update(night_s=round(time.perf_counter() - night_started, 3),
                       builder=models.calls["builder"] - builder_before,
                       night=[e for e in events if not e.startswith("search: design ")][:4],
                       attempts=loop.get("attempts", 0), store_mb=round(meta.store.size() / 1e6, 2),
                       trusted=meta.store.trusted("find_shared_starts") is not None,
                       off=bool(meta.programs.off("book_time")))
            rows.append(row)
            if kept_days:
                stream.forget(day.number - kept_days)
            if day.number % max(1, days // 20) == 0 and not args.quiet:
                done = sum(r["requests"] for r in rows)
                print(f"day {day.number:>4} of {days}: {done:>7} requests, {time.perf_counter() - started:>6.0f} s, "
                      f"store {row['store_mb']:.1f} MB, {row['ms']:.1f} ms a request", flush=True)
    weeks.save()
    out = {"settings": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
           "seconds": round(time.perf_counter() - started, 1), "calls": dict(models.calls), "jev_calls": jev.calls,
           "summary": summarize(rows), "days": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=1), encoding="utf-8")
    show(out)
    return out


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Whether the loop held: stalls, recovery after each change, where learning was spent, and overhead."""
    def share(row: dict[str, Any]) -> float:
        return row["program"] / row["requests"]

    def eventful(row: dict[str, Any]) -> bool:
        return row["changed"] or row["garbled"] or row["outage"]

    served = rows[1:]  # the first day has no program yet
    calm = [r for i, r in enumerate(served) if not eventful(r) and not eventful(rows[i])]  # and no event the day before
    usual = statistics.median(share(r) for r in rows[1:3]) if len(rows) > 2 else 0.0  # the days after the install
    stalled = [r["day"] for r in calm if share(r) < 0.5 * usual]
    def normal(row: dict[str, Any]) -> bool:
        # Programs serve about what they usually do and the day is right but for one request in thirty. A day of
        # few requests varies by chance, so the share gets two standard errors of room.
        room = 2 * math.sqrt(usual * (1 - usual) / row["requests"])
        return (share(row) >= 0.9 * usual - room
                and row["requests"] - row["right"] <= max(1, 0.03 * row["requests"]))

    changes = [i for i, r in enumerate(rows) if r["changed"]]
    after = [rows[i + 1] for i in changes if i + 1 < len(rows) and not eventful(rows[i + 1])]
    recovered = [r for r in after if normal(r)]
    false_alarms = [r["day"] for r in rows if r["switched_off"] and not eventful(r)]
    tenths = [rows[i * len(rows) // 10:(i + 1) * len(rows) // 10] for i in range(10)]
    requests = sum(r["requests"] for r in rows)
    return {
        "requests": requests, "days": len(rows), "right": sum(r["right"] for r in rows),
        "served_by_programs": sum(r["program"] for r in rows), "usual_program_share": round(usual, 3),
        "wrong_program_actions": sum(r["wrong"] for r in rows),
        "changes": len(changes), "changes_checked": len(after), "changes_recovered": len(recovered),
        "changes_repaired_by_night": sum(rows[i]["trusted"] and not rows[i]["off"] for i in changes),
        "days_ended_unrepaired": [r["day"] for r in rows if r["off"] or not r["trusted"]],
        "wrong_on_change_days": _spread([rows[i]["requests"] - rows[i]["right"] for i in changes]),
        "wrong_program_actions_on_change_days": _spread([rows[i]["wrong"] for i in changes]),
        "stalled_days": stalled, "false_alarm_days": false_alarms,
        "relearned": sum(r["relearned"] for r in rows), "relearning_skipped": sum(r["skipped"] for r in rows),
        "builder_calls": sum(r["builder"] for r in rows),
        "builder_calls_on_calm_days": sum(r["builder"] for r in rows[1:] if not eventful(r)),
        "builder_calls_per_change": round(sum(r["builder"] for r in rows if r["changed"]) / max(1, len(changes)), 2),
        "ms_by_tenth": [round(statistics.fmean(r["ms"] for r in part), 2) for part in tenths if part],
        "night_s_by_tenth": [round(max(r["night_s"] for r in part), 2) for part in tenths if part],
        "store_mb_by_tenth": [part[-1]["store_mb"] for part in tenths if part],
    }


def _spread(counts: list[int]) -> dict[str, float]:
    return {"mean": round(statistics.fmean(counts), 2), "max": max(counts)} if counts else {}


def show(out: dict[str, Any]) -> None:
    s = out["summary"]
    print(f"\n{s['requests']} requests over {s['days']} days in {out['seconds']:.0f} s; "
          f"{s['right']} right, {s['served_by_programs']} served by programs "
          f"(usually {s['usual_program_share']:.0%} of a calm day)")
    print(f"changes of note: {s['changes']}; {s['changes_repaired_by_night']} repaired by that night; of "
          f"{s['changes_checked']} followed by a calm day, {s['changes_recovered']} were back to normal the next day")
    print(f"wrong on a day of change: {s['wrong_on_change_days']}; by programs: "
          f"{s['wrong_program_actions_on_change_days']}; wrong program actions in all: {s['wrong_program_actions']}")
    print(f"stalled days: {len(s['stalled_days'])} {s['stalled_days'][:12]}; "
          f"false alarms on calm days: {len(s['false_alarm_days'])}")
    print(f"relearned {s['relearned']} times, {s['relearning_skipped']} attempts skipped by the budget; "
          f"builder calls {s['builder_calls']}, of them {s['builder_calls_on_calm_days']} on calm days, "
          f"{s['builder_calls_per_change']} a change")
    print(f"harness ms a request, by tenth of the stream: {s['ms_by_tenth']}")
    print(f"longest night in each tenth, seconds: {s['night_s_by_tenth']}")
    print(f"store MB at the end of each tenth: {s['store_mb_by_tenth']}")


def parse(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--requests", type=int, default=100_000)
    parser.add_argument("--per-day", type=int, default=250)
    parser.add_argument("--change-every", type=int, default=2, help="days between changes of Dana's note")
    parser.add_argument("--quiet-days", type=int, default=60, help="a stretch in the middle with no change at all")
    parser.add_argument("--keep", type=int, default=2000, help="runs whose episodes the store keeps; 0 keeps all")
    parser.add_argument("--careless", type=float, default=0.2, help="share of near misses the agent acts on")
    parser.add_argument("--misread", type=float, default=0.01, help="share of near misses Jev takes for a booking")
    parser.add_argument("--lifetime-limits", action="store_true", help="count attempts over the tier's whole life")
    parser.add_argument("--executor", choices=("guarded", "subprocess", "auto"), default="guarded",
                        help="where a learned function runs: in this process, or a new process for each call")
    parser.add_argument("--weeks", type=Path, default=stream.CACHE, help="where generated weeks are kept")
    parser.add_argument("--output", type=Path, default=HERE / "runs" / "long_run.json")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    asyncio.run(main(parse()))
