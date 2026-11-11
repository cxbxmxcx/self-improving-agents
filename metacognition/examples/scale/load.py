"""Serve many requests at once through one harness and one store, and check it stays right.

    python -m examples.scale.load

Every run before this one served a request at a time. Here scripted models answer after a delay that stands for a
real model's time, and a verdict arrives a moment after its action, so requests overlap as they would on a busy
server. The script answers three questions.

    throughput   requests a second at each level of concurrency, and what the harness itself can carry
    sameness     the same requests get the same outcomes however many are served at once
    a change     Dana's note changes under load: how many wrong answers get out before the switch and the detector
                 act, and whether one relearning starts or several

A harness can only react to a verdict it has. Every request already past its last check when the first wrong
verdicts arrive is exposed, so what a change costs grows with the number in flight. `--verdict-seconds 0` shows the
same change with verdicts that arrive at once, which no real server has.

Learning runs in the background here, as a server would run it, so serving does not wait for the builder.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any

from chapters.ch07.assistant.desk import check
from chapters.ch07.assistant.harness import code_search, decision_tier, harness, night, seed
from helix.metacognition import Gate, LearningBudget, MetaStore, RunConfig, Runner
from helix.metacognition.exceptions import AgentsException
from helix.metacognition.gap import outcome_of

from examples.scale import stream
from examples.scale.scripted import NOW, Jev, Models, World, function_for

HERE = Path(__file__).parent


async def ready(args: argparse.Namespace, *, agent_seconds: float, jev_seconds: float, tag: str,
                weeks: stream.Weeks, verdict_seconds: float = 0.0, store: MetaStore | None = None,
                watched: bool = True) -> tuple[Any, Any, Any, Models, Jev]:
    """An assistant whose first night is behind it: a program is installed and serving. With `watched` off no
    verdict can switch an action off or quarantine a function, which is a learned function with no loop around it."""
    NOW["conditions"] = "ordinary"
    jev = Jev(seconds=0.0)
    models = Models(careless=0.0, agent_seconds=0.0, builder_seconds=0.0)
    pace = {"verdict": 0.0}

    async def verdict(ctx: Any, result: Any) -> Any:
        """The chapter's check, after the time an action and its check take."""
        if pace["verdict"]:
            await asyncio.sleep(pace["verdict"])
        return check(ctx, result)

    tier = decision_tier(jev, [seed("component_llm")], search=code_search("dgm"), long_running=True)
    if not watched:
        tier.gating_sources = ()
    agent, meta = harness(tier, store=store or MetaStore(keep=2000), learning="background", experience_limit=8,
                          budget=LearningBudget(attempts_per_day=3, runs_per_day=250), executor=args.executor,
                          verifier=verdict, **({} if watched else {"gating_sources": ()}))
    meta.install("find_shared_starts", function_for("ordinary"), created_by="scripted")
    first = stream.Day(1, "ordinary")
    with models.active():
        for n in range(16):  # a short first day, one at a time, then the night that installs a program
            task = stream.request(weeks.get("ordinary", n % stream.SERVED_WEEKS), f"{tag}0-{n}", near=n % 4 == 3)
            await Runner.run(agent, task["text"], context=World(task), max_turns=16,
                             run_config=RunConfig(max_tool_calls=24, case_id=task["id"], case_data=task))
        tier.labeled, tier.search_cases = stream.held_back(first, weeks)
        await night(meta)
    assert meta.programs.current() is not None, "the first night installed no program"
    jev.seconds, models.agent_seconds, models.builder_seconds = jev_seconds, agent_seconds, 5 * agent_seconds
    pace["verdict"] = verdict_seconds
    return agent, meta, tier, models, jev


async def serve(agent: Any, tasks: list[dict[str, Any]], at_once: int) -> dict[str, Any]:
    """Serve `tasks` with at most `at_once` in flight, and say what happened and how long it took."""
    slots = asyncio.Semaphore(at_once)
    order: list[str] = []

    async def one(task: dict[str, Any]) -> dict[str, Any]:
        async with slots:
            started = time.perf_counter()
            try:
                result = await Runner.run(agent, task["text"], context=World(task), max_turns=16,
                                          run_config=RunConfig(max_tool_calls=24, case_id=task["id"], case_data=task))
            except AgentsException as exc:
                return {"id": task["id"], "outcome": "failed", "right": False, "seconds": 0.0, "events": [str(exc)]}
            report = result.metacognition
            right = bool(report.verdict and report.verdict.accepted)
            outcome = outcome_of(right, program=report.program, should_act=task["expect"] == "act")
            order.append(outcome)
            return {"id": task["id"], "outcome": outcome, "right": right, "seconds": time.perf_counter() - started,
                    "events": list(report.events)}

    started = time.perf_counter()
    rows = await asyncio.gather(*(one(task) for task in tasks))
    wall = time.perf_counter() - started
    took = sorted(row["seconds"] for row in rows)
    return {"at_once": at_once, "requests": len(rows), "wall_s": round(wall, 2),
            "a_second": round(len(rows) / wall, 1), "right": sum(r["right"] for r in rows),
            "outcomes": dict(Counter(r["outcome"] for r in rows)),
            "mean_s": round(statistics.fmean(took), 3), "p95_s": round(took[int(0.95 * (len(took) - 1))], 3),
            "rows": rows, "order": order}


def batch(weeks: stream.Weeks, conditions: str, count: int, tag: str) -> list[dict[str, Any]]:
    """The same weeks and the same mix at every level, with a reference of its own."""
    return [stream.request(weeks.get(conditions, (100 + n) % stream.SERVED_WEEKS), f"{tag}-{n}", near=n % 4 == 3)
            for n in range(count)]


async def throughput(args: argparse.Namespace, weeks: stream.Weeks) -> list[dict[str, Any]]:
    """Requests a second at each level: with model time as a server would see it, and with none."""
    out = []
    for label, agent_seconds, jev_seconds, verdict_seconds in (
            ("with model time", args.agent_seconds, args.jev_seconds, args.verdict_seconds),
            ("harness only", 0.0, 0.0, 0.0)):
        for at_once in args.levels:
            agent, meta, _, models, _ = await ready(args, agent_seconds=agent_seconds, jev_seconds=jev_seconds,
                                                    verdict_seconds=verdict_seconds, tag=f"t{at_once}{label[0]}",
                                                    weeks=weeks)
            count = _count(args, at_once) if agent_seconds else args.requests
            with models.active():
                result = await serve(agent, batch(weeks, "ordinary", count, f"t{at_once}{label[0]}"), at_once)
                await meta.wait_for_learning()
            result.pop("rows")
            result.pop("order")
            out.append({"timing": label, **result})
            print(f"{label:<16} {at_once:>4} at once: {result['requests']:>4} requests in {result['wall_s']:>6.1f} s, "
                  f"{result['a_second']:>6.1f} a second, mean {result['mean_s']:.2f} s, p95 {result['p95_s']:.2f} s, "
                  f"{result['right']} right, {result['outcomes']}", flush=True)
    return out


async def a_change(args: argparse.Namespace, weeks: stream.Weeks) -> list[dict[str, Any]]:
    """The note changes while requests are in flight. Wrong actions before the harness reacts, by level."""
    out = []
    for at_once, watched in [*((level, True) for level in args.levels), (args.levels[-1], False)]:
        tag = f"c{at_once}" if watched else f"u{at_once}"
        agent, meta, _, models, _ = await ready(args, agent_seconds=args.agent_seconds, jev_seconds=args.jev_seconds,
                                                verdict_seconds=args.verdict_seconds, tag=tag, weeks=weeks,
                                                watched=watched)
        NOW["conditions"] = "changed"
        meta.gate = Gate(cases=stream.FUNCTION_WEEKS["changed"])
        before = models.calls["builder: function"]
        with models.active():
            result = await serve(agent, batch(weeks, "changed", _count(args, at_once), tag), at_once)
            await meta.wait_for_learning()
        events = [event for row in result["rows"] for event in row["events"]]
        learning = meta.store.learning("find_shared_starts")
        row = {"at_once": at_once, "watched": watched, "requests": result["requests"], "right": result["right"],
               "wrong_program_actions": result["outcomes"].get("wrong", 0),
               "wrong_by_agent": result["outcomes"].get("llm_wrong", 0),
               "switched_off": sum(e.startswith("program: switched off") for e in events),
               "quarantined": sum(e.startswith("detector: quarantined") for e in events),
               "relearnings_started": sum(not str(a["outcome"]).startswith("skipped") for a in learning),
               "relearnings_skipped": sum(str(a["outcome"]).startswith("skipped") for a in learning),
               "function_writer_calls": models.calls["builder: function"] - before,
               "trusted_at_the_end": meta.store.trusted("find_shared_starts") is not None,
               "wall_s": result["wall_s"]}
        out.append(row)
        print(f"change {'under load' if watched else 'with no loop'}, {at_once:>4} at once: "
              f"{row['right']}/{row['requests']} right, "
              f"{row['wrong_program_actions']} wrong by programs, {row['wrong_by_agent']} by the agent, "
              f"quarantined {row['quarantined']}, relearnings {row['relearnings_started']} "
              f"({row['function_writer_calls']} writer calls), trusted at the end {row['trusted_at_the_end']}",
              flush=True)
    return out


async def main(args: argparse.Namespace) -> dict[str, Any]:
    weeks = stream.Weeks(args.weeks)
    out = {"settings": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
           "throughput": await throughput(args, weeks), "change": await a_change(args, weeks)}
    weeks.save()
    with_time = [r for r in out["throughput"] if r["timing"] == "with model time"]
    out["same_outcomes_at_every_level"] = len({json.dumps(_scaled(r), sort_keys=True) for r in with_time}) == 1
    print(f"same share right and same mix of outcomes at every level: {out['same_outcomes_at_every_level']}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


def _count(args: argparse.Namespace, at_once: int) -> int:
    """A batch served one request at a time waits for every model call, so it is a fifth as long."""
    return args.requests if at_once > 1 else max(40, args.requests // 5)


def _scaled(result: dict[str, Any]) -> dict[str, float]:
    """A batch's outcomes as shares, so batches of different sizes can be compared."""
    return {name: round(count / result["requests"], 3) for name, count in result["outcomes"].items()}


def parse(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--requests", type=int, default=400, help="requests in each batch")
    parser.add_argument("--levels", type=int, nargs="+", default=[1, 8, 32, 128], help="requests served at once")
    parser.add_argument("--agent-seconds", type=float, default=0.5, help="what one call to the agent model takes")
    parser.add_argument("--jev-seconds", type=float, default=0.1, help="what one call to the decision model takes")
    parser.add_argument("--verdict-seconds", type=float, default=0.2,
                        help="what an action and its check take before the harness has a verdict")
    parser.add_argument("--executor", choices=("guarded", "subprocess", "auto"), default="guarded",
                        help="where a learned function runs: in this process, or a new process for each call")
    parser.add_argument("--weeks", type=Path, default=stream.CACHE, help="where generated weeks are kept")
    parser.add_argument("--output", type=Path, default=HERE / "runs" / "load.json")
    return parser.parse_args(argv)


if __name__ == "__main__":
    asyncio.run(main(parse()))
