"""One agent from high reasoning to Jev: a learned procedure lowers its reasoning, and a search then replaces the LLM.

    python -m examples.decisions.continuous --mode record --budget 3
    python -m examples.decisions.continuous --domain d6 --mode record --budget 9

D1 or D6, with nothing given in advance but the agent's tools. The same held-out requests are measured at each stage, and
they teach nothing:

    start       no procedure, at high reasoning, with no reasoning as a control
    procedure   the agent has served development requests at high reasoning and the loop has learned the procedure.
                The reasoning floor is measured on validation trips, and the agent runs at it
    program     the agent has served more requests at that floor, the cost gap has opened, and a search has rewritten
                the LLM-only learning component. Whatever the gate admitted serves the requests
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from helix.deciders import recorded_decider
from helix.metacognition import Controller, Gate, Metacognition
from helix.metacognition.decide import JEV_MODEL
from helix.metacognition.exceptions import AgentsException
from helix.metacognition.offline import Case, calibrate
from helix.metacognition.parts import Verdict
from helix.metacognition.pricing import PRICES

from examples.decisions import llm
from examples.decisions import loop as decision_loop
from examples.decisions.wordings import load
from examples.learning.study import BudgetSpent, CappedRecording, Spend
from metacog import RunConfig, Runner

HERE = Path(__file__).parent
ACTING = ("buried", "casual", "distractor", "shorthand")
BAR = 7 / 8  # the share of validation trips a reasoning setting must get right


@dataclass(frozen=True)
class Domain:
    """What the pipeline needs from a domain: its procedure to learn, world, agent, and how a candidate is tried."""

    key: str
    name: str
    procedure: Any
    make_world: Any
    assistant: Any
    check: Any
    practice: Any
    acted: Any
    task_set: Any

    def world_for(self, task: dict[str, Any]) -> Any:
        world = self.make_world(task)
        world.style = task.get("style", "plain")
        return world

    async def on_base(self, source: str, case_id: str) -> tuple[float, str, str]:
        """Try a procedure on the case behind a worded request: its job is the answer, however the request was put."""
        return await self.practice(source, case_id.split("~")[0])


def domain_of(key: str) -> Domain:
    if key == "d1":
        from examples.expenses.agent import assistant, check, practice, reimbursable_total
        from examples.expenses.world import ExpenseWorld, task_set

        return Domain(key, "reimbursable_total", reimbursable_total, ExpenseWorld, assistant, check, practice,
                      lambda world: world.claims, task_set)
    from examples.billing.agent import assistant, check, credit_due, practice
    from examples.billing.world import BillingWorld, task_set

    return Domain(key, "credit_due", credit_due, BillingWorld, assistant, check, practice, lambda world: world.credits, task_set)


def build(args: argparse.Namespace, jev: Any, domain: Domain) -> tuple[Any, Metacognition]:
    """The agent at its starting reasoning, with no procedure, a learner for one, and the decision tier."""
    flags = ["--domain", domain.key, "--mode", args.mode, "--mechanism", "component_llm", "--search-method", "dgm",
             "--search-target", "mechanism", "--population", str(args.population), "--generations", "1", "--seed",
             str(args.seed), "--label", args.label, "--max-wrong", "0", "--stop-when-admitted", "--designer-model",
             args.designer_model, "--writer-model", args.writer_model, "--archive", str(args.archive), "--output",
             str(args.output)]
    tier = decision_loop.tier_for(domain.key, jev, decision_loop.parse(flags))
    tier.world = domain.world_for
    meta = Metacognition([domain.procedure], verifier=llm._judged(domain.check, domain.acted),
                         controller=Controller({"memory": args.start, "no_memory": args.start}), practice=domain.on_base,
                         gate=Gate(cases=[t["id"] for t in domain.task_set("validation")]), recall_k=1, learning="inline",
                         relearn_after=args.learn_after, gating_sources=(), prices=PRICES, programs=tier,
                         writer_model=args.writer_model)
    return domain.assistant(llm.MODEL, metacognition=meta), meta


async def frozen_phase(name: str, tasks: list[dict[str, Any]], agent: Any, meta: Metacognition, effort: str,
                       rows: list[dict[str, Any]], *, world: Any, concurrency: int = 6) -> None:
    """The agent on `tasks` at `effort`, offered what it trusts now, with nothing recorded and nothing learned."""
    slots = asyncio.Semaphore(concurrency)

    def row(task: dict[str, Any], right: bool, dollars: float, seconds: float, reason: str = "") -> dict[str, Any]:
        return {"phase": name, "id": task["id"], "style": task["style"], "kind": task.get("kind"), "tier": "llm",
                "handed_off": False, "read_only": False, "routed": False, "memory": None, "right": right, "reply": None,
                "dollars": dollars, "seconds": seconds, "effort": effort, "reason": reason}

    async def one(task: dict[str, Any]) -> dict[str, Any]:
        context = world(task)
        context.style = task.get("style", "plain")
        config = RunConfig(metacognition_mode="frozen", effort=effort, max_tool_calls=24, case_id=task["id"],
                           case_data=task)
        async with slots:
            try:
                result = await Runner.run(agent, task["text"], context=context, max_turns=12, run_config=config)
            except AgentsException:
                return row(task, False, 0.0, 0.0)
        verdict = Verdict.coerce(meta.verifier(result.context_wrapper, result))
        return row(task, bool(verdict and verdict.accepted), float(result.cost.dollars or 0.0),
                   float(result.cost.seconds or 0.0), verdict.reason if verdict else "no verdict")

    rows.extend(await asyncio.gather(*(one(task) for task in tasks)))


async def measure_floor(agent: Any, meta: Metacognition, start: str, trips: list[dict[str, Any]],
                        efforts: tuple[str, ...], world: Any) -> tuple[str, list[dict[str, Any]]]:
    """The lowest reasoning at which the agent, with its learned procedure, clears the bar; else where it started."""
    floors = await calibrate(agent, [Case(t["id"], t["text"], t) for t in trips], verifier=meta.verifier, world=world,
                             efforts=efforts, arms=("memory",), concurrency=4, max_turns=12,
                             run_config=RunConfig(max_tool_calls=24), prices=PRICES)
    return floors.floor("memory", BAR) or start, floors.rows()


def show_stages(rows: list[dict[str, Any]]) -> None:
    """Each held-out stage: what was right, and what a request cost."""
    for name in dict.fromkeys(r["phase"] for r in rows if r["phase"] != "experience"):
        part = [r for r in rows if r["phase"] == name]
        near = [r for r in part if r["style"] == "near_miss"]
        acting = [r for r in part if r["style"] in ACTING]
        print(f"{name:<26} right {sum(r['right'] for r in part)}/{len(part)}; near misses "
              f"{sum(r['right'] for r in near)}/{len(near)}; acting {sum(r['right'] for r in acting)}/{len(acting)}, "
              f"{sum(r['tier'] == 'program' and r['right'] for r in acting)} with no LLM; "
              f"${sum(r['dollars'] for r in part) / len(part):.5f} and {sum(r['seconds'] for r in part) / len(part):.2f} s")


async def main(args: argparse.Namespace) -> None:
    if args.mode != "replay":
        from dotenv import load_dotenv

        load_dotenv(Path.cwd() / ".env")
        os.environ.setdefault("TYPESAFE_API_KEY", os.environ.get("JEV_API_KEY", ""))
    args.output.mkdir(parents=True, exist_ok=True)
    jev = recorded_decider(args.jev_recording, mode=args.mode, model=JEV_MODEL)
    domain = domain_of(args.domain)
    agent, meta = build(args, jev, domain)
    stream = load(domain.key, "dev+near")
    random.Random(11).shuffle(stream)
    _, held = decision_loop.split_test(domain.key)
    held_out = [*held, *load(domain.key, "stress2")]
    random.Random(12).shuffle(held_out)
    recording = CappedRecording(args.recording, mode=args.mode, spend=Spend(args.budget))
    rows: list[dict[str, Any]] = []
    measured: dict[str, Any] = {}
    try:
        with recording.active():
            await frozen_phase(f"start: {args.start}, no procedure", held_out, agent, meta, args.start, rows, world=domain.world_for)
            await frozen_phase("start: none, no procedure", held_out, agent, meta, "none", rows, world=domain.world_for)
            show_stages(rows)
            for task in stream:
                await decision_loop.phase("experience", [task], agent, meta, domain.world_for, rows, quiet=args.quiet)
                if not measured and meta.store.trusted(domain.name) is not None:
                    floor, sweep = await measure_floor(agent, meta, args.start, domain.task_set("validation")[:args.floor_trips],
                                                       tuple(args.floor_efforts), domain.world_for)
                    measured.update(floor=floor, sweep=sweep)
                    print(f"floor with the learned procedure: {floor}; "
                          f"{[(r['effort'], r['solved'], r['total']) for r in sweep]}")
                    meta.controller = Controller({"memory": floor, "no_memory": args.start})
                    await frozen_phase(f"procedure: {floor}", held_out, agent, meta, floor, rows, world=domain.world_for)
                    show_stages(rows[-len(held_out):])
            if meta.programs.current() is not None:
                meta.programs.paused = True  # the held-out requests measure the loop; they must not teach it
                await decision_loop.phase("program", held_out, agent, meta, domain.world_for, rows, quiet=args.quiet)
                meta.programs.paused = False
            else:
                print("the search installed no program, so there is no program stage")
    finally:
        print()
        show_stages(rows)
        (args.output / f"continuous_{domain.key}.json").write_text(json.dumps(
            {"rows": rows, "floor": measured, "learning": {"procedure": meta.store.learning(domain.name),
                                                          "program": meta.store.learning(decision_loop.DECISION)},
             "signal": meta.programs.signal(), "search": meta.programs.search_log}, indent=1, default=str), encoding="utf-8")
        current = meta.programs.current()
        if current is not None:
            (args.output / f"continuous_{domain.key}_program.py").write_text(current["source"], encoding="utf-8", newline="\n")
            (args.output / f"continuous_{domain.key}_memories.json").write_text(json.dumps(meta.programs._stored(), indent=1),
                                                                                encoding="utf-8")
        trusted = meta.store.trusted(domain.name)
        if trusted is not None:
            (args.output / f"continuous_{domain.key}_procedure.py").write_text(trusted["source"], encoding="utf-8", newline="\n")
        for entry in meta.store.learning(domain.name) + meta.store.learning(decision_loop.DECISION):
            print(f"  learning: {entry['name'][:18]:<18} {entry['mechanism'][:24]:<24} {entry['outcome']:<20} "
                  f"${entry['dollars'] or 0:.4f}")
        if args.mode != "replay":
            print(f"models: {recording.live_calls} live calls, ${recording.spent(PRICES):.4f}; Jev: "
                  f"{jev.live_usage().calls} live calls, ${jev.live_usage().dollars:.4f}")


def parse(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--domain", default="d1", choices=["d1", "d6"])
    parser.add_argument("--start", default="high", choices=["low", "medium", "high"], help="the reasoning it starts at")
    parser.add_argument("--learn-after", type=int, default=32, help="runs before a learning attempt, and between attempts")
    parser.add_argument("--floor-trips", type=int, default=8, help="validation trips the floor is measured on")
    parser.add_argument("--floor-efforts", nargs="+", default=["none", "low", "medium"])
    parser.add_argument("--designer-model", default=llm.BUILDER, help="the model that rewrites the code in a search")
    parser.add_argument("--writer-model", default=llm.BUILDER, help="the model that writes procedures and programs")
    parser.add_argument("--population", type=int, default=4, help="rewrites a search may try")
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--label", default="continuous")
    parser.add_argument("--mode", default="replay", choices=["replay", "record", "live"])
    parser.add_argument("--recording", type=Path, default=HERE / "recorded" / "llm.jsonl")
    parser.add_argument("--jev-recording", type=Path, default=HERE / "recorded" / "jev.jsonl")
    parser.add_argument("--archive", type=Path, default=Path(":memory:"), help="a path keeps the search's history")
    parser.add_argument("--output", type=Path, default=HERE / "learned")
    parser.add_argument("--budget", type=float, default=3.0, help="dollars of live model calls; it stops before passing this")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    try:
        asyncio.run(main(parse()))
    except BudgetSpent as exc:
        sys.exit(f"stopped: {exc}")
