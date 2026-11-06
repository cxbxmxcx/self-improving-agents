"""D3, invoice intake: how does metacognition behave with several procedures and several customers? (DOMAINS.md)

    python -m examples.invoices.study --phases routing                 # free: no model calls
    python -m examples.invoices.study --mode record --budget 3         # paid calls only for missing replies

Phases (choose with --phases):
  floors    the no-procedure reasoning floors on the development requests, by kind of work
  memory    the floors with the hand-written procedures, on the test requests
  routing   calibrate_routing from labeled requests: thresholds, leave-one-out accuracy, and accuracy on the test
            requests (free)
  discover  serve the development requests with no procedure declared, then metacog.discover: is the parsing
            work found before anyone declares it?
  online    one stream per arm: --requests mixed requests from three clients; from request 21 Kestrel bills
            in its European layout, which the parser written before it misreads without failing

Arms (choose with --arms):
  routed      one Metacognition shared by the clients: calibrated routing, blame on the procedure a run called
  unrouted    the same, but every request is offered both procedures (no thresholds) and every procedure
              offered is blamed, as the chapter 6 study counted with its single procedure
  per_client  PerUser: a Metacognition per client, each with its own parser and payable, routed and called

Every arm starts from the parser written before the layout change and, with --payable reference, the
hand-written payable (each client's unwritten convention included); with --payable learn the payable starts
untrusted and is learned from the runs. A relearned version must pass a labeled gate: Kestrel's new layout and
each other vendor's (V801-V806), and labeled payments (V811-V828, each client's own under per_client), compared
with their labels by the gate's own practice, at a bar of 1.0.

The learner defaults to skill synthesis. A parser must cover four layouts and a payable three unwritten
conventions, and one run shows one layout or convention. Synthesis writes from every accepted run at once (so
it sees every layout served since the quarantine), then repairs the candidate on practice's failures, where the
audit names each rule, the unwritten ones included. Value learning writes each candidate from a single run, and
reflection from one run revised by lessons, so neither sees the layouts together.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from helix.memory.base import MemoryContext

from examples.invoices.agent import (
    PARSER,
    PARSER_BEFORE,
    PAYABLE,
    PROCEDURES,
    READS,
    ap_desk,
    check,
    labeled_practice,
    practice,
)
from examples.invoices.world import CLIENT_ORDER, InvoiceWorld, task_by_id, task_set
from metacog import ModelSettings, Reasoning, Recording, RunConfig, Runner
from metacog.discover import discover, skeleton
from metacog.meta import GENERIC, Controller, Gate, LearningBudget, Metacognition, PerUser, Router
from metacog.meta.report import summarize
from metacog.offline import Case, Floors, calibrate, calibrate_routing
from metacog.pricing import PRICES

HERE = Path(__file__).parent
EFFORTS = ("none", "low", "medium", "high")
ARMS = ("routed", "unrouted", "per_client")
LEARNERS = ("skill_synthesis", "reflection", "value_learning")
LABEL = {"intake": "parse_invoice", "payment": "amount_payable", "question": None}
OWN = {"intake": ["parse_invoice"], "payment": ["amount_payable"], "question": []}


class BudgetSpent(Exception):
    pass


def guard(recording: Recording, budget: float) -> None:
    spent = recording.spent(PRICES)
    if spent > budget:
        raise BudgetSpent(f"${spent:.2f} of live calls is over the ${budget:.2f} budget")


def cases(split: str, n: int) -> list[Case]:
    return [Case(t["id"], t["text"], t) for t in task_set(split)[:n]]


def by_kind(floors: Floors) -> dict[str, dict[str, str]]:
    """Solved by kind of work at each setting, from each run's record."""
    out: dict[str, dict[str, str]] = {}
    for arm, sweep_ in floors.sweeps.items():
        for effort, cell in sweep_.cells.items():
            totals: Counter[str] = Counter()
            solved: Counter[str] = Counter()
            for run in cell.runs:
                kind = task_by_id(run.task_id.split("#")[0])["kind"]
                totals[kind] += 1
                solved[kind] += run.solved
            out.setdefault(arm, {})[effort] = ", ".join(f"{k} {solved[k]}/{totals[k]}" for k in sorted(totals))
    return out


def show(floors: Floors, title: str) -> None:
    print(f"\n{title}")
    print(f"  {'arm':<10} {'effort':<7} {'solved':>7} {'reasoning':>9} {'dollars':>8} {'seconds':>7}  by kind")
    kinds = by_kind(floors)
    for row in floors.rows():
        print(f"  {row['arm']:<10} {row['effort']:<7} {row['solved']:>3}/{row['total']:<3} "
              f"{row['reasoning_tokens']:>9.0f} {row['dollars']:>8.4f} {row['seconds']:>7.1f}  "
              f"{kinds[row['arm']][row['effort']]}")
    for arm in floors.sweeps:
        solid, adequate = floors.floor(arm) or "not reached", floors.floor(arm, 7 / 8) or "not reached"
        print(f"  {arm}: solid floor {solid}, adequate {adequate}")


# ------------------------------------------------------------------ routing

def labeled(split: str) -> list[tuple[str, str | None]]:
    return [(t["text"], LABEL[t["kind"]]) for t in task_set(split)]


def routing_calibration() -> Any:
    """Thresholds and example requests from the labeled routing requests (deterministic, free)."""
    return calibrate_routing(PROCEDURES, labeled("routing"))


def routed_right(router: Router, split: str) -> tuple[int, int]:
    descriptions = {p.name: p.description for p in PROCEDURES}
    tasks = task_set(split)
    right = sum(router.match(t["text"], descriptions)[0] == OWN[t["kind"]] for t in tasks)
    return right, len(tasks)


async def phase_routing(args: argparse.Namespace, recording: Recording) -> None:
    calibration = routing_calibration()
    right, total = routed_right(calibration.router, "test")
    print("\nRouting calibrated from 48 labeled requests (intake, payment, and vendor questions in no class)")
    for name, info in calibration.procedures.items():
        print(f"  {name:<15} threshold {info['threshold']:.3f}")
    print(f"  leave-one-out accuracy {calibration.accuracy:.3f}; test requests routed to their own class: "
          f"{right} of {total}")
    unrouted_right, _ = routed_right(Router(k=2), "test")
    print(f"  without thresholds (k=2): {unrouted_right} of {total} (every request is offered both procedures)")
    (args.output / "routing.json").write_text(json.dumps(
        {"router": calibration.router.to_dict(), "procedures": calibration.procedures,
         "accuracy": calibration.accuracy, "test": [right, total]}, indent=2, default=str), encoding="utf-8")


# ------------------------------------------------------------------ floors and memory

async def phase_floors(args: argparse.Namespace, recording: Recording) -> None:
    floors = await calibrate(ap_desk(), cases("dev", args.tasks), verifier=check, world=InvoiceWorld,
                             efforts=args.efforts, arms=("no_memory",), concurrency=args.concurrency,
                             max_turns=12, run_config=RunConfig(max_tool_calls=24), prices=PRICES)
    show(floors, "Reasoning floors without a procedure (development requests)")
    floors.save(args.output / "floors_no_memory.json")
    guard(recording, args.budget)


async def phase_memory(args: argparse.Namespace, recording: Recording) -> None:
    meta = Metacognition(PROCEDURES, verifier=check, router=routing_calibration().router, learning="inline")
    meta.install("parse_invoice", PARSER)
    meta.install("amount_payable", PAYABLE)
    floors = await calibrate(ap_desk(metacognition=meta), cases("test", args.tasks), verifier=check,
                             world=InvoiceWorld, efforts=args.efforts, arms=("memory",),
                             concurrency=args.concurrency, max_turns=12, run_config=RunConfig(max_tool_calls=24),
                             prices=PRICES)
    show(floors, "Reasoning floors with the hand-written procedures (test requests)")
    floors.save(args.output / "floors_memory.json")
    guard(recording, args.budget)


# ------------------------------------------------------------------ discover

async def phase_discover(args: argparse.Namespace, recording: Recording) -> dict[str, Any]:
    store = args.output / "discover.meta"
    if store.exists():
        shutil.rmtree(store)
    meta = Metacognition([], verifier=check, learning="worker", store=str(store), prices=PRICES)
    agent = ap_desk(metacognition=meta)
    settings = ModelSettings(reasoning=Reasoning(args.discover_effort))
    for task in task_set("dev")[: args.tasks]:
        await Runner.run(agent, task["text"], context=InvoiceWorld(task), max_turns=12,
                         run_config=RunConfig(max_tool_calls=24, case_id=task["id"], case_data=task,
                                              model_settings=settings))
        guard(recording, args.budget)
    groups = discover(meta.store, min_runs=2)
    print(f"\nWork found in {meta.store.count()} runs with no procedure, ranked by reasoning spent")
    for rank, group in enumerate(groups, 1):
        print(f"  {rank}. {', '.join(group.tools)}: {group.runs} runs, {group.accepted} accepted, "
              f"{group.mean_reasoning_tokens:.0f} reasoning tokens a run, ${group.dollars:.4f}")
    intake = next((rank for rank, g in enumerate(groups, 1)
                   if {"read_document", "record_invoice"} <= set(g.tools) and "schedule_payment" not in g.tools), None)
    print(f"  the parsing work (read_document, record_invoice) ranks {intake or 'nowhere'}")
    if groups:
        print("  the top group's skeleton:\n" + "\n".join(
            "    " + line for line in skeleton(groups[0], read_only=[t.name for t in READS]).splitlines()))
    found = {"groups": [g.to_dict() for g in groups], "parsing_rank": intake}
    (args.output / "discover.json").write_text(json.dumps(found, indent=2), encoding="utf-8")
    return found


# ------------------------------------------------------------------ the online loop, one arm at a time

def gate_cases(client: str | None) -> list[str]:
    """Labeled cases: the parser's (Kestrel's new layout and the other vendors'), and the payments of `client`
    (every client's when shared)."""
    payments = [t["id"] for t in task_set("payable_gate") if client is None or t["client"] == client]
    return [*[t["id"] for t in task_set("gate")], *payments]


def build(arm: str, args: argparse.Namespace, store: Path, client: str | None) -> Metacognition:
    router = Router(k=2) if arm == "unrouted" else routing_calibration().router
    meta = Metacognition(
        PROCEDURES, verifier=check, learner=args.learner, prompts=GENERIC, practice=practice,
        gate=Gate(cases=gate_cases(client), practice=labeled_practice, bar=1.0), router=router,
        attribution="offered" if arm == "unrouted" else "called",
        controller=Controller({"memory": "none", "no_memory": "medium"}), relearn_after=4, learning="inline",
        budget=LearningBudget(attempts_per_day=4), store=str(store), prices=PRICES, executor=args.executor)
    meta.install("parse_invoice", PARSER_BEFORE)  # written before Kestrel's European layout
    if args.payable == "reference":
        meta.install("amount_payable", PAYABLE)
    return meta


def scope_for(arm: str, args: argparse.Namespace, root: Path) -> Metacognition | PerUser:
    if arm == "per_client":
        return PerUser(lambda client: build(arm, args, root / f"{client}.meta", client))
    return build(arm, args, root / "shared.meta", None)


def metas(scope: Metacognition | PerUser) -> dict[str, Metacognition]:
    return dict(scope.by_user) if isinstance(scope, PerUser) else {"shared": scope}


async def run_arm(arm: str, args: argparse.Namespace, recording: Recording) -> dict[str, Any]:
    root = args.output / arm
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    scope = scope_for(arm, args, root)
    agent = ap_desk(metacognition=scope)
    stream = task_set("stream")[: args.requests]
    rows: list[dict[str, Any]] = []
    print(f"\nArm {arm}: {len(stream)} requests from {', '.join(CLIENT_ORDER)}; Kestrel's European layout from "
          f"request 21; payable {args.payable}, learner {args.learner}")
    print(f"{'#':>3} {'case':<5} {'kind':<8} {'client':<8} {'offered':<34} {'effort':<6} {'audit':<6} events")
    for n, task in enumerate(stream, 1):
        world = InvoiceWorld(task)
        config = RunConfig(max_tool_calls=24, case_id=task["id"], case_data=task,
                           memory_context=MemoryContext(user_id=task["client"]))
        result = await Runner.run(agent, task["text"], context=world, max_turns=12, run_config=config)
        report = result.metacognition
        audit = world.check(result.final_output)
        rows.append({"n": n, "case": task["id"], "kind": task["kind"], "client": task["client"],
                     "vendor": task["vendor"], "drift": task["drift"], "matched": report.matched,
                     "offered": report.offered, "called": report.called, "effort": report.effort,
                     "audit": audit["success"], "reason": audit["reason"], "events": list(report.events)})
        print(f"{n:>3} {task['id']:<5} {task['kind']:<8} {task['client']:<8} {', '.join(report.offered) or '-':<34} "
              f"{report.effort or '-':<6} {'right' if audit['success'] else 'WRONG':<6} {'; '.join(report.events)}")
        guard(recording, args.budget)
    (args.output / f"online_{arm}.json").write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
    return outcome_of(arm, rows, scope)


def outcome_of(arm: str, rows: list[dict[str, Any]], scope: Metacognition | PerUser) -> dict[str, Any]:
    def when(*texts: str) -> int | None:
        return next((r["n"] for r in rows if any(all(t in e for t in texts) for e in r["events"])), None)

    def count(*texts: str) -> int:
        return sum(all(t in e for t in texts) for r in rows for e in r["events"])

    wrong = [r for r in rows if not r["audit"]]
    attempts, learning = [], {"attempts": 0, "writer_calls": 0, "dollars": 0.0}
    procedures = {}
    for who, meta in metas(scope).items():
        attempts += [{"scope": who, "procedure": a["name"], "outcome": a["outcome"], "gate": a["gate"]}
                     for a in meta.store.learning()]
        summary = summarize(meta.store)
        for key in learning:
            learning[key] += summary["learning"][key]
        procedures[who] = summary["procedures"]
    return {
        "arm": arm, "runs": len(rows),
        "wrong": len(wrong), "wrong_by_kind": dict(Counter(r["kind"] for r in wrong)),
        "wrong_by_client": dict(Counter(r["client"] for r in wrong)),
        "wrong_after_drift_kestrel_intake": sum(r["drift"] and r["vendor"] == "KES" and r["kind"] == "intake"
                                                for r in wrong),
        "routed_to_own": sum(r["matched"] == OWN[r["kind"]] for r in rows),
        "parser_quarantines": count("quarantined parse_invoice"),
        "parser_quarantined_at": when("quarantined parse_invoice"),
        "payable_quarantines": count("quarantined amount_payable"),
        "payable_quarantined_at": when("quarantined amount_payable"),
        "parser_trusted_again_at": when("wrote parse_invoice", "installed"),
        "payable_learned_at": when("wrote amount_payable", "installed"),
        "learning": learning, "attempts": attempts, "procedures": procedures,
    }


def compare(results: list[dict[str, Any]]) -> None:
    print("\nArms compared (run numbers count from the start of the stream)")
    print(f"  {'arm':<11} {'wrong':>5} {'by kind':<34} {'routed':>7} {'parser q':>9} {'payable q':>10} "
          f"{'parser back':>12} {'learning $':>11}")
    for r in results:
        kinds = ", ".join(f"{k} {v}" for k, v in sorted(r["wrong_by_kind"].items())) or "-"
        print(f"  {r['arm']:<11} {r['wrong']:>5} {kinds:<34} {r['routed_to_own']:>3}/{r['runs']:<3} "
              f"{r['parser_quarantines']:>9} {r['payable_quarantines']:>10} "
              f"{str(r['parser_trusted_again_at'] or '-'):>12} "
              f"{r['learning']['dollars']:>11.4f}")


async def phase_online(args: argparse.Namespace, recording: Recording) -> None:
    results = [await run_arm(arm, args, recording) for arm in args.arms]
    compare(results)
    (args.output / "online_summary.json").write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")


PHASES = {"floors": phase_floors, "memory": phase_memory, "routing": phase_routing, "discover": phase_discover,
          "online": phase_online}


async def main(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    recording = Recording(args.recording, mode=args.mode)
    if args.mode != "replay":
        from dotenv import load_dotenv

        load_dotenv(Path.cwd() / ".env")
    try:
        with recording.active():
            for phase in args.phases:
                await PHASES[phase](args, recording)
    finally:
        if args.mode != "replay":
            print(f"\n{recording.live_calls} live calls, ${recording.spent(PRICES):.4f}, saved to {recording.path}")


def parse(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--phases", nargs="+", default=list(PHASES), choices=list(PHASES))
    parser.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    parser.add_argument("--payable", default="reference", choices=["reference", "learn"],
                        help="start from the hand-written payable, or learn it from the runs")
    parser.add_argument("--learner", default="skill_synthesis", choices=list(LEARNERS))
    parser.add_argument("--executor", default="auto", choices=["auto", "guarded", "subprocess"])
    parser.add_argument("--mode", default="replay", choices=["replay", "record", "live"])
    parser.add_argument("--recording", type=Path, default=HERE / "recorded" / "study.jsonl")
    parser.add_argument("--output", type=Path, default=HERE / "runs")
    parser.add_argument("--tasks", type=int, default=16, help="requests per split for floors, memory and discover")
    parser.add_argument("--efforts", nargs="+", default=list(EFFORTS), choices=list(EFFORTS))
    parser.add_argument("--discover-effort", default="medium", choices=list(EFFORTS))
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--requests", type=int, default=60, help="requests in the online stream")
    parser.add_argument("--budget", type=float, default=3.0, help="dollars of live calls before stopping")
    return parser.parse_args(argv)


if __name__ == "__main__":
    try:
        asyncio.run(main(parse()))
    except BudgetSpent as exc:
        sys.exit(f"stopped: {exc}")
