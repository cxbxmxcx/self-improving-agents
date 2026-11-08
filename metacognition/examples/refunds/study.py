"""D2, the returns desk: can an agent learn safely when verdicts are scarce, late and human? (DOMAINS.md)

    python -m examples.refunds.study                                   # replay the first pass
    python -m examples.refunds.study --contract tool --controller measured --gate cases   # the second pass
    python -m examples.refunds.study --mode record --budget 4          # paid calls only for missing replies

Phases (choose with --phases):
  floors   the no-procedure reasoning floors on the development returns
  memory   the floors with the hand-written procedure, on the test returns
  online   one stream per arm: --regular returns before the holiday policy, then --holiday returns after.
           Every arm starts from the procedure as written before the holiday policy, and relearns
           with value learning (the domain-neutral prompts) and replayed practice: a candidate is
           tried only in accepted runs' recorded worlds, refunds dry-run, never in the simulator.

Arms (choose with --arms) differ only in who judges:
  checker    the store's audit judges every run as it ends, and a passing version installs itself
  reviewers  people judge --review-rate of the runs, --review-delay runs late; a person approves a
             proposed version --approve-after runs after the gate passes it
  judge      as reviewers, plus a model judge on every run, whose verdicts reach the detector

Every run is audited either way, so the summary says what each arm served wrongly and what it cost,
how long the stale procedure stayed trusted, and how often the judge agreed with the audit.

The first pass ran with the defaults. The second applies its lessons:
  --contract tool        the procedure answers in the action tools' terms ({"method": "card"}), not its
                         own words ({"action": "refund"}); the same code otherwise
  --controller measured  the arms take their reasoning settings from the floors and memory phases'
                         results (run them first; the no-procedure floors replay free from the first pass)
  --gate cases           the gate checks a candidate on the 6 held-out holiday returns, labeled, with the
                         procedure run in the simulator and the audit; the learner still only replays

Follow-ups to the second pass:
  --contract act         the procedure's description also tells the agent to act on its answer, not redo
                         its reads; with --memory-splits test stream the memory phase measures it on 76 returns
  --person-limit 1       one rejection from a person quarantines a procedure (the detector's limit stays 2
                         for every other source)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import shutil
import sys
from pathlib import Path
from typing import Any

from helix.metacognition import (
    Controller,
    Detector,
    Gate,
    Judge,
    LearnerSelection,
    LearningBudget,
    Metacognition,
    ReplayPractice,
)
from helix.metacognition.design import install_mechanism
from helix.metacognition.offline import Case, Floors, calibrate
from helix.metacognition.pricing import PRICES
from helix.metacognition.report import summarize

from examples.refunds.agent import CONTRACTS, MODEL, check, practice, returns_desk
from examples.refunds.world import ReturnsWorld, outcome, task_set
from metacog import Recording, RunConfig, Runner

HERE = Path(__file__).parent
EFFORTS = ("none", "low", "medium", "high")
ARMS = ("checker", "reviewers", "judge")
RUBRIC = ("The assistant handled a return at a store. Judge only the action it took (a refund to the card, store "
          "credit, or a decline) and its amount, against the store's returns policy exactly as the policy the "
          "assistant read states it, every note included.")


class BudgetSpent(Exception):
    pass


def guard(recording: Recording, budget: float) -> None:
    spent = recording.spent(PRICES)
    if spent > budget:
        raise BudgetSpent(f"${spent:.2f} of live calls is over the ${budget:.2f} budget")


def show(floors: Floors, title: str) -> None:
    print(f"\n{title}")
    print(f"  {'arm':<10} {'effort':<7} {'solved':>7} {'reasoning':>9} {'dollars':>8} {'seconds':>7}")
    for row in floors.rows():
        print(f"  {row['arm']:<10} {row['effort']:<7} {row['solved']:>3}/{row['total']:<3} "
              f"{row['reasoning_tokens']:>9.0f} {row['dollars']:>8.4f} {row['seconds']:>7.1f}")
    for arm in floors.sweeps:
        solid, adequate = floors.floor(arm) or "not reached", floors.floor(arm, 7 / 8) or "not reached"
        print(f"  {arm}: solid floor {solid}, adequate {adequate}")


def cases(split: str, n: int) -> list[Case]:
    return [Case(t["id"], t["text"], t) for t in task_set(split)[:n]]


async def phase_floors(args: argparse.Namespace, recording: Recording) -> None:
    floors = await calibrate(returns_desk(), cases("dev", args.tasks), verifier=check, world=ReturnsWorld,
                             efforts=args.efforts, arms=("no_memory",), concurrency=args.concurrency,
                             max_turns=12, run_config=RunConfig(max_tool_calls=24), prices=PRICES)
    show(floors, "Reasoning floors without a procedure (development returns)")
    floors.save(args.output / "floors_no_memory.json")
    guard(recording, args.budget)


async def phase_memory(args: argparse.Namespace, recording: Recording) -> None:
    contract, reference, _ = CONTRACTS[args.contract]
    meta = Metacognition([contract], verifier=check, recall_k=1, learning="inline")
    meta.install("refund_decision", reference)
    measured = [case for split in args.memory_splits for case in cases(split, args.tasks)]
    floors = await calibrate(returns_desk(metacognition=meta), measured, verifier=check,
                             world=ReturnsWorld, efforts=args.efforts, arms=("memory",),
                             concurrency=args.concurrency, max_turns=12, run_config=RunConfig(max_tool_calls=24),
                             prices=PRICES)
    show(floors, f"Reasoning floors with the hand-written procedure, contract {args.contract} "
                 f"({' and '.join(args.memory_splits)} returns)")
    floors.save(args.output / f"floors_memory_{args.contract}.json")
    guard(recording, args.budget)


# ------------------------------------------------------------------ the online loop, one arm at a time

def given(world: ReturnsWorld) -> tuple[str, float] | None:
    """What the run did with its return: (kind, amount), or None when it took no single action."""
    mine = [(kind, amount) for _, kind, amount in world.actions]
    return mine[0] if len(mine) == 1 else None


def same_actions(run: dict[str, Any], result: Any, world: Any) -> tuple[bool, str]:
    """A replayed candidate passes when it did what the accepted run did: the same actions on the same return
    by the same method, with amounts that agree to the cent (166.2 typed and 166.20 computed are one refund)."""

    def norm(key: tuple[str, str]) -> tuple[str, str, float | None]:
        args = json.loads(key[1])
        amount = args.pop("amount_usd", None)
        return key[0], json.dumps(args, sort_keys=True), None if amount is None else round(float(amount), 2)

    got, want = sorted(map(norm, world.actions)), sorted(map(norm, world.expected))
    same = len(got) == len(want) and all(
        g[:2] == w[:2] and (g[2] is None) == (w[2] is None) and (g[2] is None or abs(g[2] - w[2]) <= 0.011)
        for g, w in zip(got, want, strict=True))
    if same:
        return True, "the actions match the accepted run"
    return False, f"took {world.actions or 'no actions'} where the accepted run took {world.expected}"


def arm_metacognition(arm: str, store: Path, args: argparse.Namespace) -> Metacognition:
    verifier: Any = {"checker": check, "reviewers": None,
                     "judge": Judge(RUBRIC, params={"reasoning_effort": "medium"})}[arm]
    contract, _, regular = CONTRACTS[args.contract]
    # The gate tries a candidate on the 6 labeled holiday returns, the procedure alone in the simulator; or
    # replays the last 3 accepted runs, as the learner practices.
    gate = (Gate(cases=[t["id"] for t in task_set("gate")], practice=practice) if args.gate == "cases" else
            Gate(holdout=3))
    meta = Metacognition(
        [contract], verifier=verifier, learner=args.learner,
        writer_params={"reasoning_effort": args.writer_effort} if args.writer_effort else None,
        selection=LearnerSelection(designed=True) if args.selection else None,
        practice=ReplayPractice(compare=same_actions, ignore={"decline_return": ["reason"]}), gate=gate,
        controller=Controller({"memory": "none", "no_memory": "high"}), recall_k=1, relearn_after=4,
        learning="inline", gating_sources=("verifier", "human", "harness", *(("judge",) if arm == "judge" else ())),
        detector=Detector(limits={"human": args.person_limit} if args.person_limit else {}),
        budget=LearningBudget(attempts_per_day=4), store=str(store), prices=PRICES)
    if args.controller == "measured":  # measured floors in the store come before the hand-set table
        for name in ("floors_no_memory.json", f"floors_memory_{args.contract}.json"):
            if not (args.output / name).exists():
                raise KeyError(f"--controller measured needs {name} in {args.output}: run the floors and memory "
                               "phases first")
            Floors.load(args.output / name).record(meta)
    meta.install("refund_decision", regular)  # written before the holiday policy
    for path in args.designed:  # mechanisms a person approved for this deployment (A5)
        install_mechanism(meta, Path(path).read_text(encoding="utf-8"), Path(path).stem)
    return meta


async def run_arm(arm: str, args: argparse.Namespace, recording: Recording) -> dict[str, Any]:
    store = args.output / f"{arm}.meta"
    if store.exists():
        shutil.rmtree(store)
    meta = arm_metacognition(arm, store, args)
    agent = returns_desk(metacognition=meta)
    stream = task_set("test")[: args.regular] + task_set("stream")[: args.holiday]
    sample = random.Random(1729)  # the same runs are reviewed in every arm
    pending: list[tuple[int, str, dict[str, Any]]] = []  # (due before run, run id, audit)
    seen: dict[str, int] = {}  # proposal "name@version" -> the run after which it appeared
    rows: list[dict[str, Any]] = []
    print(f"\nArm {arm}: {args.regular} regular returns, then {args.holiday} under the holiday policy "
          f"(reasoning: {meta.setting('refund_decision', True, MODEL)} with the procedure, "
          f"{meta.setting(None, False, MODEL)} without)")
    print(f"{'#':>3} {'case':<5} {'offered':<8} {'effort':<6} {'audit':<9} {'verdict':<20} events")

    async def deliver(now: int) -> list[str]:
        """The reviews due before run `now`, and what they set off."""
        notes = []
        for due, run_id, audit in [p for p in pending if p[0] <= now]:
            pending.remove((due, run_id, audit))
            notes.extend(f"review: {event}" for event in await meta.feedback(run_id, audit["success"],
                                                                            audit["reason"]))
        return notes

    for n, task in enumerate(stream, 1):
        notes = await deliver(n) if arm != "checker" else []
        world = ReturnsWorld(task)
        result = await Runner.run(agent, task["text"], context=world, max_turns=12,
                                  run_config=RunConfig(max_tool_calls=24, case_id=task["id"], case_data=task))
        report = result.metacognition
        audit = world.check()
        if arm != "checker" and sample.random() < args.review_rate:
            pending.append((n + args.review_delay, report.run_id, audit))
        verdict = ("-" if report.verdict is None else
                   f"{'accepted' if report.verdict.accepted else 'rejected'} ({report.verdict.source})")
        action, right = given(world), outcome(task)
        error = 0.0 if action is None else round((action[1] if action[0] != "decline" else 0.0)
                                                 - (right[1] if right[0] != "decline" else 0.0), 2)
        for proposal in meta.proposals():
            seen.setdefault(f"{proposal['name']}@{proposal['version']}", n)
        for key, at in list(seen.items()):
            name, version = key.split("@")
            if at + args.approve_after <= n and meta.store.version(name, int(version))["state"] == "proposed":
                meta.approve(name, int(version))
                notes.append(f"a person approved {key}")
        events = [*notes, *report.events]
        rows.append({"n": n, "case": task["id"], "holiday": task["drift"], "offered": report.offered,
                     "effort": report.effort, "audit": audit["success"], "reason": audit["reason"],
                     "verdict": None if report.verdict is None else report.verdict.accepted,
                     "verdict_source": None if report.verdict is None else report.verdict.source,
                     "error_usd": error, "events": events})
        print(f"{n:>3} {task['id']:<5} {'yes' if report.offered else 'no':<8} {report.effort or '-':<6} "
              f"{'right' if audit['success'] else 'WRONG':<9} {verdict:<20} {'; '.join(events)}")
        guard(recording, args.budget)
    (args.output / f"online_{arm}.json").write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
    return outcome_of(arm, rows, meta)


def outcome_of(arm: str, rows: list[dict[str, Any]], meta: Metacognition) -> dict[str, Any]:
    first_holiday = next((r["n"] for r in rows if r["holiday"]), None)

    def when(text: str) -> int | None:
        return next((r["n"] for r in rows if any(text in e for e in r["events"])), None)

    wrong = [r for r in rows if not r["audit"]]
    judged = [r for r in rows if r["verdict_source"] == "judge"]
    summary = summarize(meta.store)
    return {
        "arm": arm, "runs": len(rows), "first_holiday_run": first_holiday,
        "wrong": len(wrong), "wrong_after_change": sum(r["holiday"] for r in wrong),
        "over_refunded_usd": round(sum(r["error_usd"] for r in rows if r["error_usd"] > 0), 2),
        "under_refunded_usd": round(-sum(r["error_usd"] for r in rows if r["error_usd"] < 0), 2),
        "quarantined_at": when("quarantined"), "relearned_at": when("wrote refund_decision"),
        "trusted_again_at": when("approved") or when("installed"),
        "judge_agreed": sum(r["verdict"] == r["audit"] for r in judged), "judge_runs": len(judged),
        "judge_false_accepts": sum(r["verdict"] and not r["audit"] for r in judged),
        "learning": summary["learning"], "procedures": summary["procedures"]["refund_decision"],
        "attempts": [{"mechanism": a["mechanism"], "outcome": a["outcome"], "gate": a["gate"]}
                     for a in meta.store.learning()],
    }


def compare(results: list[dict[str, Any]]) -> None:
    print("\nArms compared (run numbers count from the start of the stream)")
    print(f"  {'arm':<10} {'wrong':>6} {'over $':>8} {'under $':>8} {'quarantined':>12} {'relearned':>10} "
          f"{'trusted again':>14} {'learning $':>11}")
    for r in results:
        print(f"  {r['arm']:<10} {r['wrong']:>6} {r['over_refunded_usd']:>8.2f} {r['under_refunded_usd']:>8.2f} "
              f"{str(r['quarantined_at'] or '-'):>12} {str(r['relearned_at'] or '-'):>10} "
              f"{str(r['trusted_again_at'] or '-'):>14} {r['learning']['dollars']:>11.4f}")
    for r in results:
        if r["judge_runs"]:
            print(f"  judge: agreed with the audit on {r['judge_agreed']} of {r['judge_runs']} runs; accepted "
                  f"{r['judge_false_accepts']} wrong ones")


async def phase_online(args: argparse.Namespace, recording: Recording) -> None:
    results = [await run_arm(arm, args, recording) for arm in args.arms]
    compare(results)
    (args.output / "online_summary.json").write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")


async def main(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    recording = Recording(args.recording, mode=args.mode)
    if args.mode != "replay":
        from dotenv import load_dotenv

        load_dotenv(Path.cwd() / ".env")
    try:
        with recording.active():
            for phase in args.phases:
                await {"floors": phase_floors, "memory": phase_memory, "online": phase_online}[phase](args, recording)
    finally:
        if args.mode != "replay":
            print(f"\n{recording.live_calls} live calls, ${recording.spent(PRICES):.4f}, saved to {recording.path}")


def parse(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--phases", nargs="+", default=["floors", "memory", "online"],
                        choices=["floors", "memory", "online"])
    parser.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    parser.add_argument("--contract", default="words", choices=list(CONTRACTS),
                        help="the procedure answers in its own words, or in the action tools' terms")
    parser.add_argument("--controller", default="hand", choices=["hand", "measured"],
                        help="the hand-set settings, or the floors the floors and memory phases measured")
    parser.add_argument("--gate", default="holdout", choices=["holdout", "cases"],
                        help="replay the last 3 accepted runs, or check the 6 labeled holiday returns")
    parser.add_argument("--memory-splits", nargs="+", default=["test"], choices=["test", "stream"],
                        help="the returns the memory phase measures the procedure on")
    parser.add_argument("--person-limit", type=int, default=None,
                        help="a person's rejections that quarantine a procedure (default: the detector's limit)")
    parser.add_argument("--learner", default="value_learning", help="the arms' learner, before any tournament")
    parser.add_argument("--writer-effort", default=None, choices=["none", "low", "medium"],
                        help="the learning writer's effort (default: the harness's, medium)")
    parser.add_argument("--selection", action="store_true",
                        help="learner tournaments when learning keeps failing (A5, L2)")
    parser.add_argument("--designed", nargs="*", default=[],
                        help="designed mechanisms a person approved, to compete in the tournaments")
    parser.add_argument("--mode", default="replay", choices=["replay", "record", "live"])
    parser.add_argument("--recording", type=Path, default=HERE / "recorded" / "study.jsonl")
    parser.add_argument("--output", type=Path, default=HERE / "runs")
    parser.add_argument("--tasks", type=int, default=16, help="cases per split for the floors phases")
    parser.add_argument("--efforts", nargs="+", default=list(EFFORTS), choices=list(EFFORTS))
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--regular", type=int, default=10, help="returns before the holiday policy")
    parser.add_argument("--holiday", type=int, default=50, help="returns after it")
    parser.add_argument("--review-rate", type=float, default=0.3)
    parser.add_argument("--review-delay", type=int, default=10)
    parser.add_argument("--approve-after", type=int, default=5)
    parser.add_argument("--budget", type=float, default=2.0, help="dollars of live calls before stopping")
    return parser.parse_args(argv)


if __name__ == "__main__":
    try:
        asyncio.run(main(parse()))
    except BudgetSpent as exc:
        sys.exit(f"stopped: {exc}")
    except KeyError as exc:
        sys.exit(f"{exc.args[0]}")
