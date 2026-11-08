"""Learn decision programs from the agent's runs: the learning problem, and one mechanism's attempt at it.

    python -m examples.decisions.learn --domain d3 --mode record

The agent's judged runs on the development wordings are the experience (examples/decisions/llm.py, replayed for
free). A mechanism writes a decision program from them and practices it on those requests, with Jev answering.
The program it returns is scored on the test wordings, which it never saw: an action must be right, an
out-of-scope request must be handed off. A hand-off of work it could have done earns little and isn't accepted.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

import helix
from helix.deciders import Decider, recorded_decider
from helix.memory.learning_mechanisms import Contract
from helix.memory.mechanism_runtime import MechanismBudget
from helix.metacognition import GENERIC
from helix.metacognition.decide import JEV_MODEL, KIT_GUIDE, RETURNS, RETURNS_ROUTED, UnsafeProgram, serve
from helix.metacognition.pricing import PRICES
from helix.metacognition.selection import LearningProblem, run_tournament
from helix.metacognition.sources import SourceLearner

from examples.decisions import llm, study
from examples.decisions.wordings import load
from metacog import Recording

HERE = Path(__file__).parent
SEEDS = Path(helix.__file__).parent / "metacognition" / "programs"
WRITER = "gpt-5.4-mini"
HANDOFF_REWARD = 0.25  # better than a wrong action, far from serving the request

JOBS = {
    "d1": """Serve one request to a travel expense assistant. Its only job is filing the expense claim for one trip.
Anything else, such as a claim's status, cancelling a claim or a question about policy, must be handed off.
Read tools: read_trip(trip_id), read_policy(). Procedure: reimbursable_total(trip_id), answering
{"total_usd": float}. Action: file_claim, with arguments {"trip_id": str, "amount_usd": float}.""",
    "d3": """Serve one request to an accounts payable desk. Its jobs are scheduling the payment of an invoice already
in the ledger, and entering a vendor invoice document into the ledger. Anything else, such as an invoice's status,
cancelling or disputing, or a question about a vendor, must be handed off.
Read tools: read_document(doc_id), read_invoice(invoice_id), read_vendor(vendor). Procedures:
amount_payable(invoice_id), answering {"invoice_id": str, "amount_usd": float, "pay_on": "YYYY-MM-DD"}, and
parse_invoice(doc_id), answering a record for the ledger as a dict. Actions: schedule_payment, with the arguments
{"invoice_id": str, "amount_usd": float, "pay_on": str}, and record_invoice, whose arguments are the parser's record.""",
}


def contract(domain: str, routes: bool = False) -> Contract:
    return Contract(name="decide", signature="async def decide(request, kit) -> dict", description=JOBS[domain],
                    api=KIT_GUIDE, returns=RETURNS_ROUTED if routes else RETURNS)


def practice_of(tasks: dict[str, dict[str, Any]], desk: Any, jev: Decider) -> Any:
    async def practice(source: str, case_id: str) -> tuple[float, str, str]:
        try:
            row = await serve(source, tasks[case_id], desk, jev)
        except UnsafeProgram as exc:
            return 0.0, f"the guard refused the program: {exc}", ""
        if row.right:
            return 1.0, row.reason, ""
        if not row.handled:
            return HANDOFF_REWARD, f"it was handed off, though it could be served: {row.reason}", ""
        return 0.0, row.reason, ""

    return practice


async def problem(domain: str, jev: Decider, split: str = "dev") -> LearningProblem:
    """The learning problem: the agent's runs on `split` to learn from, and the test wordings to be scored on."""
    with Recording(HERE / "recorded" / "llm.jsonl").active():
        experience = await llm.collect(domain, split)
    order = {task["id"]: n for n, task in enumerate(load(domain, split))}
    episodes = sorted(experience["episodes"], key=lambda e: order[e.task_id])  # runs finish in no fixed order
    desk, _, _ = study.domain(domain)
    dev = {t["id"]: t for t in load(domain, split)}
    test = {t["id"]: t for t in load(domain, "test")}
    return LearningProblem(procedure=domain, contract=contract(domain), episodes=episodes,
                           practice=practice_of(dev, desk, jev), cases=list(test), score=practice_of(test, desk, jev))


async def main(args: argparse.Namespace) -> dict[str, Any]:
    if args.mode != "replay":
        import os

        from dotenv import load_dotenv

        load_dotenv(Path.cwd() / ".env")
        os.environ.setdefault("TYPESAFE_API_KEY", os.environ.get("JEV_API_KEY", ""))
    jev = recorded_decider(args.jev_recording, mode=args.mode, model=JEV_MODEL)
    learning = await problem(args.domain, jev, args.experience)
    source = (SEEDS / f"{args.mechanism}.py").read_text(encoding="utf-8")
    budget = MechanismBudget(writer_calls=8, practice_calls=1000, seconds=1800)
    learner = SourceLearner(source, args.mechanism, GENERIC, budget)
    recording = Recording(args.recording, mode=args.mode)
    with recording.active():
        choice = await run_tournament(learning, [learner], writer_model=WRITER,
                                      writer_params={"reasoning_effort": "medium"}, prices=PRICES)
    trial = choice.winner
    print(f"{args.domain}: {trial.mechanism} accepted {trial.accepted} of {trial.total} held-out requests; "
          f"{trial.writer_calls} writer calls, ${trial.dollars:.4f}" + (f"; {trial.note}" if trial.note else ""))
    for failure in trial.failures[:8]:
        print("  ", failure[:170])
    if trial.source is None:
        return {"trial": trial.to_dict()}
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / f"{args.domain}_{args.mechanism}.py"
    path.write_text(trial.source, encoding="utf-8", newline="\n")
    summary = await study.main(study.parse(["--domain", args.domain, "--wordings", "--quiet", "--program", str(path),
                                            "--mode", args.mode, "--jev-recording", str(args.jev_recording),
                                            "--output", str(args.output)]))
    (args.output / f"{args.domain}_{args.mechanism}.json").write_text(json.dumps(
        {"trial": trial.to_dict(), "by_style": summary["by_style"], "handled": summary["handled"],
         "right": summary["right"], "handoffs": summary["handoffs"], "requests": summary["requests"],
         "dollars_per_request": summary["dollars_per_request"], "seconds_per_request": summary["seconds_per_request"]},
        indent=2), encoding="utf-8")
    if args.mode != "replay":
        print(f"writer: {recording.live_calls} live calls, ${recording.spent(PRICES):.4f}; "
              f"Jev: {jev.live_usage().calls} live calls, ${jev.live_usage().dollars:.4f}")
    return {"trial": trial.to_dict(), "summary": summary}


def parse(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--domain", default="d3", choices=["d1", "d3"])
    parser.add_argument("--mechanism", default="program_synthesis")
    parser.add_argument("--experience", default="dev", choices=["dev", "dev+near"],
                        help="the agent's runs to learn from; dev+near adds near misses")
    parser.add_argument("--mode", default="replay", choices=["replay", "record", "live"])
    parser.add_argument("--recording", type=Path, default=HERE / "recorded" / "learn.jsonl")
    parser.add_argument("--jev-recording", type=Path, default=HERE / "recorded" / "jev.jsonl")
    parser.add_argument("--output", type=Path, default=HERE / "learned")
    return parser.parse_args(argv)


if __name__ == "__main__":
    asyncio.run(main(parse()))
