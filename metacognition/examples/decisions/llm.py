"""The LLM agent on the varied wordings: the baseline a decision program must beat, and the runs it learns from.

    python -m examples.decisions.llm --domain d3 --split dev --mode record --budget 1

The agent has its trusted procedures and runs at no reasoning (the floor both domains reached with a procedure).
Each run is judged by the domain's check; an out-of-scope request is right only if the agent took no action.
Nothing is quarantined here, so every run sees the same procedures.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from helix.metacognition import Controller, Metacognition, Router
from helix.metacognition.pricing import PRICES

from examples.decisions.wordings import load
from metacog import Recording, RunConfig, Runner

HERE = Path(__file__).parent
MODEL = "gpt-5.4-mini"


def _judged(check: Any, acted: Any) -> Any:
    def verifier(ctx: Any, result: Any) -> dict[str, Any]:
        world = ctx.context
        if world.style in ("out_of_scope", "near_miss"):
            took = acted(world)
            return {"success": not took, "reason": "took no action" if not took else "acted on an out-of-scope request"}
        return check(ctx, result)

    return verifier


def _setup(domain: str, model: str, *, programs: Any = None, stale_parser: bool = False) -> tuple[Any, Any, Any]:
    """The agent with its trusted procedures, and the factory for a request's world. `programs` adds the decision
    tier; `stale_parser` installs D3's parser from before Kestrel's layout change, for a drift stream."""
    controller = Controller({"memory": "none", "no_memory": "none"})
    if domain == "d1":
        from examples.expenses.agent import assistant, check, reimbursable_total
        from examples.expenses.world import ExpenseWorld

        source = (HERE.parent / "expenses" / "memories" / "learned_by_gpt_5_4_mini.py").read_text(encoding="utf-8")
        meta = Metacognition([reimbursable_total], verifier=_judged(check, lambda w: w.claims), controller=controller,
                             recall_k=1, learning="inline", relearn_after=10**9, gating_sources=(), prices=PRICES, programs=programs)
        meta.install("reimbursable_total", source, created_by="value_learning")
        return assistant(model, metacognition=meta), meta, ExpenseWorld
    from examples.invoices.agent import CONTRACTS, PARSER, PARSER_BEFORE, PAYABLE, ap_desk, check
    from examples.invoices.world import InvoiceWorld

    meta = Metacognition(CONTRACTS["act"], verifier=_judged(check, lambda w: w.actions), controller=controller,
                         router=Router(k=2), recall_k=2, learning="inline", relearn_after=10**9, gating_sources=(),
                         prices=PRICES, programs=programs)
    meta.install("parse_invoice", PARSER_BEFORE if stale_parser else PARSER)
    meta.install("amount_payable", PAYABLE)
    return ap_desk(model, metacognition=meta), meta, InvoiceWorld


async def collect(domain: str, split: str, *, model: str = MODEL, concurrency: int = 6) -> dict[str, Any]:
    """Run the agent on a split's requests inside the active recording: the per-run rows and the judged episodes."""
    agent, meta, world_of = _setup(domain, model)
    tasks = load(domain, split)
    gate = asyncio.Semaphore(concurrency)

    async def one(task: dict[str, Any]) -> dict[str, Any]:
        world = world_of(task)
        world.style = task["style"]
        async with gate:
            result = await Runner.run(agent, task["text"], context=world, max_turns=12,
                                      run_config=RunConfig(max_tool_calls=24, case_id=task["id"], case_data=task))
        report = result.metacognition
        return {"id": task["id"], "style": task["style"], "right": bool(report.verdict.accepted),
                "reason": report.verdict.reason, "offered": bool(report.offered), "effort": report.effort,
                "dollars": float(result.cost.dollars or 0.0), "seconds": float(result.cost.seconds or 0.0)}

    rows = await asyncio.gather(*(one(task) for task in tasks))
    styles: dict[str, dict[str, float]] = {}
    for row in rows:
        cell = styles.setdefault(row["style"], {"requests": 0, "right": 0, "dollars": 0.0, "seconds": 0.0})
        cell["requests"] += 1
        cell["right"] += int(row["right"])
        cell["dollars"] += row["dollars"]
        cell["seconds"] += row["seconds"]
    return {"domain": domain, "split": split, "model": model, "requests": len(rows),
            "right": sum(r["right"] for r in rows), "dollars_per_request": sum(r["dollars"] for r in rows) / len(rows),
            "seconds_per_request": sum(r["seconds"] for r in rows) / len(rows), "by_style": styles, "rows": rows,
            "episodes": meta.store.episodes()}


async def main(args: argparse.Namespace) -> dict[str, Any]:
    if args.mode != "replay":
        from dotenv import load_dotenv

        load_dotenv(Path.cwd() / ".env")
    recording = Recording(args.recording, mode=args.mode)
    try:
        with recording.active():
            summary = await collect(args.domain, args.split, model=args.model)
    finally:
        if args.mode != "replay":
            print(f"{recording.live_calls} live calls, ${recording.spent(PRICES):.4f}, saved to {recording.path}")
    print(f"{args.domain} {args.split}, {args.model} with procedures at no reasoning: {summary['right']} of "
          f"{summary['requests']} right; ${summary['dollars_per_request']:.4f} and "
          f"{summary['seconds_per_request']:.2f} s a request")
    for style, cell in summary["by_style"].items():
        print(f"  {style:<13} {cell['right']:>3}/{cell['requests']:<3} ${cell['dollars'] / cell['requests']:.4f} "
              f"{cell['seconds'] / cell['requests']:.2f} s")
    args.output.mkdir(parents=True, exist_ok=True)
    slim = {key: value for key, value in summary.items() if key != "episodes"}
    (args.output / f"llm_{args.domain}_{args.split}.json").write_text(json.dumps(slim, indent=2), encoding="utf-8")
    return summary


def parse(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--domain", default="d3", choices=["d1", "d3"])
    parser.add_argument("--split", default="test", choices=["dev", "dev+near", "test", "stress", "stress2"])
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--mode", default="replay", choices=["replay", "record", "live"])
    parser.add_argument("--recording", type=Path, default=HERE / "recorded" / "llm.jsonl")
    parser.add_argument("--output", type=Path, default=HERE / "runs")
    return parser.parse_args(argv)


if __name__ == "__main__":
    asyncio.run(main(parse()))
