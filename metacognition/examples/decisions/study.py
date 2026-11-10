"""Decision programs on D1 and D3: requests served with no language model, Jev answering the judgments.

    python -m examples.decisions.study --domain d3                       # replay the committed Jev recording
    python -m examples.decisions.study --domain d3 --mode record         # paid Jev calls only for missing answers

For each held-out request the program either serves it (an action, judged by the domain's own verifier) or
hands it off to the agent. The report gives how many it kept, how many of those were right, and what each
request cost in Jev dollars and seconds. A handed-off request costs what the agent costs; the dashboard adds
that from the LLM studies. --wordings runs the varied-wording split (wordings.py) instead of the templated one.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from helix.deciders import recorded_decider
from helix.metacognition.decide import JEV_MODEL, serve

HERE = Path(__file__).parent
ROOT = HERE.parent


def domain(name: str):
    if name == "d1":
        from examples.expenses.decide import desk
        from examples.expenses.world import task_set

        procedure = (ROOT / "expenses" / "memories" / "learned_by_gpt_5_4_mini.py").read_text(encoding="utf-8")
        return desk(procedure), task_set, ROOT / "expenses" / "programs" / "file_claim.py"
    if name == "d3":
        from examples.invoices.decide import desk
        from examples.invoices.world import task_set

        return desk(), task_set, ROOT / "invoices" / "programs" / "ap_desk.py"
    if name == "d5":
        from examples.shifts.decide import desk
        from examples.shifts.world import task_set

        return desk(), task_set, None
    raise SystemExit(f"no domain {name!r}")


async def main(args: argparse.Namespace) -> dict:
    desk, task_set, default_program = domain(args.domain)
    program = (args.program or default_program).read_text(encoding="utf-8")
    if args.mode != "replay":
        import os

        from dotenv import load_dotenv

        load_dotenv(Path.cwd() / ".env")
        os.environ.setdefault("TYPESAFE_API_KEY", os.environ.get("JEV_API_KEY", ""))
    jev = recorded_decider(args.jev_recording, mode=args.mode, model=JEV_MODEL)
    if args.wordings:
        from examples.decisions.wordings import load

        tasks = load(args.domain, args.split)
    else:
        tasks = task_set(args.split)[: args.tasks]
    served = []
    print(f"\n{args.domain} {args.split}: {len(tasks)} requests, program {(args.program or default_program).name}")
    if not args.quiet:
        print(f"{'case':<20} {'kind':<9} {'outcome':<10} {'jev':>4} {'$':>9} {'s':>6}  reason")
    for task in tasks:
        row = await serve(program, task, desk, jev)
        served.append(row)
        outcome = ("right" if row.right else "WRONG") if row.handled else "handoff"
        if not args.quiet:
            print(f"{row.case_id:<20} {task.get('kind', '-'):<9} {outcome:<10} {row.decider_calls:>4} {row.dollars:>9.6f} "
                  f"{row.seconds:>6.2f}  {row.reason[:70]}")
    handled = [s for s in served if s.handled]
    summary = {"domain": args.domain, "split": args.split, "requests": len(served), "handled": len(handled),
               "right": sum(bool(s.right) for s in handled), "handoffs": len(served) - len(handled),
               "dollars_per_request": sum(s.dollars for s in served) / len(served),
               "seconds_per_request": sum(s.seconds for s in served) / len(served),
               "jev_calls": sum(s.decider_calls for s in served), "rows": [s.to_dict() for s in served]}
    print(f"\nkept {summary['handled']} of {summary['requests']}, right {summary['right']} of {summary['handled']}, "
          f"handed off {summary['handoffs']}; ${summary['dollars_per_request']:.6f} and "
          f"{summary['seconds_per_request']:.2f} s a request")
    if args.wordings:
        summary["by_style"] = by_style(tasks, served)
        for style, row in summary["by_style"].items():
            print(f"  {style:<13} {row['requests']:>3} requests: kept {row['kept']:>3}, right {row['right']:>3}, "
                  f"wrong {row['wrong']:>3}, handed off {row['handoffs']:>3}")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / f"decide_{args.domain}_{args.split}.json").write_text(json.dumps(summary, indent=2),
                                                                         encoding="utf-8")
    if args.mode != "replay":
        print(f"{jev.live_usage().calls} live Jev calls, ${jev.live_usage().dollars:.6f}, saved to {args.jev_recording}")
    return summary


def by_style(tasks: list[dict], served: list) -> dict[str, dict[str, int]]:
    """Kept, right, wrong and handed off, for each wording style."""
    out: dict[str, dict[str, int]] = {}
    for task, row in zip(tasks, served, strict=True):
        cell = out.setdefault(task["style"], {"requests": 0, "kept": 0, "right": 0, "wrong": 0, "handoffs": 0})
        cell["requests"] += 1
        cell["kept"] += int(row.handled)
        cell["right"] += int(row.handled and bool(row.right))
        cell["wrong"] += int(row.handled and not row.right)
        cell["handoffs"] += int(not row.handled)
    return out


def parse(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--domain", default="d3", choices=["d1", "d3"])
    parser.add_argument("--split", default="test")
    parser.add_argument("--tasks", type=int, default=16)
    parser.add_argument("--program", type=Path, default=None, help="a decision program's source file")
    parser.add_argument("--wordings", action="store_true", help="use the varied-wording split (wordings.json)")
    parser.add_argument("--quiet", action="store_true", help="leave out the per-request table")
    parser.add_argument("--mode", default="replay", choices=["replay", "record", "live"])
    parser.add_argument("--jev-recording", type=Path, default=HERE / "recorded" / "jev.jsonl")
    parser.add_argument("--output", type=Path, default=HERE / "runs")
    return parser.parse_args(argv)


if __name__ == "__main__":
    try:
        asyncio.run(main(parse()))
    except KeyError as exc:
        sys.exit(f"{exc.args[0]}")
