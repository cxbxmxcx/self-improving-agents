"""Listing 6.3: does metacognition pay for itself?

Each condition runs the 20 held-out requests from the same starting week,
with each request's prepared memory and nothing written back. Two
conditions never decide (always fast, always deliberate); the others let a
decision model direct the scaffold. The report sets acceptance and safety
against dollars and seconds, then checks the decision model itself.

By default the assistant is the scripted stand-in and every decision is
replayed, so no key is needed. --record refreshes the recorded decisions
with live Jev; --live runs gpt-5.4-mini, Jev, and the LLM decision model.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from helix.signals.behavior_diff import BehaviorDiffSignal

from chapter_appendices.ch06_decision_scaffold_legacy.assistant.environment import AssistantEnvironment, load_requests
from chapter_appendices.ch06_decision_scaffold_legacy.assistant.evaluation import (
    LLM_DECIDER,
    WEIGHTS,
    best_route,
    build_paired,
    calibration,
    conditions,
    estimate,
    interventions,
    measure,
    run_request,
    searched_artifacts,
    stamped_reward,
    summarize,
)
from chapter_appendices.ch06_decision_scaffold_legacy.assistant.recording import RECORDED, decision_model
from chapter_appendices.ch06_decision_scaffold_legacy.assistant.scripted import model_mode

THEATER = "Keep writing blocks productive and focused."


def deciders(mode: str, output: Path):
    """Jev and the LLM decision model: replayed, or live behind the recording."""
    folder = output if mode == "live" else RECORDED
    return (decision_model(mode, folder=folder),
            decision_model(mode, folder=folder, name=LLM_DECIDER, file="llm_decisions.jsonl"))


def regret(paired, row) -> float:
    """What the chosen route left behind against the best one in hindsight."""
    values = {r: v.value(WEIGHTS) for r, v in paired.route_results(row["request"], WEIGHTS).items()}
    return max(values.values()) - values[row["route"]] if row["route"] in values else 0.0


async def behavior(jev, live: bool) -> list[dict]:
    """Does a recalled lesson change what the assistant does, or only what it says?"""
    env = AssistantEnvironment()
    held_out = [r for r in load_requests(split="test") if r["category"] == "memory"]
    theater = {**held_out[0], "memory": [{"kind": "lesson", "user_message": held_out[0]["text"],
                                          "lesson": THEATER}]}
    diff = BehaviorDiffSignal(reward_fn=stamped_reward)
    rows = []
    for request, label in [(r, r["id"]) for r in held_out] + [(theater, f"{held_out[0]['id']} theater")]:
        without = await run_request(request, env=env, decider=jev, live=live, memory=False)
        with_lesson = await run_request(request, env=env, decider=jev, live=live)
        m = await diff.compare(without, with_lesson)
        rows.append({"request": label, "divergence": m.score, "flat": m.triggered, "feedback": m.feedback,
                     "outcomes": m.metadata["rewards"]})
    return rows


async def evaluate(mode: str, output: Path, *, trials: int = 1, searched: Path | None = None) -> dict:
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    live = mode == "live"
    jev, llm = deciders(mode, output)
    held_out = load_requests(split="test")
    found = searched_artifacts(searched or RECORDED / "searched.json")
    report = {"mode": mode, "requests": len(held_out), "trials": trials, "weights": vars(WEIGHTS), "conditions": {}}
    with model_mode(live):
        paired = await build_paired(held_out, decider=jev, live=live, path=output / "paired_test.json")
        for name, condition in conditions(jev, llm=llm, searched=found).items():
            rows = []
            for _ in range(trials):
                env = AssistantEnvironment()
                for request in held_out:
                    trajectory = await run_request(request, env=env, decider=condition["decider"], live=live,
                                                   artifacts=condition["artifacts"], pins=condition["pins"])
                    row = await measure(env, request, trajectory)
                    row["regret"] = regret(paired, row)
                    rows.append(row)
            report["conditions"][name] = {"summary": summarize(rows), "rows": rows,
                                          "regret": sum(r["regret"] for r in rows) / len(rows)}
        directed = {k: v for k, v in report["conditions"].items() if not k.startswith("always")}
        report["calibration"] = {k: calibration(v["rows"], paired) for k, v in directed.items()}
        report["interventions"] = {k: interventions(v["rows"], paired) for k, v in directed.items()}
        report["hindsight"] = {r["id"]: best_route(paired, r["id"]) for r in held_out}
        report["behavior"] = await behavior(jev, live)
    print_report(report)
    (output / "report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print("Report: " + str(output / "report.json"))
    return report


def print_report(report: dict) -> None:
    n = report["requests"] * report["trials"]
    print(f"Held-out requests: {report['requests']} x {report['trials']} trial(s); mode: {report['mode']}")
    print(f"{'condition':24} {'accepted':>8} {'unsafe':>6} {'score':>6} {'$/request':>10} {'s/request':>9} "
          f"{'decisions':>9} {'regret':>6}")
    for name, c in report["conditions"].items():
        s = c["summary"]
        print(f"{name:24} {s['accepted']:>5}/{n:<2} {s['unsafe']:>6} {s['score']:>6.3f} {s['dollars']:>10.5f} "
              f"{s['seconds']:>9.2f} {s['decisions']:>9.1f} {c['regret']:>6.3f}")
    for name, cal in report["calibration"].items():
        parts = [f"{q} Brier {v['brier']:.3f}" + (f", AUROC {v['auroc']:.2f}" if v["auroc"] is not None else "")
                 for q, v in cal.items() if v["brier"] is not None]
        print(f"Calibration, {name}: " + "; ".join(parts))
    for name, moves in report["interventions"].items():
        parts = []
        for move in ("ask", "deliberate", "confirm"):
            v = moves[move]
            p = "-" if v["precision"] is None else f"{v['precision']:.2f}"
            r = "-" if v["recall"] is None else f"{v['recall']:.2f}"
            parts.append(f"{move} {v['chosen']} (precision {p}, recall {r})")
        rev = moves["revise"]
        parts.append(f"revised {rev['revised']}, accepted after {rev['accepted_after']}")
        print(f"Interventions, {name}: " + "; ".join(parts))
    for row in report["behavior"]:
        verdict = "reflection theater" if row["flat"] else "behavior changed"
        print(f"Lesson, {row['request']}: {verdict} (divergence {row['divergence']:.2f}); {row['feedback']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--record", action="store_true", help="Refresh the recorded decisions with live Jev and "
                                                             "the LLM decision model (scripted assistant).")
    group.add_argument("--live", action="store_true", help="Run gpt-5.4-mini, Jev, and the LLM decision model live.")
    parser.add_argument("--trials", type=int, default=1, help="Repeat each condition to show spread (live runs).")
    parser.add_argument("--searched", type=Path, help="The searched configuration from listing 6.4.")
    parser.add_argument("--output", type=Path, help="Directory for this run's evidence.")
    parser.add_argument("--yes", action="store_true", help="With --live, start without stopping at the cost estimate.")
    args = parser.parse_args()
    mode = "live" if args.live else "record" if args.record else "replay"
    if mode == "live" and not args.yes:
        print(estimate(70 + 5 * 20 * args.trials + 8))
        return
    if mode != "replay":
        from helix.env import load_env
        load_env()
    tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6]
    output = args.output or Path(__file__).parent / "runs/evals" / tag
    asyncio.run(evaluate(mode, output, trials=args.trials, searched=args.searched))


if __name__ == "__main__":
    main()
