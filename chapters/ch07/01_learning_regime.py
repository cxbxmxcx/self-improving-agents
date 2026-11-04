"""Measure chapter 6's learning mechanisms under harder learning regimes.

Chapter 6's mechanisms learned from 32 runs at medium and high effort with
the writer at medium, and four of five matched every validation case. A
search for better mechanisms needs room to improve, so this listing runs
the same mechanisms, now held as source, with less experience, a cheaper
writer, and a writer-call budget, on both meta-training classes.

A regime leaves headroom when the best mechanism averages at most 12 of 16
validation matches across the two classes. Replay uses saved replies
without an API key, while --record adds paid calls for missing replies.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import asdict, dataclass
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chapters.ch06.assistant.harness import PRICES
from chapters.ch06.assistant.learning import (
    CLASSES,
    act,
    behavior_cases,
    gather_experience,
    match_signal,
    recall_experience,
    remember,
)
from chapters.ch06.assistant.recording import Recording
from chapters.ch07.mechanisms import CONTROL, SEEDS, seed_source
from helix.memory.learning_mechanisms import Writer
from helix.memory.mechanism_runtime import (
    Lab,
    MechanismBudget,
    MechanismStopped,
    run_mechanism,
)
from helix.signals.cost import call_dollars

HERE = Path(__file__).parent
CH06 = REPO_ROOT / "chapters" / "ch06" / "assistant" / "recorded"
RECORDED = HERE / "recorded" / "learning_regime.jsonl"
# Where each class's runs without memory were recorded in chapter 6.
EXPERIENCE = {
    "multi_attendee": CH06 / "reasoning_floor.jsonl",
    "same_time": CH06 / "generalization.jsonl",
}
WRITER_MODEL = "gpt-5.4-mini"
HEADROOM = 12  # best mean validation match, out of 16, that leaves room


@dataclass(frozen=True)
class Regime:
    """How much experience a mechanism gets and what it may spend."""

    efforts: tuple[str, ...]  # the agent's practice runs it learns from
    tasks: int  # development requests experienced, from the first
    writer_effort: str
    writer_calls: int


REGIMES = {
    "low_runs": Regime(("low",), 16, "medium", 12),
    "low_writer": Regime(("low",), 16, "low", 12),
    "few_runs": Regime(("low",), 8, "low", 8),
}


async def experience(task_class: str, regime: Regime) -> list:
    """The regime's episodes, replayed from chapter 6's recordings."""
    with Recording(EXPERIENCE[task_class]).active():
        episodes = await gather_experience(
            efforts=regime.efforts, task_class=task_class
        )
    dev = [t["id"] for t in CLASSES[task_class].task_set("dev")]
    kept = set(dev[: regime.tasks])
    tier = await remember([e for e in episodes if e.task_id in kept])
    recalled = await recall_experience(tier)
    tier.close()
    return recalled


async def measure(name: str, task_class: str, regime: Regime,
                  episodes: list) -> dict:
    """One mechanism learns under the regime; its function is compared."""
    writer = Writer(
        model=WRITER_MODEL,
        params={"reasoning_effort": regime.writer_effort},
    )
    lab = Lab(
        episodes,
        act,
        CLASSES[task_class].contract,
        writer,
        MechanismBudget(writer_calls=regime.writer_calls, seconds=1800),
    )
    try:
        learned = await run_mechanism(seed_source(name), lab, name=name)
    except MechanismStopped as stopped:
        # A design that stops learns nothing; its spending still counts.
        return {"matched": 0, "writer_calls": stopped.writer_calls,
                "dollars": dollars(stopped.usage),
                "stopped": stopped.reason}
    signal = match_signal(task_class)
    match = await signal.measure(
        learned.source,
        ground_truth=behavior_cases("validation", task_class),
    )
    return {"matched": match.metadata["matched"],
            "writer_calls": learned.writer_calls,
            "dollars": dollars(learned.usage),
            "source": learned.source}


def dollars(usage: list) -> float:
    return round(sum(call_dollars(WRITER_MODEL, u, PRICES) or 0.0
                     for u in usage), 4)


async def main(args: argparse.Namespace) -> None:
    recording = Recording(
        args.recording, mode="record" if args.record else "replay"
    )
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / "learning_regime.json"
    report = (json.loads(report_path.read_text(encoding="utf-8"))
              if report_path.exists() else {})
    names = [*SEEDS, CONTROL]
    for regime_name in args.regimes:
        regime = REGIMES[regime_name]
        results: dict[str, dict] = {}
        for task_class in CLASSES:
            episodes = await experience(task_class, regime)
            solved = sum(e.success for e in episodes)
            print(f"\n{regime_name}, {task_class}: {len(episodes)} runs,"
                  f" {solved} accepted, writer at"
                  f" {regime.writer_effort}, {regime.writer_calls} calls")
            print(f"  {'mechanism':<24} {'match':>6} {'calls':>6}"
                  f" {'dollars':>8}")
            results[task_class] = {}
            for name in names:
                spent = recording.spent(PRICES)
                if spent > args.budget:
                    sys.exit(f"Budget: ${spent:.2f} spent of"
                             f" ${args.budget:.2f}; stopped before"
                             f" {name}.")
                with recording.active():
                    row = await measure(name, task_class, regime,
                                        episodes)
                results[task_class][name] = row
                note = f"  stopped: {row['stopped']}" \
                    if "stopped" in row else ""
                print(f"  {name:<24} {row['matched']:>3}/16"
                      f" {row['writer_calls']:>6}"
                      f" {row['dollars']:>8.4f}{note}")
        means = {n: sum(results[c][n]["matched"] for c in CLASSES)
                 / len(CLASSES) for n in SEEDS}
        best = max(means, key=lambda n: means[n])
        room = means[best] <= HEADROOM
        print(f"  best: {best}, mean {means[best]:.1f}/16;"
              f" {'headroom' if room else 'no headroom'}")
        report[regime_name] = {"regime": asdict(regime),
                               "results": results,
                               "best": best, "best_mean": means[best],
                               "headroom": room}
        # Saved per regime, so an interrupted sweep keeps what finished.
        report_path.write_text(json.dumps(report, indent=2),
                               encoding="utf-8")
    if args.record:
        print(f"\n{recording.live_calls} live writer calls added to"
              f" {recording.path.name}, ${recording.spent(PRICES):.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--regimes", nargs="+", choices=list(REGIMES),
                        default=list(REGIMES))
    parser.add_argument("--record", action="store_true",
                        help="Call gpt-5.4-mini for calls the recording"
                             " lacks.")
    parser.add_argument("--recording", type=Path, default=RECORDED)
    parser.add_argument("--budget", type=float, default=6.0,
                        help="Dollars of live writer calls after which no"
                             " further mechanism starts.")
    parser.add_argument("--output", type=Path,
                        default=HERE / "runs" / "learning_regime")
    args = parser.parse_args()
    if args.record:
        from helix.env import load_env

        load_env()
    try:
        asyncio.run(main(args))
    except KeyError as exc:
        sys.exit(f"{exc.args[0]}. Record the missing replies with"
                 " --record (paid calls to gpt-5.4-mini).")
