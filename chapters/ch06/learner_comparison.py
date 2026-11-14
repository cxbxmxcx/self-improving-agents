"""Search for a mechanism that learns the reference function's behavior.

Recall development episodes from medium- and high-effort runs without memory.
Five mechanisms produce candidate functions, and a contract-only writer
provides the control. Compare complete outputs on validation cases and keep
the first mechanism with the highest match score.

The selected function receives a separate test comparison and validation
agent sweep. Replay uses saved replies without an API key, while --record
adds paid calls for missing replies. Review the results before --promote,
which writes a local source file without enforcing a score threshold.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chapters.ch06.assistant.generator import task_set
from chapters.ch06.assistant.harness import (
    STARTING_MEMORY,
    MEMORIES,
    PRICES,
    install_memory,
    run_task,
)
from chapters.ch06.assistant.learning import (
    CONTRACT,
    act,
    behavior_cases,
    gather_experience,
    match_signal,
    recall_experience,
    remember,
)
from chapters.ch06.assistant.recording import Recording
from helix.archive import SQLiteArchive
from helix.artifact import ArtifactKind, genesis
from helix.eval.effort import ADEQUATE, SOLID, sweep
from helix.memory.learning_mechanisms import CONTROL, MECHANISMS, Writer
from helix.search.candidates import CandidateSearch

HERE = Path(__file__).parent
RECORDED = HERE / "assistant" / "recorded" / "learning_search.jsonl"
FLOOR_RECORDED = HERE / "assistant" / "recorded" / "reasoning_floor.jsonl"
WRITER = {
    "model": "gpt-5.4-mini",
    "params": {"reasoning_effort": "medium"},
}


class BudgetSpent(Exception):
    """The search spent its budget; mechanisms measured so far count."""


def mechanism_config(name: str) -> dict:
    return {"mechanism": name, "writer": WRITER}


async def learn(name: str, episodes: list) -> tuple:
    mechanism = MECHANISMS.get(name) or CONTROL
    writer = Writer(model=WRITER["model"], params=dict(WRITER["params"]))
    learned = await mechanism.learn(episodes, act, writer, CONTRACT)
    return learned


async def main(args: argparse.Namespace) -> None:
    recording = Recording(
        args.recording, mode="record" if args.record else "replay"
    )
    signal = match_signal()
    validation = behavior_cases("validation")
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    with recording.active():
        # Experience: replayed from the reasoning-floor recording, which
        # holds these runs.
        floor_recording = Recording(FLOOR_RECORDED, mode="replay")
        with floor_recording.active():
            episodes = await gather_experience()
        tier = await remember(episodes)
        experience = await recall_experience(tier)
        solved = sum(e.success for e in experience)
        print(
            f"Experience: {len(experience)} runs recalled from episodic"
            f" memory, {solved} accepted"
        )

        control = await learn(CONTROL.name, experience)
        control_match = await signal.measure(
            control.source, ground_truth=validation
        )
        print(
            f"\n{'mechanism':<24} {'validation match':>16}"
            f" {'writer calls':>12}"
        )
        print(
            f"{'(control) ' + CONTROL.name:<24}"
            f" {control_match.metadata['matched']:>9}/{len(validation):<6}"
            f" {control.writer_calls:>12}"
        )

        learned_by: dict[str, object] = {}

        async def evaluate(artifact):
            spent = recording.spent(PRICES)
            if spent > args.budget:
                raise BudgetSpent(
                    f"stopped before {artifact.content['mechanism']}:"
                    f" ${spent:.2f} spent, budget ${args.budget:.2f}"
                )
            learned = await learn(
                artifact.content["mechanism"], experience
            )
            learned_by[artifact.content["mechanism"]] = learned
            measurement = await signal.measure(
                learned.source, ground_truth=validation
            )
            print(
                f"{learned.mechanism:<24}"
                f" {measurement.metadata['matched']:>9}"
                f"/{len(validation):<6}"
                f" {learned.writer_calls:>12}"
            )
            return measurement, {
                "writer_calls": learned.writer_calls,
                "notes": learned.notes,
            }

        archive = SQLiteArchive(out / "learning_search.sqlite")
        names = list(MECHANISMS)
        seed = genesis(
            "learning_mechanism.ch06",
            ArtifactKind.CODE,
            mechanism_config(names[0]),
        )
        search = CandidateSearch(
            [mechanism_config(n) for n in names], evaluate
        )
        variants = []
        try:
            async for variant in search.propose(
                seed, signal, archive, None
            ):
                variants.append(variant)
        except BudgetSpent as exc:
            print(f"\nBudget: {exc}")
        winner = await search.select(variants, signal, archive)
        archive.close()
        tier.close()
        name = winner.content["mechanism"]
        learned = learned_by[name]
        print(
            f"\nKept: {name} (version {winner.version}); it replaced the"
            " incumbent only on strict improvement."
        )

        test = await signal.measure(
            learned.source, ground_truth=behavior_cases("test")
        )
        print(
            "Test tasks, scored once:"
            f" {test.metadata['matched']}/{test.metadata['cases']} match"
            " the starting memory"
        )
        if test.feedback:
            print(f"  mismatches: {test.feedback}")

        (out / f"learned_{name}.py").write_text(
            learned.source, encoding="utf-8"
        )
        report = {
            "control": {
                "matched": control_match.metadata["matched"],
                "source": control.source,
            },
            "variants": [
                {
                    "mechanism": v.artifact.content["mechanism"],
                    "version": v.artifact.version,
                    "matched": v.measurement.metadata["matched"],
                    **v.metadata,
                }
                for v in variants
            ],
            "kept": name,
            "test_matched": test.metadata["matched"],
        }

        if args.floor:
            # The learned memory's own floor, measured as the starting
            # memory's was.
            entry = {
                **STARTING_MEMORY,
                "body": learned.source,
                "created_by": name,
            }
            memory = await install_memory(entry)
            floor_recording_live = Recording(
                FLOOR_RECORDED, mode="record" if args.record else "replay"
            )
            with floor_recording_live.active():
                result = await sweep(
                    f"learned_{name}",
                    task_set("validation"),
                    lambda t, e: run_task(t, e, memory=memory),
                    efforts=["none", "low"],
                    prices=PRICES,
                )
            for row in result.rows():
                print(
                    f"  validation, {row['effort']:<4}:"
                    f" {row['solved']}/{row['total']} solved,"
                    f" ${row['dollars']:.4f} a task"
                )
            print(
                f"  floor: solid {result.floor(SOLID) or 'not reached'},"
                f" adequate {result.floor(ADEQUATE) or 'not reached'}"
            )
            report["floor"] = result.to_dict()

    (out / "learning_search.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    if args.record:
        print(
            f"\n{recording.live_calls} live writer calls added to"
            f" {recording.path.name}, ${recording.spent(PRICES):.2f}"
        )
    if args.promote:
        target = MEMORIES / "learned_find_shared_starts.py"
        target.write_text(
            f"# Learned by {name}, promoted by a person.\n"
            + learned.source,
            encoding="utf-8",
        )
        print(f"Promoted: {target.relative_to(REPO_ROOT)}")
    else:
        print(
            "Not promoted; a person promotes the kept memory with"
            " --promote."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--record",
        action="store_true",
        help="Call gpt-5.4-mini for calls the recording lacks.",
    )
    parser.add_argument("--recording", type=Path, default=RECORDED)
    parser.add_argument(
        "--budget",
        type=float,
        default=2.5,
        help=(
            "Dollars of live writer calls after which no further mechanism"
            " starts."
        ),
    )
    parser.add_argument(
        "--no-floor",
        dest="floor",
        action="store_false",
        help="Skip measuring the kept memory's reasoning floor.",
    )
    parser.add_argument(
        "--promote",
        action="store_true",
        help="Save the kept memory as the promoted memory.",
    )
    parser.add_argument(
        "--output", type=Path, default=HERE / "runs" / "learning_search"
    )
    args = parser.parse_args()
    if args.record:
        from helix.env import load_env

        load_env()
    try:
        asyncio.run(main(args))
    except KeyError as exc:
        sys.exit(
            f"{exc.args[0]}. The recording has no runs for this search"
            " yet; record them with --record (paid calls to"
            " gpt-5.4-mini)."
        )
