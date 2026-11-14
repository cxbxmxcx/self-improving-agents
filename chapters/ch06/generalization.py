"""Apply the selected learning mechanism to a second scheduling class.

The second class requires the same meeting time on several days. Measure its
floor with and without a reference function, then let the selected mechanism
learn from this class's experience. Compare the resulting function with its
reference and confirm its use in the agent.

Run listing 6.3 first to supply the first class's learned function. The final
comparison offers both functions from one procedural memory, with a labeled
reference fallback if the first learned source is absent. Replay needs no
API key, while --record adds paid calls when recorded replies are missing.
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

from chapters.ch06.assistant.harness import (
    STARTING_MEMORY,
    STARTING_SAME_TIME,
    GLOBAL,
    MEMORIES,
    PRICES,
    install_memory,
    run_task,
)
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
from helix.eval.effort import ADEQUATE, EFFORTS, SOLID, sweep
from helix.memory.learning_mechanisms import CONTROL, MECHANISMS, Writer

HERE = Path(__file__).parent
RECORDED = HERE / "assistant" / "recorded" / "generalization.jsonl"
SEARCH_REPORT = HERE / "runs" / "learning_search" / "learning_search.json"
WRITER = {
    "model": "gpt-5.4-mini",
    "params": {"reasoning_effort": "medium"},
}
CLASS = CLASSES["same_time"]


def kept_mechanism(choice: str | None) -> str:
    """The mechanism listing 6.3 kept, unless one is named."""
    if choice:
        return choice
    if SEARCH_REPORT.exists():
        return json.loads(SEARCH_REPORT.read_text(encoding="utf-8"))[
            "kept"
        ]
    return next(iter(MECHANISMS))


def first_class_memory(mechanism: str) -> tuple[str, str]:
    """The first class's learned memory, as listing 6.3 left it."""
    for path, label in (
        (
            MEMORIES / "learned_find_shared_starts.py",
            "promoted learned memory",
        ),
        (
            HERE / "runs" / "learning_search" / f"learned_{mechanism}.py",
            "learned memory, not promoted",
        ),
    ):
        if path.exists():
            return path.read_text(encoding="utf-8"), label
    return (MEMORIES / STARTING_MEMORY["file"]).read_text(
        encoding="utf-8"
    ), "starting memory"


def report(result) -> None:
    print(f"\n{result.arm}")
    for row in result.rows():
        print(
            f"  {row['effort']:<7} {row['solved']:>3}/{row['total']:<2}"
            f" solved  {row['reasoning_tokens']:>6.0f} reasoning tokens "
            f" ${row['dollars']:.4f} a task"
        )
    print(
        f"  floor: solid {result.floor(SOLID) or 'not reached'}, adequate"
        f" {result.floor(ADEQUATE) or 'not reached'}"
    )


async def main(args: argparse.Namespace) -> None:
    recording = Recording(
        args.recording, mode="record" if args.record else "replay"
    )
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    name = kept_mechanism(args.mechanism)
    results: dict = {"mechanism": name}
    with recording.active():
        # The second class's reasoning floor, without memory and with its
        # starting memory.
        dev = CLASS.task_set("dev")
        starting = await install_memory(STARTING_SAME_TIME)
        for arm, memory in (
            ("no_memory", None),
            ("starting_memory", starting),
        ):
            result = await sweep(
                arm,
                dev,
                lambda t, e, m=memory: run_task(t, e, memory=m),
                efforts=EFFORTS,
                prices=PRICES,
            )
            report(result)
            results[arm] = result.to_dict()

        # The kept mechanism, unchanged, learns from this class's
        # experience.
        experience = await recall_experience(
            await remember(await gather_experience(task_class=CLASS.name))
        )
        print(
            f"\nExperience: {len(experience)} runs recalled from episodic"
            f" memory, {sum(e.success for e in experience)} accepted"
        )
        signal = match_signal(CLASS.name)
        validation = behavior_cases("validation", CLASS.name)
        learned = {}
        for mechanism in (CONTROL, MECHANISMS[name]):
            spent = recording.spent(PRICES)
            if spent > args.budget:
                print(
                    f"Budget: stopped before {mechanism.name}:"
                    f" ${spent:.2f} spent, budget ${args.budget:.2f}"
                )
                break
            writer = Writer(
                model=WRITER["model"], params=dict(WRITER["params"])
            )
            try:
                learned[mechanism.name] = await mechanism.learn(
                    experience, act, writer, CLASS.contract
                )
            # Nothing in the experience to learn from.
            except ValueError as exc:
                print(f"{mechanism.name}: no learned memory ({exc})")
                results[mechanism.name] = {
                    "matched": 0,
                    "feedback": str(exc),
                }
                continue
            match = await signal.measure(
                learned[mechanism.name].source, ground_truth=validation
            )
            label = (
                f"(control) {mechanism.name}"
                if mechanism is CONTROL
                else mechanism.name
            )
            print(
                f"{label:<26} validation match"
                f" {match.metadata['matched']}/{len(validation)},"
                f" {learned[mechanism.name].writer_calls} writer calls"
            )
            results[mechanism.name] = {
                "matched": match.metadata["matched"],
                "feedback": match.feedback,
            }

        if name in learned:
            source = learned[name].source
            (out / f"learned_{CLASS.contract.name}.py").write_text(
                source, encoding="utf-8"
            )
            test = await signal.measure(
                source, ground_truth=behavior_cases("test", CLASS.name)
            )
            print(
                "Test tasks, scored once:"
                f" {test.metadata['matched']}/{test.metadata['cases']}"
                " match the starting memory"
            )
            results["test_matched"] = test.metadata["matched"]

            # The learned memory's own floor on validation, as listing 6.2
            # measured the starting one.
            memory = await install_memory(
                {
                    **STARTING_SAME_TIME,
                    "body": source,
                    "created_by": name,
                }
            )
            floor = await sweep(
                f"learned_{name}",
                validation,
                lambda t, e: run_task(t, e, memory=memory),
                efforts=["none", "low"],
                prices=PRICES,
            )
            report(floor)
            results["learned_floor"] = floor.to_dict()

            # One memory, both learned functions; recall offers both and
            # the agent picks.
            first_source, first_label = first_class_memory(name)
            both = await install_memory(
                {
                    **STARTING_MEMORY,
                    "body": first_source,
                    "created_by": name,
                }
            )
            await both.save_skill(
                CLASS.contract.name,
                CLASS.contract.description,
                source,
                body_kind="python",
                scope_key=GLOBAL,
                metadata={"created_by": name},
            )
            print(
                "\nMemory holding both functions (first class:"
                f" {first_label}), reasoning none:"
            )
            for task_class in (CLASSES["multi_attendee"], CLASS):
                tasks = task_class.task_set("validation")
                outcomes = await asyncio.gather(
                    *(
                        run_task(t, "none", memory=both, skill_k=2)
                        for t in tasks
                    )
                )
                solved = sum(ok for ok, _, _ in outcomes)
                print(
                    f"  {task_class.name:<15} {solved}/{len(tasks)} solved"
                )
                results[f"both_{task_class.name}"] = solved

    (out / "generalization.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    if args.record:
        print(
            f"\n{recording.live_calls} live calls added to"
            f" {recording.path.name}, ${recording.spent(PRICES):.2f}"
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
        "--mechanism",
        choices=list(MECHANISMS),
        help="Override the mechanism listing 6.3 kept.",
    )
    parser.add_argument(
        "--budget",
        type=float,
        default=2.5,
        help=(
            "Dollars of live calls after which no further mechanism"
            " starts."
        ),
    )
    parser.add_argument(
        "--output", type=Path, default=HERE / "runs" / "generalization"
    )
    args = parser.parse_args()
    if args.record:
        from helix.env import load_env

        load_env()
    try:
        asyncio.run(main(args))
    except KeyError as exc:
        sys.exit(
            f"{exc.args[0]}. The recording has no runs for this listing"
            " yet; record them with --record (paid calls to"
            " gpt-5.4-mini)."
        )
