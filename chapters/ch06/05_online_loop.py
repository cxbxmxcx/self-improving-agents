"""Run the metacognition loop online while the user's conditions change.

The assistant starts with the memory the kept learning mechanism wrote in
listing 6.3, trusted from its offline measurements, and a controller that
reads its setting from listing 6.2's floors. It serves first-class requests, then requests
after Dana rewords her note to ask for half an hour. The detector watches
the runs that used memory and quarantines the memory when they fail.

While the memory is quarantined, the controller raises reasoning and the
recorder keeps writing runs to episodic memory. The learner writes a new
function from those runs, and the gate installs it only when it clears the
bar on held-back cases. Replay uses saved replies; --record adds paid calls.
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

from chapters.ch06.assistant import generator, generator_drift
from chapters.ch06.assistant.harness import (
    BOOTSTRAPPED_MEMORY,
    PRICES,
    install_memory,
    run_task,
)
from chapters.ch06.assistant.learning import CONTRACT, act
from chapters.ch06.assistant.online import recorder as make_recorder
from chapters.ch06.assistant.recording import Recording
from helix.eval.effort import RunRecord
from helix.metacognition import Controller, Detector, Gate
from helix.memory.learning_mechanisms import MECHANISMS, Episode, Writer
from helix.signals.cost import call_dollars

HERE = Path(__file__).parent
RECORDED = HERE / "assistant" / "recorded" / "online_loop.jsonl"
FLOORS = HERE / "runs" / "reasoning_floor.json"
LEARNED = HERE / "runs" / "learning_search"
WRITER = {
    "model": "gpt-5.4-mini",
    "params": {"reasoning_effort": "medium"},
}


async def install(source: str, created_by: str):
    entry = {**BOOTSTRAPPED_MEMORY, "body": source, "created_by": created_by}
    return await install_memory(entry)


def log_run(log, n, task, trusted, effort, solved, reason, trajectory):
    """One line per request, and the row the summary and figures read."""
    state = "trusted" if trusted else "quarantined"
    verdict = "accepted" if solved else "rejected"
    print(f"{n:>2} {task['id']:<5} {state:<11} {effort:<6} {verdict}")
    spent = RunRecord.of(task["id"], effort, solved, trajectory, prices=PRICES)
    log.append({"n": n, "task": task["id"], "memory": state, "effort": effort,
                "solved": solved, "reason": reason,
                "reasoning_tokens": spent.reasoning_tokens, "dollars": spent.dollars})


def log_learning(log, kept, episodes, learned, writer, gate_result):
    accepted, total, passed = gate_result.accepted, gate_result.total, gate_result.passed
    log[-1].update(event=f"{kept} relearned from {len(episodes)} runs; gate {accepted}/{total}",
                   writer_calls=learned.writer_calls,
                   learning_dollars=round(sum(call_dollars(WRITER["model"], u, PRICES) or 0.0
                                              for u in writer.usage), 4))
    print(f"   learner: {kept} wrote a function from {len(episodes)} runs;"
          f" gate {accepted}/{total} {'passed, installed' if passed else 'refused'}")


def summarize(log) -> None:
    print()
    for effort in dict.fromkeys(row["effort"] for row in log):
        rows = [row for row in log if row["effort"] == effort]
        print(f"  {effort:<5} {sum(r['solved'] for r in rows)}/{len(rows)} accepted,"
              f" {sum(r['reasoning_tokens'] for r in rows) / len(rows):.0f} reasoning tokens and"
              f" ${sum(r['dollars'] for r in rows) / len(rows):.4f} a request")
    learning = [row for row in log if "learning_dollars" in row]
    print(f"  learning: {sum(r['writer_calls'] for r in learning)} writer calls,"
          f" ${sum(r['learning_dollars'] for r in learning):.4f}")


async def main(args: argparse.Namespace) -> None:
    recording = Recording(
        args.recording, mode="record" if args.record else "replay"
    )
    # The mechanism listing 6.3 kept, and the memory it wrote.
    kept = json.loads(
        (LEARNED / "learning_search.json").read_text(encoding="utf-8")
    )["kept"]
    learner = MECHANISMS[kept]
    source = (LEARNED / f"learned_{kept}.py").read_text(encoding="utf-8")
    memory = await install(source, kept)
    trusted = True

    controller = Controller.from_sweeps(FLOORS, memory="bootstrapped_memory", no_memory="no_memory")
    detector = Detector()
    gate = Gate(cases=[task["id"] for task in generator_drift.task_set("gate")])
    recorder = make_recorder()
    print(
        f"Controller: {controller.table['memory']} with trusted memory,"
        f" {controller.table['no_memory']} without"
    )
    stream = (
        generator.task_set("test")[: args.before]
        + generator_drift.task_set("stream")[: args.after]
    )
    quarantined_at = next_attempt = 0
    log = []
    print(f"\n{'#':>2} {'task':<5} {'memory':<11} {'effort':<6} verdict")
    with recording.active():
        for n, task in enumerate(stream, 1):
            if recording.spent(PRICES) > args.budget:
                print("Budget spent; the stream stops here.")
                break
            effort = controller.effort(trusted=trusted)
            solved, reason, trajectory = await run_task(
                task, effort, memory=memory if trusted else None
            )
            await recorder.record(
                Episode.from_trajectory(
                    task["id"], trajectory, solved, reason
                ),
                effort=effort,
                used_memory=trusted,
            )
            log_run(log, n, task, trusted, effort, solved, reason,
                    trajectory)

            if trusted and detector.observe(solved):
                # The memory passed its gate; recent runs say the world
                # it learned no longer holds.
                trusted = False
                quarantined_at = recorder.count
                next_attempt = quarantined_at + args.relearn_after
                log[-1]["event"] = "detector quarantined the memory"
                print(f"   detector: memory quarantined ({reason})")
            elif not trusted and recorder.count >= next_attempt:
                episodes = await recorder.since(quarantined_at)
                next_attempt = recorder.count + args.relearn_after
                if not any(e.success for e in episodes):
                    continue
                writer = Writer(
                    model=WRITER["model"], params=dict(WRITER["params"])
                )
                learned = await learner.learn(
                    episodes, act, writer, CONTRACT
                )
                result = await gate.check(
                    learned.source, act, gate.cases
                )
                log_learning(log, kept, episodes, learned, writer, result)
                if result.passed:
                    memory = await install(learned.source, kept)
                    source, trusted = learned.source, True
                    detector.reset()
    recorder.close()

    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    (out / "online_loop.json").write_text(
        json.dumps({"controller": controller.table, "runs": log}, indent=2),
        encoding="utf-8",
    )
    (out / "relearned_find_shared_starts.py").write_text(
        source, encoding="utf-8"
    )
    summarize(log)
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
        "--before", type=int, default=6,
        help="Requests served before the note changes.",
    )
    parser.add_argument(
        "--after", type=int, default=14,
        help="Requests served after the note changes.",
    )
    parser.add_argument(
        "--relearn-after", type=int, default=4,
        help="Runs recorded in quarantine between learning attempts.",
    )
    parser.add_argument(
        "--budget", type=float, default=1.5,
        help="Dollars of live calls after which the stream stops.",
    )
    parser.add_argument(
        "--output", type=Path, default=HERE / "runs" / "online_loop"
    )
    args = parser.parse_args()
    if args.record:
        from helix.env import load_env

        load_env()
    try:
        asyncio.run(main(args))
    except KeyError as exc:
        sys.exit(
            f"{exc.args[0]}. The recording has no runs for this stream"
            " yet; record them with --record (paid calls to"
            " gpt-5.4-mini)."
        )
