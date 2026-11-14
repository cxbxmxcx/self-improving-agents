"""Search the bank of learners when the starting learner cannot learn a task.

The assistant starts as listing 6.2's does: the starting memory, trusted, and
the starting learner, value learning. Then four of the user's colleagues
add scheduling rules to their calendar notes. The first requests recall the
starting memory and fail, so the detector quarantines it, and the agent works
at the higher setting while the recorder keeps its runs. The learner tries
after 4 and 8 runs; when the gate refuses both, the learner monitor reports a
gap in the learner itself.

At the next learning point, candidate search tries every learner in the bank
on the same runs and scores each by the same gate. It keeps the starting
learner unless a later one scores strictly higher. Promotion stays a person's
choice: with --promote, the loop takes the kept learner, installs its memory,
serves the remaining requests at the controller's setting, and measures the
promoted memory on test requests the gate never saw. Replay uses saved
replies; --record adds paid calls.
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chapters.ch06.assistant import generator_rules
from chapters.ch06.assistant.harness import (
    PRICES,
    STARTING_LEARNER,
    STARTING_MEMORY,
    install_memory,
    run_task,
)
from chapters.ch06.assistant.learning import CONTRACT, act
from chapters.ch06.assistant.online import (
    Controller,
    Detector,
    Gate,
    LearnerMonitor,
    Recorder,
)
from chapters.ch06.assistant.recording import Recording, call_key
from helix.archive import SQLiteArchive
from helix.artifact import ArtifactKind, genesis
from helix.eval.effort import RunRecord
from helix.memory.learning_mechanisms import MECHANISMS, Episode, Writer
from helix.search.candidates import CandidateSearch
from helix.signal import GapMeasurement
from helix.signals.cost import call_dollars

HERE = Path(__file__).parent
RECORDED = HERE / "assistant" / "recorded" / "learner_search.jsonl"
FLOORS = HERE / "runs" / "reasoning_floor.json"
WRITER = {
    "model": "gpt-5.4-mini",
    "params": {"reasoning_effort": "medium"},
}
# The most one live call is expected to cost: an agent turn, or a learner's
# writing call (a long rewrite can run to tens of thousands of tokens).
AGENT_MARGIN, WRITER_MARGIN = 0.03, 0.35


class BudgetSpent(Exception):
    """The next live call could take this run past its budget."""


class CappedRecording(Recording):
    """A recording that refuses a live call which could cross the budget."""

    def __init__(self, path: Path, *, mode: str, budget: float) -> None:
        super().__init__(path, mode=mode)
        self.budget = budget

    async def __call__(self, **kwargs):
        if self.mode == "record" and call_key(kwargs) not in self.rows:
            margin = AGENT_MARGIN if kwargs.get("tools") else WRITER_MARGIN
            if self.spent(PRICES) + margin > self.budget:
                raise BudgetSpent(
                    f"${self.spent(PRICES):.4f} of live calls spent; the"
                    f" next could cross the ${self.budget:.2f} budget"
                )
        return await super().__call__(**kwargs)


async def install(source: str, created_by: str):
    entry = {**STARTING_MEMORY, "body": source, "created_by": created_by}
    return await install_memory(entry)


def writer_dollars(writer: Writer) -> float:
    return round(sum(call_dollars(WRITER["model"], u, PRICES) or 0.0
                     for u in writer.usage), 4)


def log_run(log, n, task, trusted, effort, solved, reason, trajectory):
    """One line per request, and the row the summary and figures read."""
    state = "trusted" if trusted else "quarantined"
    verdict = "accepted" if solved else "rejected"
    print(f"{n:>2} {task['id']:<5} {state:<11} {effort:<6} {verdict}")
    spent = RunRecord.of(task["id"], effort, solved, trajectory, prices=PRICES)
    log.append({"n": n, "task": task["id"], "memory": state, "effort": effort,
                "solved": solved, "reason": reason,
                "reasoning_tokens": spent.reasoning_tokens,
                "dollars": spent.dollars, "seconds": spent.seconds})


@dataclass
class LoopState:
    """What the loop changes between requests: the learner, its memory, and trust."""

    learner: Any
    memory: Any
    trusted: bool = True
    search_due: bool = False
    search: dict | None = None
    promoted: str | None = None


async def learning_point(
    state: LoopState, episodes: list[Episode], gate: Gate,
    detector: Detector, monitor: LearnerMonitor,
    args: argparse.Namespace, log: list, attempts: list,
) -> bool:
    """Relearn with the loop's learner, or search the bank after refusals.

    Returns False when the stream stops: the search has kept a learner, and
    promotion waits for a person.
    """
    if state.search_due:
        state.search = await search_bank(episodes, gate, args.output)
        log[-1]["event"] = f"search kept {state.search['kept']}"
        if not args.promote:
            print("   promotion is a person's choice: rerun with --promote"
                  " to install it; the stream stops here")
            return False
        kept = state.search["kept"]
        state.learner = MECHANISMS[kept]
        state.memory = await install(state.search["source"], kept)
        state.promoted, state.trusted = state.search["source"], True
        state.search_due = False
        detector.reset()
        monitor.reset()
        print(f"   promoted: {kept} is now the loop's learner, and its"
              " memory is installed")
        return True
    writer = Writer(model=WRITER["model"], params=dict(WRITER["params"]))
    learned = await state.learner.learn(episodes, act, writer, CONTRACT)
    accepted, total, passed = await gate.check(learned.source)
    name = state.learner.name
    attempts.append({"learner": name, "runs": len(episodes),
                     "gate": f"{accepted}/{total}",
                     "writer_calls": learned.writer_calls,
                     "dollars": writer_dollars(writer)})
    log[-1]["event"] = (f"{name} learned from {len(episodes)} runs;"
                        f" gate {accepted}/{total}")
    print(f"   learner: {name} wrote a function from {len(episodes)} runs;"
          f" gate {accepted}/{total}"
          f" {'passed, installed' if passed else 'refused'}")
    if passed:
        state.memory = await install(learned.source, name)
        state.trusted = True
        detector.reset()
    if monitor.observe(passed):
        state.search_due = True
        print(f"   learner monitor: {monitor.refusals} refusals in a row;"
              " search the bank at the next learning point")
    return True


async def search_bank(
    episodes: list[Episode], gate: Gate, out: Path
) -> dict:
    """Each learner learns from the same runs; the gate scores its memory."""
    learned_by, rows = {}, []

    async def evaluate(artifact):
        name = artifact.content["mechanism"]
        writer = Writer(model=WRITER["model"], params=dict(WRITER["params"]))
        try:
            learned = await MECHANISMS[name].learn(
                episodes, act, writer, CONTRACT
            )
        except (KeyError, BudgetSpent):
            raise
        except Exception as exc:  # noqa: BLE001 - a learner that fails scores 0
            row = {"learner": name, "gate": "failed", "accepted": 0,
                   "writer_calls": writer.calls, "dollars": writer_dollars(writer),
                   "error": f"{type(exc).__name__}: {exc}"}
            rows.append(row)
            print(f"   {name:<24} failed to learn: {row['error']}")
            return GapMeasurement(score=0.0, feedback=row["error"]), row
        accepted, total, _ = await gate.check(learned.source)
        learned_by[name] = learned
        row = {"learner": name, "gate": f"{accepted}/{total}", "accepted": accepted,
               "writer_calls": learned.writer_calls, "dollars": writer_dollars(writer)}
        rows.append(row)
        calls = "learning call" if row["writer_calls"] == 1 else "learning calls"
        print(f"   {name:<24} gate {row['gate']:<5} {row['writer_calls']:>3}"
              f" {calls:<14}  ${row['dollars']:.4f}")
        return GapMeasurement(score=accepted / total,
                              feedback=f"gate {accepted}/{total}"), row

    archive_path = out / "learner_search.sqlite"
    archive_path.unlink(missing_ok=True)
    archive = SQLiteArchive(archive_path)
    # The starting learner comes first in the bank, so it is the incumbent.
    names = list(MECHANISMS)
    seed = genesis(
        "learner.ch06", ArtifactKind.CODE, {"mechanism": names[0]}
    )
    search = CandidateSearch([{"mechanism": n} for n in names], evaluate)
    print(f"   search: {len(names)} learners from the bank learn from"
          f" {len(episodes)} runs")
    variants = [v async for v in search.propose(seed, None, archive, None)]
    kept = (await search.select(variants)).content["mechanism"]
    archive.close()
    print(f"   search: kept {kept}; a later learner replaces the incumbent only"
          " with a strictly higher gate score")
    return {"kept": kept, "source": learned_by[kept].source, "rows": rows}


async def measure(tasks: list[dict], effort: str, memory) -> dict:
    """The agent on requests no learner or gate has seen, at one setting."""
    records = []
    for task in tasks:
        try:
            solved, _, trajectory = await run_task(task, effort, memory=memory)
        except (KeyError, BudgetSpent):
            raise
        except Exception:  # noqa: BLE001 - a crashed run counts as rejected
            records.append({"solved": False, "reasoning_tokens": 0,
                            "dollars": 0.0, "seconds": 0.0})
            continue
        spent = RunRecord.of(task["id"], effort, solved, trajectory, prices=PRICES)
        records.append({"solved": solved, "reasoning_tokens": spent.reasoning_tokens,
                        "dollars": spent.dollars, "seconds": spent.seconds})
    n = len(records)
    return {"effort": effort, "memory": memory is not None, "tasks": n,
            "accepted": sum(r["solved"] for r in records),
            "reasoning_tokens": round(sum(r["reasoning_tokens"] for r in records) / n),
            "dollars": round(sum(r["dollars"] for r in records) / n, 5),
            "seconds": round(sum(r["seconds"] for r in records) / n, 1)}


def summarize(log, attempts, search, measured) -> None:
    print()
    for effort in dict.fromkeys(row["effort"] for row in log):
        rows = [row for row in log if row["effort"] == effort]
        print(f"  {effort:<5} {sum(r['solved'] for r in rows)}/{len(rows)} accepted,"
              f" {sum(r['reasoning_tokens'] for r in rows) / len(rows):.0f} reasoning"
              f" tokens and ${sum(r['dollars'] for r in rows) / len(rows):.4f} a request")
    learning = sum(a["dollars"] for a in attempts)
    if search:
        learning += sum(r["dollars"] for r in search["rows"])
    print(f"  learning: {len(attempts)} attempts by the starting learner"
          + (f", then a search of {len(search['rows'])} learners" if search else "")
          + f", ${learning:.4f}")
    for m in measured:
        print(f"  test requests at {m['effort']}"
              f" {'with' if m['memory'] else 'without'} memory:"
              f" {m['accepted']}/{m['tasks']} accepted, {m['reasoning_tokens']}"
              f" reasoning tokens, ${m['dollars']:.4f} and {m['seconds']} s a request")


async def main(args: argparse.Namespace) -> None:
    recording = CappedRecording(
        args.recording, mode="record" if args.record else "replay",
        budget=args.budget,
    )
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    # The learner and the memory the harness holds when metacognition starts.
    state = LoopState(MECHANISMS[STARTING_LEARNER],
                      await install_memory(STARTING_MEMORY))

    controller = Controller.from_floors(FLOORS)
    detector = Detector()
    monitor = LearnerMonitor(limit=args.refusals)
    gate = Gate(generator_rules.task_set("gate"))
    recorder = Recorder()
    stream = (generator_rules.task_set("first_contact")
              + generator_rules.task_set("stream"))
    quarantined_at = next_attempt = 0
    stopped = None
    log, attempts, measured = [], [], []
    print(f"\n{'#':>2} {'task':<5} {'memory':<11} {'effort':<6} verdict")
    try:
        with recording.active():
            for n, task in enumerate(stream, 1):
                effort = controller.effort(state.trusted)
                try:
                    solved, reason, trajectory = await run_task(
                        task, effort, memory=state.memory if state.trusted else None
                    )
                except (KeyError, BudgetSpent):
                    raise
                except Exception as exc:  # noqa: BLE001 - logged, not learned from
                    print(f"{n:>2} {task['id']:<5} crashed: {type(exc).__name__}")
                    continue
                await recorder.record(
                    Episode.from_trajectory(task["id"], trajectory, solved, reason),
                    effort=effort, used_memory=state.trusted,
                )
                log_run(log, n, task, state.trusted, effort, solved, reason, trajectory)

                if state.trusted and detector.observe(solved):
                    state.trusted = False
                    quarantined_at = recorder.count
                    next_attempt = quarantined_at + args.relearn_after
                    log[-1]["event"] = "detector quarantined the memory"
                    print(f"   detector: memory quarantined ({reason})")
                elif not state.trusted and recorder.count >= next_attempt:
                    episodes = await recorder.since(quarantined_at)
                    next_attempt = recorder.count + args.relearn_after
                    if any(e.success for e in episodes) and not await learning_point(
                            state, episodes, gate, detector, monitor, args, log, attempts):
                        break
            if state.promoted and args.measure:
                tests = generator_rules.task_set("test")
                measured.append(await measure(tests, controller.effort(True), state.memory))
                if args.paired:
                    measured.append(await measure(tests, controller.effort(False), None))
    except BudgetSpent as exc:
        stopped = str(exc)
        print(f"\nStopped: {exc}")
    recorder.close()

    search = state.search
    (out / "learner_search.json").write_text(json.dumps({
        "controller": controller.table, "runs": log, "attempts": attempts,
        "search": search and {"kept": search["kept"], "rows": search["rows"]},
        "promoted": bool(state.promoted), "measured": measured, "stopped": stopped,
    }, indent=2), encoding="utf-8")
    if state.promoted:
        (out / "promoted_find_shared_starts.py").write_text(state.promoted, encoding="utf-8")
    if log:
        summarize(log, attempts, search, measured)
    if args.record:
        print(f"\n{recording.live_calls} live calls added to"
              f" {recording.path.name}, ${recording.spent(PRICES):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--record",
        action="store_true",
        help="Call gpt-5.4-mini for calls the recording lacks.",
    )
    parser.add_argument("--recording", type=Path, default=RECORDED)
    parser.add_argument(
        "--promote", action="store_true",
        help="Install the learner the search keeps and continue the stream.",
    )
    parser.add_argument(
        "--refusals", type=int, default=2,
        help="Gate refusals in a row that start a search of the bank.",
    )
    parser.add_argument(
        "--relearn-after", type=int, default=4,
        help="Runs recorded in quarantine between learning attempts.",
    )
    parser.add_argument(
        "--no-measure", dest="measure", action="store_false",
        help="Skip measuring the promoted memory on the test requests.",
    )
    parser.add_argument(
        "--paired", action="store_true",
        help="Also measure the test requests at the higher setting without memory.",
    )
    parser.add_argument(
        "--budget", type=float, default=1.0,
        help="Dollars of live calls after which the run stops.",
    )
    parser.add_argument(
        "--output", type=Path, default=HERE / "runs" / "learner_search"
    )
    args = parser.parse_args()
    if args.record:
        from helix.env import load_env

        load_env()
    try:
        asyncio.run(main(args))
    except KeyError as exc:
        sys.exit(
            f"{exc.args[0]}. This run needs a reply the recording"
            " lacks; record it with --record (paid calls to"
            " gpt-5.4-mini)."
        )
