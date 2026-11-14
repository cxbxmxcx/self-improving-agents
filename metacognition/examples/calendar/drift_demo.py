"""Chapter 6's drift, replayed into a store you can open in the dashboard. No API key needed.

    python -m examples.calendar.drift_demo                    # replay into examples/runs/drift.meta
    metacog serve examples.calendar.drift_demo:agent          # then open http://127.0.0.1:8620

The study's online loop (E5) serves six requests, then fourteen more after Maya's
client starts asking for "half an hour". This demo replays the first twelve from
the book's recording with one change: a person approves installs. So the
dashboard opens on the stale procedure quarantined at run 8, four runs at the
no-memory setting, and the version relearned at run 12 waiting for you, with
its diff and its gate result. Approving it from the dashboard trusts it.

Asking the served agent anything new makes live, paid calls: the recording holds
the study's replies, not yours.
"""

from __future__ import annotations

import asyncio
import shutil
import sys
from pathlib import Path

import helix
from chapters.ch06.assistant import generator, generator_drift
from chapters.ch06.assistant.environment import SchedulingEnvironment
from chapters.ch06.assistant.learning import CONTRACT, act, gather_experience, recall_experience, remember
from helix.memory.learning_mechanisms import Writer
from helix.metacognition import CALENDAR, Controller, Gate, Metacognition, ValueLearning
from helix.metacognition.pricing import PRICES

from examples.calendar.agent import FIND_SHARED_STARTS, assistant, check
from metacog import Agent, Recording, RunConfig, Runner

RECORDED = Path(helix.__file__).resolve().parents[1] / "chapters" / "ch06" / "assistant" / "recorded"
STORE = Path(__file__).resolve().parents[1] / "runs" / "drift.meta"
WRITER = {"model": "gpt-5.4-mini", "params": {"reasoning_effort": "medium"}}
UPTO = 12


def build(store: str | Path = STORE) -> Agent:
    """The study's E5 deployment, except that a person approves installs."""
    gate = Gate(cases=[t["id"] for t in generator_drift.task_set("gate")])
    meta = Metacognition(
        [FIND_SHARED_STARTS], verifier=check, learner=ValueLearning(CALENDAR), writer_model=WRITER["model"],
        writer_params=WRITER["params"], practice=act, gate=gate,
        controller=Controller({"memory": "none", "no_memory": "high"}), recall_k=1, relearn_after=4,
        learning="inline", approval="human", prices=PRICES, store=str(store))
    return assistant(metacognition=meta)


def agent() -> Agent:
    """The demo's agent on its store, for `metacog serve examples.calendar.drift_demo:agent`."""
    return build(STORE)


async def kept_memory() -> str:
    """The memory the study started E5 with: value learning on E1's medium and high runs (listing 6.3)."""
    with Recording(RECORDED / "reasoning_floor.jsonl").active():
        episodes = await gather_experience()
    experience = await recall_experience(await remember(episodes))
    with Recording(RECORDED / "learning_search.jsonl").active():
        learned = await ValueLearning(CALENDAR).learn(experience, act, Writer(**WRITER), CONTRACT)
    return learned.source


async def replay(served: Agent, upto: int = UPTO, out=print) -> None:
    """Serve E5's first `upto` requests from the book's recording."""
    served.metacognition.install("find_shared_starts", await kept_memory(), created_by="value_learning")
    stream = generator.task_set("test")[:6] + generator_drift.task_set("stream")[:14]
    with Recording(RECORDED / "online_loop.jsonl").active():
        for n, task in enumerate(stream[:upto], 1):
            result = await Runner.run(served, task["text"], context=SchedulingEnvironment(task), max_turns=16,
                                      run_config=RunConfig(max_tool_calls=24, case_id=task["id"]))
            report = result.metacognition
            out(f"{n:>3} {task['id']:<5} {'offered' if report.offered else '-':<8} {report.effort:<5} "
                f"{'accepted' if report.verdict.accepted else 'rejected'}")
            for event in report.events:
                out(f"      {event}")


def main() -> int:
    if STORE.exists():
        try:
            shutil.rmtree(STORE)
        except OSError as exc:
            sys.exit(f"can't reset {STORE} ({exc}); stop `metacog serve` first")
    asyncio.run(replay(build(STORE)))
    print(f"\nstore: {STORE}\nnext:  metacog serve examples.calendar.drift_demo:agent   "
          "(then open http://127.0.0.1:8620)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
