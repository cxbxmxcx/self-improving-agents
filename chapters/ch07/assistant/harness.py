"""Put Chapter 6's assistant on helix's packaged harness and give it a decision tier that a code search can fill.

Listing 6.5 wired a recorder, a detector, a learner, a gate and a controller around the agent by hand. helix's
`Metacognition` holds those parts in one object. A `ProgramTier` adds what this chapter searches: a learning component
that stores memories and says what serves each one. Everything here is a setting of helix or a way to print what
happened, and the chapter adds no mechanism of its own.

The agent is gpt-5.4-mini, as in Chapter 6. All code is written by the builder, the strongest model available: a
search's rewrites, the programs a component asks for, and a relearned function.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from typing import Any

import helix
from chapters.ch06.assistant import generator, generator_drift
from chapters.ch06.assistant.learning import act, bootstrapped_source
from chapters.ch07.assistant import requests
from chapters.ch07.assistant.desk import (
    FIND_COMMON_STARTS,
    FIND_SHARED_STARTS,
    INTENT,
    assistant,
    check,
    contract,
    desk,
    world_for,
)
from helix.deciders import Decider, MissingAnswer, recorded_decider
from helix.metacognition import (
    CALENDAR,
    Agent,
    Controller,
    Gate,
    LearningBudget,
    Metacognition,
    RunConfig,
    Runner,
    ValueLearning,
)
from helix.metacognition.decide import JEV_MODEL
from helix.metacognition.decisions import ProgramSearch, ProgramTier
from helix.metacognition.exceptions import AgentsException
from helix.metacognition.gap import Good, outcome_of
from helix.metacognition.model_settings import BUILDER_MODEL
from helix.metacognition.parts import Verdict
from helix.metacognition.pricing import PRICES
from helix.metacognition.recording import (
    BudgetSpent,
    CappedRecording,
    MissingReply,
    Spend,
)

HERE = Path(__file__).parent
REPO_ROOT = HERE.parents[2]
RECORDED = HERE / "recorded"
RUNS = HERE.parent / "runs"
CH06_RUNS = HERE.parents[1] / "ch06" / "runs"
SEEDS = Path(helix.__file__).parent / "metacognition" / "programs"
BUILDER = BUILDER_MODEL
GOOD = Good()  # distances from good: 0 with no language model, 0.4 right by the model, 1 not done, 3 a wrong action
KINDS = ("one meeting", "two meetings", "same time", "near miss")
FRESH_WEEKS = range(901, 907)  # first-class weeks no request set uses, held back for the function's gate


# ---------------------------------------------------------------- what Chapter 6 left


def kept_procedure() -> tuple[str, str]:
    """The function listing 6.3's kept mechanism wrote, and that mechanism's name."""
    summary = CH06_RUNS / "learning_search" / "learning_search.json"
    if not summary.exists():
        print("Listing 6.3 has not been run here, so the assistant starts with Chapter 6's reference function.")
        return bootstrapped_source(), "reference"
    kept = json.loads(summary.read_text(encoding="utf-8"))["kept"]
    return (summary.parent / f"learned_{kept}.py").read_text(encoding="utf-8"), kept


def second_procedure() -> tuple[str, str]:
    """The function listing 6.4 learned for Chapter 6's second class, or that class's reference function."""
    learned = CH06_RUNS / "generalization" / "learned_find_common_starts.py"
    if learned.exists():
        return learned.read_text(encoding="utf-8"), "listing_6_4"
    print("Listing 6.4 has not been run here, so the second class starts with Chapter 6's reference function.")
    return bootstrapped_source("same_time"), "reference"


def controller() -> Controller:
    """The reasoning listing 6.2 measured: its floor with a trusted memory, and its floor without one."""
    floors = CH06_RUNS / "reasoning_floor.json"
    if floors.exists():
        return Controller.from_sweeps(floors, memory="bootstrapped_memory", no_memory="no_memory")
    return Controller({"memory": "none", "no_memory": "high"})


def function_gate(changed: bool = False) -> Gate:
    """The held-back weeks a relearned function has to pass: fresh ones, or Chapter 6's from after Dana's note changed."""
    held_back = generator_drift.task_set("gate") if changed else [generator.generate(seed) for seed in FRESH_WEEKS]
    return Gate(cases=[task["id"] for task in held_back])


async def on_week(source: str, case_id: str) -> tuple[float, str, str]:
    """Try a function on the week behind a request, however the request was worded."""
    return await act(source, case_id.split("~")[0])


# ---------------------------------------------------------------- the harness, as settings


def seed(name: str = "component_llm") -> tuple[str, str]:
    """One of helix's starting components. The default keeps every request on the language model."""
    return name, (SEEDS / f"{name}.py").read_text(encoding="utf-8")


def code_search(method: str = "dgm", archive: Path | str = ":memory:", **changes: Any) -> ProgramSearch:
    """A search over learning components: the method, how many rewrites it may measure, and who writes them."""
    settings: dict[str, Any] = {"target": "mechanism", "population": 6, "generations": 1, "designer_model": BUILDER,
                                "penalty": 2.0, "stop_when_admitted": True, "seed": 7,
                                "label": f"calendar, {method}"}
    return ProgramSearch(method=method, archive=str(archive), **{**settings, **changes})


def decision_tier(jev: Decider, components: Sequence[tuple[str, str]], *, search: ProgramSearch | None = None,
                  gate: str = "gate", seen: str = "search", **settings: Any) -> ProgramTier:
    """What a candidate component is tried on, what the gate holds back, and the bars a program has to clear."""
    settings.setdefault("contract", contract())
    return ProgramTier(desk=desk(), decider=jev, mechanisms=list(components), search=search,
                       labeled={r["id"]: r for r in requests.load(gate)},
                       search_cases={r["id"]: r for r in requests.load(seen)}, world=world_for,
                       bar=0.9, max_wrong=0, routed_bar=0.7, read_only_handoffs=True, writer_model=BUILDER,
                       # A program that acts on a request not meant for action is switched off like any wrong action.
                       gating_sources=("verifier", INTENT), **settings)


def harness(tier: ProgramTier | None = None, *, second_class: bool = False,
            **changes: Any) -> tuple[Agent, Metacognition]:
    """Chapter 6's assistant and loop in one object, with the function it learned trusted and `tier` beside it.

    `second_class` also gives the assistant the function for the same meeting at the same time on several days.
    Nothing changes under the functions in that listing, and the gate here holds cases for the first function only,
    so there the detector is off: no verdict can quarantine a function and start a relearning that could not pass.
    `changes` replace any of the loop's settings, such as its store or its learning budget.
    """
    source, kept = kept_procedure()
    procedures = [FIND_SHARED_STARTS, FIND_COMMON_STARTS] if second_class else [FIND_SHARED_STARTS]
    settings: dict[str, Any] = {
        "verifier": check, "controller": controller(), "recall_k": len(procedures),
        "learner": ValueLearning(CALENDAR), "writer_model": BUILDER, "practice": on_week,
        "gate": function_gate(), "budget": LearningBudget(attempts_per_day=2),
        "relearn_after": 4, "learning": "inline", "prices": PRICES, "programs": tier,
        # Only the checker's verdict on real work can quarantine the function. A verdict on a request's intent
        # cannot.
        "gating_sources": () if second_class else ("verifier",)}
    meta = Metacognition(procedures, **{**settings, **changes})
    meta.install("find_shared_starts", source, created_by=kept)
    if second_class:
        second, origin = second_procedure()
        meta.install("find_common_starts", second, created_by=origin)
    if tier is not None:
        meta.programs.paused = True  # by day the assistant serves; a listing says when it may improve
    meta.agent = assistant(metacognition=meta)  # the agent a component's routes are tried on
    return meta.agent, meta


def recordings(record: bool, budget: float) -> tuple[CappedRecording, Decider]:
    """Saved replies for the models and for Jev. Replay needs no key, and recording stops before passing `budget`."""
    mode = "record" if record else "replay"
    if record:
        from dotenv import load_dotenv

        load_dotenv(REPO_ROOT / ".env")
        os.environ.setdefault("TYPESAFE_API_KEY", os.environ.get("JEV_API_KEY", ""))
    return (CappedRecording(RECORDED / "models.jsonl", mode=mode, spend=Spend(budget, PRICES)),
            recorded_decider(RECORDED / "jev.jsonl", mode=mode, model=JEV_MODEL))


def run(main: Callable[[argparse.Namespace], Awaitable[None]], args: argparse.Namespace) -> None:
    """Run a listing, and say plainly why it stopped if the budget ran out or a reply was never recorded."""
    try:
        asyncio.run(main(args))
    except BudgetSpent as exc:
        sys.exit(f"Stopped: {exc}")
    except (MissingReply, MissingAnswer) as exc:
        sys.exit(f"Stopped: {exc.args[0]}. Replay reads saved replies only; run the listing with --record to add them.")


# ---------------------------------------------------------------- serving, and what to print


def kind_of(task: dict[str, Any]) -> str:
    return "near miss" if task["expect"] == "handoff" else task["kind"]


def _row(stage: str, task: dict[str, Any], right: bool, program: str | None, effort: str | None, result: Any,
         events: Sequence[str] = (), memory: str | None = None) -> dict[str, Any]:
    cost = getattr(result, "cost", None)
    return {"stage": stage, "id": task["id"], "kind": kind_of(task), "right": right, "effort": effort,
            "outcome": outcome_of(right, program=program, should_act=task["expect"] == "act"), "memory": memory,
            "dollars": float(getattr(cost, "dollars", 0.0) or 0.0), "seconds": float(getattr(cost, "seconds", 0.0) or 0.0),
            "events": list(events)}


async def serve(agent: Agent, tasks: Sequence[dict[str, Any]], stage: str, log: list[dict[str, Any]]) -> None:
    """Serve requests as they arrive. The loop records and judges each run, and may quarantine or relearn."""
    said: set[str] = set()
    for task in tasks:
        config = RunConfig(max_tool_calls=24, case_id=task["id"], case_data=task)
        result = await Runner.run(agent, task["text"], context=world_for(task), max_turns=16, run_config=config)
        report = result.metacognition
        log.append(_row(stage, task, bool(report.verdict.accepted), report.program, report.effort, result,
                        report.events, report.memory))
        for event in report.events:
            routine = event.startswith(("program: used memory", "program: handed off", "program: routed"))
            # A program that breaks is worth a line, once; a routine route is not.
            if (not routine or "failed" in event) and event not in said:
                said.add(event)
                print(f"   {task['id']}: {event}")


async def night(meta: Metacognition) -> list[str]:
    """What happens while nobody is asking: if the day opened a gap, the decision tier improves offline."""
    events: list[str] = []
    meta.programs.paused = False
    try:
        await meta.programs.maybe_improve(events)
    finally:
        meta.programs.paused = True
    return events


async def measure(agent: Agent, meta: Metacognition, tasks: Sequence[dict[str, Any]], effort: str, stage: str,
                  log: list[dict[str, Any]], *, at_once: int = 6) -> None:
    """The language model alone at one reasoning level, frozen: nothing is recorded and nothing is learned."""
    slots = asyncio.Semaphore(at_once)

    async def one(task: dict[str, Any]) -> dict[str, Any]:
        config = RunConfig(metacognition_mode="frozen", effort=effort, max_tool_calls=24, case_id=task["id"],
                           case_data=task)
        async with slots:
            try:
                result = await Runner.run(agent, task["text"], context=world_for(task), max_turns=16,
                                          run_config=config)
            except AgentsException:
                return _row(stage, task, False, None, effort, None)
        verdict = Verdict.coerce(meta.verifier(result.context_wrapper, result))
        return _row(stage, task, bool(verdict and verdict.accepted), None, effort, result)

    log.extend(await asyncio.gather(*(one(task) for task in tasks)))


def gap(rows: Sequence[dict[str, Any]]) -> float:
    """The mean distance from good over the requests served: 0 is good."""
    return sum(GOOD.gap(row["outcome"]) for row in rows) / len(rows) if rows else 0.0


def served_by(row: dict[str, Any]) -> str:
    if row["outcome"] in ("program", "wrong"):
        return "program"
    return "read-only" if row["outcome"] in ("read_only", "lost") else f"model at {row['effort']}"


def summarize(log: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Each stage in one line, then each kind of request: how many were right, what served them, and the cost."""
    summary: dict[str, dict[str, Any]] = {}
    for stage in dict.fromkeys(row["stage"] for row in log):
        rows = [row for row in log if row["stage"] == stage]
        cell = {"requests": len(rows), "right": sum(r["right"] for r in rows), "gap": round(gap(rows), 3),
                "wrong_actions": sum(r["outcome"] == "wrong" for r in rows),
                "dollars": sum(r["dollars"] for r in rows) / len(rows),
                "seconds": sum(r["seconds"] for r in rows) / len(rows), "kinds": {}}
        print(f"{stage}: {cell['right']}/{cell['requests']} right, gap {cell['gap']:.2f}, "
              f"${cell['dollars']:.5f} and {cell['seconds']:.2f} s a request")
        for kind in KINDS:
            part = [r for r in rows if r["kind"] == kind]
            if not part:
                continue
            by: dict[str, int] = {}
            for r in part:
                by[served_by(r)] = by.get(served_by(r), 0) + 1
            cell["kinds"][kind] = {"requests": len(part), "right": sum(r["right"] for r in part), "served_by": by,
                                   "dollars": sum(r["dollars"] for r in part) / len(part)}
            served = ", ".join(f"{name} {count}" for name, count in by.items())
            print(f"   {kind:<12} {sum(r['right'] for r in part):>2}/{len(part):<2} "
                  f"${sum(r['dollars'] for r in part) / len(part):.5f}  {served}")
        summary[stage] = cell
    return summary


def save(folder: Path, name: str, log: Sequence[dict[str, Any]], meta: Metacognition | None = None,
         **more: Any) -> None:
    """Write a listing's rows for the figures, and the program the assistant now trusts, if it has one."""
    folder.mkdir(parents=True, exist_ok=True)
    data: dict[str, Any] = {"rows": list(log), **more}
    current = meta.programs.current() if meta is not None and meta.programs is not None else None
    if current is not None:
        data["memories"] = meta.programs.memories()
        (folder / f"{name}_router.py").write_text(current["source"], encoding="utf-8", newline="\n")
    (folder / f"{name}.json").write_text(json.dumps(data, indent=1, default=str), encoding="utf-8")
