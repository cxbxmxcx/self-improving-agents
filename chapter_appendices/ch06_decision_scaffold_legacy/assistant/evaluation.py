"""Evaluation support shared by listings 6.3 and 6.4.

Every run starts from the same week with only its request's prepared
memory, and memory never grows during evaluation. Conditions therefore
differ only in what the meta level decides, which is what a controlled
comparison needs. This module holds the conditions, one fresh run of one
request, the paired runs, and the measurements the chapter reports.
"""

from __future__ import annotations

import json
from contextlib import nullcontext
from importlib import import_module
from pathlib import Path
from typing import Any

from chapters.ch05.personal.encoder import embed
from chapter_appendices.ch06_decision_scaffold_legacy.assistant.environment import AssistantEnvironment, preload_memory
from helix.artifact import Artifact
from helix.eval.paired import PairedRuns
from helix.memory.base import MemoryContext
from helix.memory.episodic import EpisodicMemory
from helix.scaffold import config_artifact, decider_usage, record, thresholds_artifact
from helix.scaffold.common import pinned
from helix.signals.cost import CostWeights
from helix.signals.metacognition import MetacognitionObjective, objective_score
from helix.signals.outcome import OutcomeSignal
from helix.signals.safety import SafetySignal
from helix.trajectory import Trajectory

listing = import_module("chapter_appendices.ch06_decision_scaffold_legacy.02_metacognitive_assistant")
# What a dollar and a second are worth in outcome points: a cent costs 0.1,
# and so do ten seconds of waiting. The chapter's comparison depends on them.
WEIGHTS = CostWeights(dollars=10.0, seconds=0.01)
LLM_DECIDER = "llm/gpt-5.4-mini"


def fresh_tier() -> EpisodicMemory:
    return EpisodicMemory(":memory:", utility_weight=0.35, recency_weight=0.0, embedding_fn=embed)


def stamped_reward(trajectory: Trajectory) -> float | None:
    """The reward the run's own environment stamped when it ended."""
    return trajectory.metadata.get("reward")


def objective(env: AssistantEnvironment) -> MetacognitionObjective:
    return MetacognitionObjective(OutcomeSignal(stamped_reward), SafetySignal(env.consequential, confirmed=env.confirmed),
                                  weights=WEIGHTS, prices=listing.PRICES)


def baseline_config() -> Artifact:
    """The seed configuration without the monitor's questions, for the
    always-fast and always-deliberate conditions: they decide nothing."""
    seed = listing.SEED_CONFIG
    routes = {n: q for n, q in seed["questions"].items() if q["hook"] == "route"}
    return config_artifact("assistant.metacognition.config", {
        "questions": routes, "rules": [r for r in seed["rules"] if r["question"] in routes]})


def searched_artifacts(path: Path) -> dict[str, Artifact] | None:
    """The configuration and thresholds a search kept, as children of the seed."""
    if not path.exists():
        return None
    found = json.loads(path.read_text(encoding="utf-8"))
    seed = listing.seed_artifacts()
    return {"config": seed["config"].mutate(json.dumps(found["config"], indent=2), created_by="search",
                                            version=found["config_version"]),
            "thresholds": seed["thresholds"].mutate(json.dumps(found["thresholds"], indent=2), created_by="search",
                                                    version=found["thresholds_version"])}


def conditions(jev: Any, *, llm: Any = None, searched: dict[str, Artifact] | None = None) -> dict[str, dict]:
    """The comparison's conditions: two that never decide, and three that do."""
    baseline = {"config": baseline_config()}
    out = {
        "always fast": {"artifacts": baseline, "pins": {"route": "fast", "tool": "allow"}, "decider": jev},
        "always deliberate": {"artifacts": baseline, "pins": {"route": "deliberate", "tool": "allow"}, "decider": jev},
        "seed configuration": {"artifacts": {}, "pins": None, "decider": jev},
    }
    if searched is not None:
        out["searched configuration"] = {"artifacts": searched, "pins": None, "decider": jev}
    if llm is not None:
        out["LLM decision model"] = {"artifacts": {}, "pins": None, "decider": llm}
    return out


async def run_request(request: dict, *, env: AssistantEnvironment, decider: Any, live: bool = False,
                      artifacts: dict[str, Artifact] | None = None, pins: dict[str, str] | None = None,
                      memory: bool = True) -> Trajectory:
    """One request from the starting week, with its prepared memory and none written."""
    env.reset()
    env.begin(request)
    tier = fresh_tier()
    if memory:
        await preload_memory(tier, request)
    agent = listing.build_assistant(env, tier, decider, live=live, artifacts=artifacts)
    agent.memory_hooks.read_only = True
    with pinned(**pins) if pins else nullcontext():
        _, trajectory = await agent.run(request["text"],
                                        context=MemoryContext(user_id=request["user"], session_id=request["session"]))
    return trajectory


async def measure(env: AssistantEnvironment, request: dict, trajectory: Trajectory) -> dict[str, Any]:
    """One run's row: the objectives, the route, and what the meta level did."""
    measured = await objective(env).measure(listing.seed_artifacts()["config"], trajectory=trajectory)
    objectives = measured.metadata["objectives"]
    safety = measured.metadata["components"][1]["metadata"]
    meta = record(trajectory)
    usage = decider_usage(trajectory)
    answer = trajectory.final_output if isinstance(trajectory.final_output, str) else ""
    # Why the user did or did not accept it, so a failure can be read from the report.
    verdict = env.check(answer, held=env.held(trajectory))
    return {"request": request["id"], "category": request["category"], "kind": request["truth"]["kind"],
            "reason": verdict["reason"], "answer": answer[:400],
            "score": measured.score, "outcome": objectives["outcome"], "safe": objectives["safe"],
            "dollars": objectives["dollars"], "seconds": objectives["seconds"], "route": meta.get("route"),
            "revised": bool(meta.get("revised")), "calls": safety.get("calls", []),
            "decisions": usage.calls, "decision_dollars": usage.dollars,
            "route_answers": {d["question"]: d["value"] for d in meta["decisions"] if d["hook"] == "route"}}


async def build_paired(requests: list[dict], *, decider: Any, live: bool = False,
                       path: Path | None = None) -> PairedRuns:
    """Run each request down every path once; reuse the file at `path` when it exists."""
    paired = PairedRuns(path)
    if path is not None and path.exists() and {r["id"] for r in requests} <= set(paired.entries):
        return paired
    env = AssistantEnvironment()
    probe = listing.build_assistant(env, fresh_tier(), decider, live=live)

    async def run(_agent: Any, request: dict) -> tuple[None, Trajectory]:
        return None, await run_request(request, env=env, decider=decider, live=live)

    await paired.build(probe, requests, run, objective(env))
    return paired


def best_route(paired: PairedRuns, request_id: str) -> str | None:
    """The route hindsight prefers for a request, from its paired runs."""
    values = {route: result.value(WEIGHTS) for route, result in paired.route_results(request_id, WEIGHTS).items()}
    return max(values, key=values.get) if values else None


# ---------------------------------------------------------------- the chapter's measures


def summarize(rows: list[dict]) -> dict[str, Any]:
    n = len(rows) or 1
    return {"requests": len(rows), "accepted": sum(1 for r in rows if r["outcome"] == 1.0),
            "unsafe": sum(1 for r in rows if not r["safe"]), "score": sum(r["score"] or 0.0 for r in rows) / n,
            "dollars": sum(r["dollars"] for r in rows) / n, "seconds": sum(r["seconds"] for r in rows) / n,
            "decisions": sum(r["decisions"] for r in rows) / n}


def brier(points: list[tuple[float, int]]) -> float | None:
    return sum((p - y) ** 2 for p, y in points) / len(points) if points else None


def auroc(points: list[tuple[float, int]]) -> float | None:
    """How often a request that needed the move scored above one that did not."""
    positives = [p for p, y in points if y]
    negatives = [p for p, y in points if not y]
    if not positives or not negatives:
        return None
    wins = sum(1.0 if a > b else 0.5 if a == b else 0.0 for a in positives for b in negatives)
    return wins / (len(positives) * len(negatives))


def calibration(rows: list[dict], paired: PairedRuns) -> dict[str, dict]:
    """Each routing probability against whether its move paid in hindsight."""
    labels = {"ambiguous_reference": "ask", "multi_event_limit": "deliberate"}
    out = {}
    for question, route in labels.items():
        points = [(r["route_answers"][question], int(best_route(paired, r["request"]) == route))
                  for r in rows if r["route_answers"].get(question) is not None]
        out[question] = {"points": len(points), "positives": sum(y for _, y in points),
                         "brier": brier(points), "auroc": auroc(points)}
    return out


def precision_recall(hits: int, chosen: int, needed: int) -> dict[str, float | None]:
    return {"chosen": chosen, "needed": needed, "precision": hits / chosen if chosen else None,
            "recall": hits / needed if needed else None}


def interventions(rows: list[dict], paired: PairedRuns) -> dict[str, dict]:
    """Whether each move was made where it paid, and only there."""
    asked = [r for r in rows if r["route"] == "ask"]
    needs_ask = [r for r in rows if r["kind"] == "ask"]
    deliberated = [r for r in rows if r["route"] == "deliberate"]
    needs_thought = [r for r in rows if best_route(paired, r["request"]) == "deliberate"]
    calls = [c for r in rows for c in r["calls"]]
    held_right = sum(1 for c in calls if c["cancelled"] and c["consequential"])
    return {
        "ask": precision_recall(sum(1 for r in asked if r["kind"] == "ask"), len(asked), len(needs_ask)),
        "deliberate": precision_recall(sum(1 for r in deliberated if r in needs_thought), len(deliberated),
                                       len(needs_thought)),
        "confirm": precision_recall(held_right, sum(1 for c in calls if c["cancelled"]),
                                    held_right + sum(1 for c in calls if c["violation"])),
        "revise": {"revised": sum(1 for r in rows if r["revised"]),
                   "accepted_after": sum(1 for r in rows if r["revised"] and r["outcome"] == 1.0)},
    }


def score_of(objectives: dict) -> float | None:
    return objective_score(objectives, WEIGHTS)


def thresholds_from(values: dict[str, float]) -> Artifact:
    return thresholds_artifact("assistant.routing.thresholds", values)


# A live gpt-5.4-mini run averaged about half a cent in the 2026-10-12 walkthrough.
DOLLARS_PER_RUN = 0.006


def estimate(runs: int) -> str:
    return (f"A live pass makes about {runs} agent runs, roughly ${runs * DOLLARS_PER_RUN:.2f} at "
            f"${DOLLARS_PER_RUN} a run, plus a few cents of decision-model calls. Add --yes to start it.")
