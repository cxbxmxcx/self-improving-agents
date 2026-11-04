"""Paired runs: each request run once down every path, kept for scoring variants.

A routing variant changes only which path a request takes. Once each request
has been run down every path (fast, deliberate, and ask, and, where it makes
a call the monitor checks, with and without the confirmation), a variant is
scored by asking the decision model its routing and confirmation questions
again on the recorded states and reading the result of the path it picks.
That turns an agent run per request into one decision call per request.

Planner and reflector variants change what happens on a path, so they need
new runs. So does a variant that changes the progress, output, or reflect
questions, which the cache refuses to score. Each request may make at most
one checked call for its confirmation to be scored this way.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Iterable
from pathlib import Path
from typing import Any

from helix._async import resolve
from helix.artifact import Artifact
from helix.deciders.base import DecisionUsage, ask_measured
from helix.scaffold.common import META_KEY, pinned
from helix.scaffold.config import DecisionConfig, apply_rules, load_config, load_thresholds
from helix.scaffold.router import choose_route
from helix.search.objectives import aggregate
from helix.signal import GapMeasurement, Signal
from helix.signals.cost import CostWeights
from helix.signals.metacognition import objective_score
from helix.signals.routing import BranchResult
from helix.trajectory import Trajectory

ROUTES = ("fast", "deliberate", "ask")
# Hooks a cached run has already decided; a variant must leave them alone.
FIXED_HOOKS = ("progress", "output", "reflect")

# (agent, request) -> (answer, trajectory)
RunFn = Callable[[Any, Any], Awaitable[tuple[Any, Trajectory]]]


def _request_id(request: Any) -> str:
    return str(request["id"] if isinstance(request, dict) else request.id)


def _fixed_part(config: DecisionConfig) -> dict[str, Any]:
    """The questions and rules of the hooks a cached run has already decided."""
    hook_of = {name: spec.hook for name, spec in config.questions.items()}
    return {
        hook: {"questions": {n: q.model_dump(exclude_none=True) for n, q in config.questions.items() if q.hook == hook},
               "rules": [r.model_dump(exclude_none=True) for r in config.rules if hook_of[r.question] == hook]}
        for hook in FIXED_HOOKS
    }


class PairedRuns:
    """The paired-run cache: build it once with live runs, then score variants from it."""

    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path) if path else None
        self.entries: dict[str, dict[str, Any]] = {}
        self.fixed: dict[str, Any] = {}
        if self.path is not None and self.path.exists():
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self.entries, self.fixed = data["entries"], data.get("fixed", {})

    def save(self, path: Path | str | None = None) -> None:
        target = Path(path) if path else self.path
        if target is None:
            raise ValueError("no path to save the paired runs to")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps({"fixed": self.fixed, "entries": self.entries}, indent=1, default=str),
                          encoding="utf-8")

    # ---------------------------------------------------------------- building

    async def build(self, agent: Any, requests: Iterable[Any], run: RunFn, objective: Signal, *,
                    prepare: Callable[[Any], Any] | None = None, routes: tuple[str, ...] = ROUTES,
                    keep_trajectories: bool = False) -> None:
        """Run every request down every route, and down both confirmation
        verdicts where the run makes a checked call.

        `prepare(request)` resets the environment before each run, so every
        path starts from the same state. `objective` must report
        `metadata["objectives"]`, as `MetacognitionObjective` does.
        """
        scaffold = getattr(agent, "scaffold", None)
        if scaffold is None:
            raise ValueError("paired runs need an agent with a metacognitive scaffold")
        self.fixed = _fixed_part(scaffold.decision_config)
        for request in requests:
            entry: dict[str, Any] = {"task": None, "route_state": None, "tool_states": {}, "branches": {}}
            for route in routes:
                trajectory = await self._run(agent, request, run, prepare, route=route, tool="allow")
                states = (trajectory.metadata.get(META_KEY) or {}).get("states", [])
                entry["task"] = trajectory.task
                if entry["route_state"] is None:
                    entry["route_state"] = next((s["state"] for s in states if s["hook"] == "route"), None)
                checked = next((s for s in states if s["hook"] == "tool"), None)
                if checked is None:
                    entry["branches"][route] = await self._branch(trajectory, objective, scaffold.config, keep_trajectories)
                    continue
                entry["tool_states"][route] = {"tool": checked.get("tool"), "state": checked["state"]}
                entry["branches"][f"{route}|allow"] = await self._branch(
                    trajectory, objective, scaffold.config, keep_trajectories)
                confirmed = await self._run(agent, request, run, prepare, route=route, tool="confirm")
                entry["branches"][f"{route}|confirm"] = await self._branch(
                    confirmed, objective, scaffold.config, keep_trajectories)
            self.entries[_request_id(request)] = entry
        if self.path is not None:
            self.save()

    @staticmethod
    async def _run(agent: Any, request: Any, run: RunFn, prepare: Callable[[Any], Any] | None,
                   **pins: str) -> Trajectory:
        if prepare is not None:
            await resolve(prepare(request))
        with pinned(**pins):
            _, trajectory = await run(agent, request)
        return trajectory

    @staticmethod
    async def _branch(trajectory: Trajectory, objective: Signal, candidate: Artifact,
                      keep: bool) -> dict[str, Any]:
        measured = await objective.measure(candidate, trajectory=trajectory)
        objectives = (measured.metadata or {}).get("objectives")
        if not isinstance(objectives, dict):
            raise ValueError("the objective must report metadata['objectives'], as MetacognitionObjective does")
        branch = {"objectives": objectives, "measurement": measured.to_dict()}
        if keep:
            branch["trajectory"] = trajectory.to_dict()
        return branch

    # ---------------------------------------------------------------- scoring

    def changed_hooks(self, config: DecisionConfig) -> list[str]:
        """The already-decided hooks a variant changes; scoring it needs new runs."""
        mine = _fixed_part(config)
        return [hook for hook in FIXED_HOOKS if mine[hook] != self.fixed.get(hook)]

    async def evaluate(self, config: DecisionConfig | Artifact, thresholds: dict[str, float] | Artifact,
                       decider: Any, *, weights: CostWeights | None = None, requests: Iterable[str] | None = None,
                       signal_id: str | None = None) -> GapMeasurement:
        """Score a configuration and thresholds on the cached requests.

        Each request's measurement adds the variant's own decision calls to
        the cached path's dollars and seconds, and its feedback names the
        path hindsight preferred, which is what a reflective search reads.
        """
        config = load_config(config) if isinstance(config, Artifact) else config
        thresholds = load_thresholds(thresholds) if isinstance(thresholds, Artifact) else dict(thresholds)
        changed = self.changed_hooks(config)
        if changed:
            raise ValueError(f"this variant changes the {', '.join(changed)} hooks, which the cached runs "
                             "already decided; score it with new runs")
        weights = weights or CostWeights()
        rows = []
        for rid in (list(requests) if requests is not None else list(self.entries)):
            rows.append((rid, await self._score_one(self.entries[rid], config, thresholds, decider, weights)))
        result = aggregate(rows, signal_id=signal_id)
        result.metadata["paths"] = {rid: m.metadata["path"] for rid, m in rows}
        return result

    async def _score_one(self, entry: dict[str, Any], config: DecisionConfig, thresholds: dict[str, float],
                         decider: Any, weights: CostWeights) -> GapMeasurement:
        spent = DecisionUsage()
        route = choose_route(await self._actions(config, "route", entry["route_state"], thresholds, decider, spent))
        path = route
        checked = entry["tool_states"].get(route)
        if checked is not None:
            actions = await self._actions(config, "tool", checked["state"], thresholds, decider, spent)
            path = f"{route}|{'confirm' if 'confirm' in actions else 'allow'}"
        branch = entry["branches"].get(path)
        if branch is None:
            raise KeyError(f"no cached run for {entry['task']!r} down the {path} path")
        objectives = dict(branch["objectives"])
        objectives["dollars"] = float(objectives.get("dollars") or 0.0) + spent.dollars
        objectives["seconds"] = float(objectives.get("seconds") or 0.0) + spent.seconds
        score = objective_score(objectives, weights)
        best, best_score = self._best(entry, weights)
        if best == path or best_score is None or (score is not None and score >= best_score):
            feedback = f"{entry['task']!r} went down the {path} path, the best in hindsight"
        else:
            shown = "no score" if score is None else f"{score:.2f}"
            feedback = f"{entry['task']!r} went down the {path} path ({shown}); {best} would have scored {best_score:.2f}"
        return GapMeasurement(score=score, feedback=feedback, triggered=objectives.get("safe") is False,
                              raw_value=score, cost=spent.as_cost(),
                              metadata={"objectives": objectives, "path": path, "best": best})

    @staticmethod
    async def _actions(config: DecisionConfig, hook: str, state: Any, thresholds: dict[str, float],
                       decider: Any, spent: DecisionUsage) -> set[str]:
        questions = config.questions_for(hook)
        answers = {}
        if questions and state is not None:
            answers, call = await ask_measured(decider, state, questions)
            spent.add(call)
        actions, _ = apply_rules(config, hook, answers, thresholds)
        return actions

    @staticmethod
    def _best(entry: dict[str, Any], weights: CostWeights) -> tuple[str | None, float | None]:
        scored = [(key, objective_score(b["objectives"], weights)) for key, b in entry["branches"].items()]
        scored = [(key, s) for key, s in scored if s is not None]
        return max(scored, key=lambda row: row[1]) if scored else (None, None)

    def evaluator(self, decider: Any, *, config: DecisionConfig | Artifact | None = None,
                  thresholds: dict[str, float] | Artifact | None = None,
                  weights: CostWeights | None = None) -> Callable[[Artifact, Any], Awaitable[GapMeasurement]]:
        """A GEPA or ThresholdSearch evaluator: hold one of the configuration
        and the thresholds fixed, and score candidates for the other."""
        if (config is None) == (thresholds is None):
            raise ValueError("fix exactly one of config and thresholds; candidates supply the other")

        async def evaluate(candidate: Artifact, questions: Any = None) -> GapMeasurement:
            ids = [q.id for q in questions] if questions else None
            if config is None:
                return await self.evaluate(candidate, thresholds, decider, weights=weights, requests=ids)
            return await self.evaluate(config, candidate, decider, weights=weights, requests=ids)

        return evaluate

    # ---------------------------------------------------------------- routing regret

    def route_results(self, request_id: str, weights: CostWeights | None = None) -> dict[str, BranchResult]:
        """Each route's result for a request, taking the better confirmation
        verdict where there were two; an unsafe run counts as a zero outcome."""
        weights = weights or CostWeights()
        results: dict[str, BranchResult] = {}
        best: dict[str, float] = {}
        for key, branch in self.entries[request_id]["branches"].items():
            route = key.split("|", 1)[0]
            objectives = branch["objectives"]
            score = objective_score(objectives, weights)
            if score is None or score <= best.get(route, -1.0):
                continue
            best[route] = score
            outcome = 0.0 if objectives.get("safe") is False else float(objectives.get("outcome") or 0.0)
            results[route] = BranchResult(outcome, float(objectives.get("dollars") or 0.0),
                                          float(objectives.get("seconds") or 0.0))
        return results

    def lookup(self, trajectory: Trajectory) -> dict[str, BranchResult] | None:
        """Route results for the request a trajectory ran, for `RoutingSignal`."""
        rid = next((rid for rid, e in self.entries.items() if e["task"] == trajectory.task), None)
        return self.route_results(rid) if rid is not None else None
