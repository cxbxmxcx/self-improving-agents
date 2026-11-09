"""The decision tier: a rung below no reasoning, where a learned program replaces the model.

The loop chapter 6 built lowers reasoning once a procedure is trusted. This one takes the next step, and it is the
same loop:

    signal   the store's runs: what the LLM costs on this work, and how much a program hands back
    gap      the LLM still serves work a program could, or a program has been switched off, or hands off too much
    improve  a mechanism writes a decision program from the judged runs (helix.metacognition.decide)
    search   if the gate refuses what the mechanisms write, a search rewrites a mechanism's own source and tries again,
             or edits the best program they wrote, with what already works frozen
    gate     a program is admitted only if it clears the bar on labeled cases, with at most `max_wrong` wrong actions
    switch   a trusted program runs before the agent; an action it takes ends the run with no model. A class that
             was switched off comes back when the program it has passes the gate again, or a new one does. A hand-off
             continues into the agent in the same run, at the reasoning and with the memory the program named, if it
             named them. Each action is a task class with its own detector window, so a class that starts failing is
             switched off alone and goes back to the LLM until a new program passes.

Programs are kept as versions of `DECISION` in the store, beside the procedures. Runs a program served are
recorded with the reasoning setting "program". Jev's calls are charged as the run's decider calls.
"""

from __future__ import annotations

import inspect
import json
import random
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field, replace
from types import SimpleNamespace
from typing import Any

from helix.archive import SQLiteArchive
from helix.artifact import ArtifactKind, genesis
from helix.memory.learning_mechanisms import Contract, Episode
from helix.memory.mechanism_runtime import MechanismBudget, UnsafeMechanism, check_mechanism
from helix.scaffold.common import META_KEY
from helix.search.base import SearchBudget
from helix.search.methods import build_search
from helix.signal import GapMeasurement
from helix.trajectory import StepKind, Trajectory

from helix.deciders.base import Decider, MissingAnswer
from helix.metacognition.decide import Desk, Kit, UnsafeProgram, check_program, frozen_change, run_program, serve
from helix.metacognition.api import SkillAPI
from helix.metacognition.mechanisms import contract_text
from helix.metacognition.parts import Detector, GateResult
from helix.metacognition.prompts import GENERIC
from helix.metacognition.selection import LearningProblem, run_tournament
from helix.metacognition.sources import SourceLearner, source_id
from helix.metacognition.model_settings import DEFAULT_MODEL
from helix.metacognition.pricing import PRICES, run_cost
from helix.metacognition.result import RunResult

DECISION = "decision_program"
RUNG = "program"
HANDOFF_REWARD = 0.25  # better than a wrong action, far from serving the request
ROUTED_REWARD = 0.6  # the LLM got it right where the program sent it: served, but not for free
ROUTED_WRONG = 0.1  # the program's choice of route failed; the LLM's own mistake is not a program's wrong action
_LOOP = "decision:loop"
_OUTCOMES = "decision:outcomes"

PROGRAM_LAB = """A program-learning mechanism is Python source, and it is a whole learning component: what it learns from
experience, how that is stored, and how it is used. It writes a decision program: code that keeps what was learned as
constants, and for each request retrieves what applies and decides how to serve it. Some requests the program serves
itself with no language model, calling tools and procedures and asking Jev, a small model that returns typed
judgments, for what code can't decide. Others it hands to the language model, naming the reasoning level to use and the
memory to read. The mechanism is given runs a language model already did, each judged by a checker. It is scored on
held-out requests: an action must be right, a request outside the job must be handed off, a wrong action is penalized
heavily, a request routed to the language model earns less than one served without it, and a plain hand-off of work the
program could have done earns little. It must work for any job, so nothing in it may assume one domain; the job is
described by `lab.contract`.

Its form:
- top-level constants (prompts are plain strings with {{placeholder}} fields) and helper functions;
- one entry point, `async def learn(lab)`, returning the decision program's source as a string;
- no imports, no names that start with an underscore, no bare `except`, and no str.format; fill a template
  with `lab.fill(TEMPLATE, name=value, ...)`.

What `lab` offers, every call metered (a mechanism over budget stops and scores nothing):
- `lab.episodes`: the judged runs, each with .task_id, .request, .steps, .success and .reason;
- `lab.render(episode)`: a run as text; `lab.contract`: the decision program's contract as text;
- `await lab.ask(prompt)`: the writer's text reply, from which the mechanism takes the program's source;
- `await lab.practice(source, task_id)`: try a program on a run's request, returning (reward, reason, observations);
- `lab.stream()`: the distinct task ids in order; `lab.request(task_id)`: a task's request;
- `lab.note(text)`: keep a note.

Writer calls are the cost: the budget is {writer_calls} per problem."""

MUTATION_PROMPT = """You are improving a program-learning mechanism by reflective mutation in a population-based search.

""" + PROGRAM_LAB + """

You will see the current mechanism, feedback from scoring the programs it wrote (requests served right, wrong actions,
hand-offs, and why they failed), and sometimes a sibling mechanism. Produce an improved mechanism that fixes what the
feedback shows and keeps what works.

Output only the mechanism's source."""

CROSSOVER_PROMPT = """You are combining two program-learning mechanisms into one offspring.

""" + PROGRAM_LAB + """

Take what each parent does well and drop what either does poorly.

Output only the mechanism's source."""

PROGRAM_RULES = """A decision program serves requests with no language model, and asks Jev, a small model that returns typed
judgments, for what code can't decide. Its contract:

{contract}

What you may change is how the program judges a request, in the body of `decide`: the questions it asks Jev, the state
it gives Jev, the thresholds it applies to the answers, and its hand-offs, including any hand-off the contract lets it
route. You may add constants and helpers.

Everything else is frozen, and a proposal that changes it is rejected: every existing constant and helper, every call to
`kit.find`, `kit.read`, `kit.procedure` or a helper, and every action the program returns. You may add no new action.

Decide what a request asks for with Jev, never with a list of words. A wrong action costs far more than a hand-off."""

EDIT_PROMPT = """You are improving a decision program by one small edit, in a search that keeps every version it tries.

""" + PROGRAM_RULES + """

You will see the program, how it scored, and the requests it got wrong with what Jev answered on each. Make the smallest
change that fixes those requests and keeps what works. Output only the program's source."""

COMBINE_PROMPT = """You are combining two versions of a decision program into one.

""" + PROGRAM_RULES + """

Take the judgments that work in each. Output only the program's source."""


@dataclass
class ProgramSearch:
    """A search for when no mechanism writes a program the gate admits.

    `method` is any of helix.search.METHODS. `archive` is a path to keep the search's history across searches: a
    later gap forks from anything earlier ones found instead of starting over. `target` is what it changes:
    "mechanism" rewrites the mechanism, whose every variant then writes a program from scratch. "program" edits the
    best program the mechanisms wrote, so a good change is inherited, and `frozen_change` keeps what already works.
    """

    method: str = "dgm"
    target: str = "mechanism"
    archive: str = ":memory:"
    population: int = 4
    generations: int = 2
    designer_model: str | None = None
    designer_params: dict[str, Any] = field(default_factory=lambda: {"reasoning_effort": "medium"})
    penalty: float = 2.0  # what one wrong action costs in the search's score, in requests served right
    seed: int = 7
    label: str | None = None
    max_searches: int = 1


@dataclass
class ProgramTier:
    """How one agent's decision tier learns and is judged."""

    desk: Desk
    decider: Decider
    contract: Contract
    labeled: dict[str, dict[str, Any]]  # the gate's labeled requests, by id
    mechanisms: Sequence[tuple[str, str]]  # (name, source), tried in order
    search: ProgramSearch | None = None
    search_cases: dict[str, dict[str, Any]] = field(default_factory=dict)  # what a search scores on; never the gate's
    min_runs: int = 12  # judged LLM runs before the gap can open
    min_gap_usd: float = 0.0005  # the LLM's mean cost per run that makes replacing it worthwhile
    handoff_window: int = 12
    handoff_limit: float = 0.5
    detector: Detector = field(default_factory=lambda: Detector(window=6, limit=2))
    bar: float = 0.9
    max_wrong: int = 0
    max_attempts: int = 4
    retry_after: int = 12
    writer_model: str | None = None
    writer_params: dict[str, Any] = field(default_factory=lambda: {"reasoning_effort": "medium"})
    budget: MechanismBudget = field(default_factory=lambda: MechanismBudget(writer_calls=8, practice_calls=1000,
                                                                            seconds=1800))
    prices: dict[str, tuple[float, ...]] = field(default_factory=lambda: dict(PRICES))
    gating_sources: Sequence[str] = ("verifier", "human", "harness")  # verdicts that can switch a class off
    routed_bar: float = 0.0  # the share of routed requests the LLM must get right, at the program's chosen route
    world: Any = None  # task -> the agent's context; with it, a program's route is tried on the real agent
    route_practice: Any = None  # async (task, effort, memory) -> (accepted, reason); overrides `world`


@dataclass
class Judgment:
    """A program on the gate's labeled requests."""

    accepted: int
    total: int
    wrong: int
    handoffs: int
    failures: list[str]
    routed: int = 0  # requests sent to the LLM with a route and a memory
    routed_right: int = 0
    answers: dict[str, Any] = field(default_factory=dict)  # what Jev said on each failure, by case id

    def score(self, penalty: float) -> float:
        if not self.total:
            return 0.0
        served = self.accepted - self.routed_right
        return max(0.0, (served + ROUTED_REWARD * self.routed_right - penalty * self.wrong) / self.total)

    def feedback(self) -> str:
        shown = "\n".join(f"- {failure}" for failure in self.failures[:12])
        return (f"{self.accepted}/{self.total} requests served right or rightly handed off, {self.wrong} wrong "
                f"actions, {self.handoffs} hand-offs of work it could have done, {self.routed} routed to the language "
                f"model ({self.routed_right} right).\n{shown}")


class DecisionLoop:
    """The decision tier of one Metacognition: serving, the per-class switch, the signal, and improvement."""

    def __init__(self, meta: Any, tier: ProgramTier) -> None:
        self.meta = meta
        self.tier = tier
        self.paused = False  # no improvement while paused: a study's held-out requests must not teach the loop
        self.search_log: list[dict[str, Any]] = []  # one row per design the search scored, and each it refused

    # ------------------------------------------------------------ what serves

    def current(self) -> dict[str, Any] | None:
        return self.meta.store.trusted(DECISION)

    def off(self, action: str) -> bool:
        return bool(self.meta.store.state(f"program:{action}").get("off"))

    def _kit(self, agent: Any, ctx: Any, loop: Any, trajectory: Trajectory, calls: list[dict[str, Any]]) -> Kit:
        def noted(name: str, arguments: dict[str, Any], result: Any) -> None:
            trajectory.append(StepKind.TOOL_CALL, {"name": name, "arguments": arguments})
            trajectory.append(StepKind.TOOL_RESULT, {"result": result})

        def read(tool: Any) -> Any:
            def call(**arguments: Any) -> Any:
                result = tool.fn(ctx, **arguments) if tool.takes_context else tool.fn(**arguments)
                if inspect.isawaitable(result):
                    getattr(result, "close", lambda: None)()
                    raise RuntimeError(f"{tool.name} is async; a decision program reads sync tools only")
                noted(tool.name, arguments, result)
                return result

            return call

        api = SkillAPI(self.meta._api_tools(), ctx, loop)

        def procedure(name: str, version: dict[str, Any]) -> Any:
            entry = SimpleNamespace(content={"skill_name": name, "description": self.meta.procedures[name].description,
                                             "body": version["source"], "body_kind": "python"},
                                    metadata={"created_by": version["created_by"], "version": version["version"]})
            tool = self.meta._procedure_tool(entry, api)

            async def call(**arguments: Any) -> Any:
                result = await tool.call(**arguments)
                noted(name, arguments, result)
                return result

            return call

        trusted = {name: v for name in self.meta.procedures if (v := self.meta.store.trusted(name)) is not None}
        kit = Kit(reads={t.name: read(t) for t in agent.tools if getattr(t, "read_only", False)},
                  procedures={name: procedure(name, v) for name, v in trusted.items()}, decider=self.tier.decider)
        ask = kit.ask

        async def metered(state: Any, questions: dict[str, Any]) -> dict[str, Any]:
            before = replace(kit.usage)
            answers = await ask(state, questions)
            calls.append({"dollars": kit.usage.dollars - before.dollars, "seconds": kit.usage.seconds - before.seconds,
                          "input_tokens": kit.usage.input_tokens - before.input_tokens, "hook": "decide"})
            return answers

        kit.ask = metered  # type: ignore[method-assign]
        return kit

    async def serve(self, plan: Any, agent: Any, ctx: Any, request: str, loop: Any) -> RunResult | None:
        """Run the trusted program. A RunResult if it acted; None to continue into the agent, in this same run."""
        version = plan.program
        trajectory = Trajectory(task=request)
        calls: list[dict[str, Any]] = []
        kit = self._kit(agent, ctx, loop, trajectory, calls)
        try:
            decision = await run_program(version["source"], request, kit)
        except MissingAnswer:
            raise
        except Exception as exc:  # noqa: BLE001 - a program's own failure hands the request to the agent
            decision = {"handoff": f"the program failed: {type(exc).__name__}: {exc}"}
        plan.decider_calls = calls
        action = decision.get("action")
        tool = next((t for t in agent.tools if t.name == action and not getattr(t, "read_only", False)), None)
        if "handoff" in decision or tool is None or self.off(str(action)):
            why = decision.get("handoff") or (f"{action} is switched off" if tool else f"no action {action!r}")
            route = decision.get("route")
            if route:
                plan.program_outcome = "routed"
                plan.effort, plan.report.effort, plan.memory = route, route, decision.get("memory") or None
                plan.report.events.append(f"program: routed to the LLM at {route} ({why})")
                return None
            plan.program_outcome = "handoff"
            plan.report.events.append(f"program: handed off ({why})")
            return None
        arguments = decision.get("arguments") or {}
        started = time.perf_counter()
        try:
            result = await tool.invoke(ctx, **arguments)
        except Exception as exc:  # noqa: BLE001 - arguments the action can't take: the agent tries instead
            plan.program_outcome = "handoff"
            plan.report.events.append(f"program: handed off (the action failed: {type(exc).__name__}: {exc})")
            return None
        calls.append({"dollars": 0.0, "seconds": time.perf_counter() - started, "input_tokens": 0, "hook": "act"})
        trajectory.append(StepKind.TOOL_CALL, {"name": str(action), "arguments": arguments})
        trajectory.append(StepKind.TOOL_RESULT, {"result": result})
        reply = str(decision.get("reply") or "Done.")
        trajectory.metadata.setdefault(META_KEY, {})["decider_calls"] = calls
        trajectory.complete(reply)
        plan.program_outcome, plan.program_action = "served", str(action)
        plan.effort = RUNG
        plan.report.effort = RUNG
        plan.offered = []
        label = [f"{DECISION}@{version['version']}"]
        plan.report.offered, plan.report.called = label, label
        return RunResult(input=request, new_items=[], raw_responses=[], final_output=reply, last_agent=agent,
                         context_wrapper=ctx, trajectory=trajectory, cost=run_cost(trajectory, self.tier.prices))

    def after_run(self, plan: Any, verdict: Any, events: list[str]) -> None:
        """Count how the request went, and let a verdict on a program's action reach that action's detector."""
        store = self.meta.store
        outcome = plan.program_outcome or "llm"
        state = store.state(_OUTCOMES)
        state["recent"] = [*state.get("recent", []), outcome][-100:]
        store.put_state(_OUTCOMES, state)
        if outcome != "served" or verdict is None or verdict.source not in self.tier.gating_sources:
            return
        key = f"program:{plan.program_action}"
        cell = store.state(key)
        window, quarantine = self.tier.detector.record(cell.get("window", []), plan.run_id, verdict.accepted,
                                                       source=verdict.source)
        cell["window"] = window
        if quarantine:
            cell.update(off=True, reason=verdict.reason, window=[])
            events.append(f"program: switched off {plan.program_action} ({verdict.reason})")
        store.put_state(key, cell)

    # ------------------------------------------------------------ the signal and the gap

    def signal(self) -> dict[str, Any]:
        """What the store says about this work: the LLM's cost, and how the current program is doing."""
        store, tier = self.meta.store, self.tier
        since = store.state(_LOOP).get("since", 0)
        runs = [r for r in store.runs(since) if r["verdict"] is not None]
        agent = [r for r in runs if r["effort"] != RUNG]
        cost = sum(r["dollars"] or 0.0 for r in agent) / len(agent) if agent else 0.0
        seconds = sum(r["seconds"] or 0.0 for r in agent) / len(agent) if agent else 0.0
        recent = store.state(_OUTCOMES).get("recent", [])[-tier.handoff_window:]
        current = self.current()
        off = [name.split(":", 1)[1] for name in self._classes() if store.state(name).get("off")]
        handoffs = recent.count("handoff") / len(recent) if recent else 0.0
        return {"llm_runs": len(agent), "llm_dollars": cost, "llm_seconds": seconds, "program": current["version"]
                if current else None, "off": off, "handoff_share": handoffs, "window": len(recent)}

    def _classes(self) -> list[str]:
        return [f"program:{t.name}" for t in (getattr(self.meta.agent, "tools", None) or [])
                if not getattr(t, "read_only", False)]

    def gap(self, signal: dict[str, Any]) -> str | None:
        """Why improvement is due now, or None."""
        tier = self.tier
        if not any(self.meta.store.trusted(name) for name in self.meta.procedures):
            return None  # learn the procedures first: a program stands on them
        if signal["off"]:
            return f"the program is switched off for {', '.join(signal['off'])}"
        if signal["program"] is None:
            if signal["llm_runs"] >= tier.min_runs and signal["llm_dollars"] >= tier.min_gap_usd:
                return (f"the LLM spends ${signal['llm_dollars']:.4f} and {signal['llm_seconds']:.1f} s a request over "
                        f"{signal['llm_runs']} runs, and no program serves this work")
            return None
        if signal["window"] >= tier.handoff_window and signal["handoff_share"] >= tier.handoff_limit:
            return f"the program handed off {signal['handoff_share']:.0%} of the last {signal['window']} requests"
        return None

    async def maybe_improve(self, events: list[str]) -> None:
        if self.paused:
            return
        store, tier = self.meta.store, self.tier
        state = store.state(_LOOP)
        count = store.count()
        if state.get("attempts", 0) >= tier.max_attempts or count < state.get("next_attempt", 0):
            return
        reason = self.gap(self.signal())
        if reason is None:
            return
        state.update(attempts=state.get("attempts", 0) + 1, next_attempt=count + tier.retry_after)
        store.put_state(_LOOP, state)
        events.append(f"gap: {reason}")
        await self.improve(events)

    # ------------------------------------------------------------ improvement

    async def practice(self, source: str, case_id: str) -> tuple[float, str, str]:
        reward, reason, _ = await self._practice(source, case_id)
        return reward, reason, ""

    async def _practice(self, source: str, case_id: str) -> tuple[float, str, dict[str, Any]]:
        task = self.tier.labeled.get(case_id) or self.tier.search_cases.get(case_id)
        if task is None:
            run = self.meta.store.case(case_id)
            if run is None or run["case_data"] is None:
                return 0.0, f"no recorded request for {case_id}", {}
            task = run["case_data"]
        try:
            row = await serve(source, task, self.tier.desk, self.tier.decider)
        except UnsafeProgram as exc:
            return 0.0, f"the guard refused the program: {exc}", {}
        if row.route and (self.tier.route_practice is not None or self.tier.world is not None):
            attempt = self.tier.route_practice or self._try_route
            accepted, reason = await attempt(task, row.route, row.memory)
            reward = ROUTED_REWARD if accepted else ROUTED_WRONG
            return reward, f"routed to the LLM at {row.route}: {reason}", row.answers
        if row.right:
            return 1.0, row.reason, row.answers
        if not row.handled:
            return HANDOFF_REWARD, f"it was handed off, though it could be served: {row.reason}", row.answers
        return 0.0, row.reason, row.answers

    async def _try_route(self, task: dict[str, Any], effort: str, memory: str | None) -> tuple[bool, str]:
        """Run the agent as a program's route would, frozen so the try teaches nothing, and judge the outcome."""
        from helix.metacognition.exceptions import AgentsException
        from helix.metacognition.parts import Verdict
        from helix.metacognition.run import RunConfig, Runner

        config = RunConfig(metacognition_mode="frozen", case_id=task["id"], case_data=task, effort=effort, memory=memory,
                           max_tool_calls=24)
        try:
            result = await Runner.run(self.meta.agent, task["text"], context=self.tier.world(task), max_turns=12,
                                      run_config=config)
        except AgentsException as exc:
            return False, f"the run failed: {exc}"
        verdict = self.meta.verifier(result.context_wrapper, result)
        verdict = Verdict.coerce(await verdict if inspect.isawaitable(verdict) else verdict)
        return (False, "no verdict") if verdict is None else (verdict.accepted, verdict.reason or "accepted")

    def _cases(self) -> list[str]:
        """The gate's labeled requests, and every request a program served wrongly since the last install."""
        store = self.meta.store
        since = store.state(_LOOP).get("since", 0)
        recent = [r["case_id"] for r in store.runs(since) if r["effort"] == RUNG and r["verdict"] == "rejected"]
        return [*self.tier.labeled, *[c for c in dict.fromkeys(recent) if c not in self.tier.labeled]]

    async def judge(self, source: str, cases: Sequence[str] | None = None) -> Judgment:
        accepted, wrong, handoffs, routed, routed_right, failures, answers = 0, 0, 0, 0, 0, [], {}
        cases = list(self._cases() if cases is None else cases)
        for case_id in cases:
            reward, reason, said = await self._practice(source, case_id)
            accepted += int(reward >= 1.0 or reward == ROUTED_REWARD)
            wrong += int(reward == 0.0)
            handoffs += int(reward == HANDOFF_REWARD)
            routed += int(reward in (ROUTED_REWARD, ROUTED_WRONG))
            routed_right += int(reward == ROUTED_REWARD)
            if reward < 1.0:
                failures.append(f"{case_id}: {reason}")
                answers[case_id] = said
        return Judgment(accepted, len(cases), wrong, handoffs, failures, routed, routed_right, answers)

    def _gate(self, judged: Judgment) -> GateResult:
        # Each route answers for itself: what the program serves must clear the bar with next to no wrong actions,
        # and what it sends to the LLM must do as well as the tier asks of the LLM.
        own = judged.accepted - judged.routed_right
        own_total = judged.total - judged.routed
        passed = (own_total > 0 and own / own_total >= self.tier.bar and judged.wrong <= self.tier.max_wrong
                  and (judged.routed == 0 or judged.routed_right / judged.routed >= self.tier.routed_bar))
        return GateResult(judged.accepted, judged.total, passed, judged.failures)

    def _problem(self) -> LearningProblem:
        episodes: list[Episode] = self.meta.store.episodes()[-80:]
        return LearningProblem(procedure=DECISION, contract=self.tier.contract, episodes=episodes,
                               practice=self.practice, cases=list(self.tier.labeled), score=self.practice)

    def _mechanisms(self) -> list[tuple[str, str]]:
        found = self.meta.store.setting("program_mechanisms") or []
        return [(m["name"], m["source"]) for m in found] + list(self.tier.mechanisms)

    async def _learn(self, problem: LearningProblem, name: str, source: str) -> tuple[Any, Judgment | None]:
        learner = SourceLearner(source, name, GENERIC, self.tier.budget)
        choice = await run_tournament(problem, [learner], writer_model=self.tier.writer_model or DEFAULT_MODEL,
                                      writer_params=self.tier.writer_params, prices=self.tier.prices)
        trial = choice.winner
        return trial, await self.judge(trial.source) if trial.source else None

    def _install(self, program: str, mechanism: str, problem: LearningProblem, trial: Any, gate: GateResult,
                 events: list[str]) -> None:
        store = self.meta.store
        current = self.current()
        if current is not None:
            store.set_state(DECISION, current["version"], "retired")
        version = store.add_version(DECISION, program, state="trusted", created_by=mechanism)
        for key in self._classes():
            store.put_state(key, {})
        state = store.state(_LOOP)
        state["since"] = store.count()
        store.put_state(_LOOP, state)
        store.put_state(_OUTCOMES, {})
        store.log_learning(name=DECISION, mechanism=mechanism, version=version, episodes=len(problem.episodes),
                           writer_calls=trial.writer_calls, dollars=trial.dollars, gate=gate.to_dict(),
                           outcome="passed, installed")
        events.append(f"program: {mechanism} wrote {DECISION}@{version}; gate {gate.accepted}/{gate.total} "
                      "passed, installed")

    async def _retest(self, events: list[str]) -> bool:
        """Switch classes back on if the current program now passes: what broke may not have been the program."""
        store, current = self.meta.store, self.current()
        off = [key for key in self._classes() if store.state(key).get("off")]
        if current is None or not off:
            return False
        judged = await self.judge(current["source"])
        gate = self._gate(judged)
        label = f"{DECISION}@{current['version']}"
        if not gate.passed:
            events.append(f"program: re-tested {label}; still refused, {judged.accepted}/{judged.total}, "
                          f"{judged.wrong} wrong actions")
            return False
        for key in off:
            store.put_state(key, {})
        state = store.state(_LOOP)
        state["since"] = store.count()
        store.put_state(_LOOP, state)
        store.put_state(_OUTCOMES, {})
        store.log_learning(name=DECISION, mechanism="retest", version=current["version"], episodes=0, writer_calls=0,
                           dollars=0.0, gate=gate.to_dict(), outcome="passed, switched back on")
        events.append(f"program: re-tested {label}; gate {gate.accepted}/{gate.total} passed, switched back on: "
                      f"{', '.join(key.split(':', 1)[1] for key in off)}")
        return True

    async def improve(self, events: list[str]) -> None:
        store, tier = self.meta.store, self.tier
        if await self._retest(events):
            return
        problem = self._problem()
        written: list[tuple[str, Any]] = []
        for name, source in self._mechanisms():
            trial, judged = await self._learn(problem, name, source)
            if judged is None:
                store.log_learning(name=DECISION, mechanism=name, version=None, episodes=len(problem.episodes),
                                   writer_calls=trial.writer_calls, dollars=trial.dollars, gate=None,
                                   outcome=f"failed: {trial.note}")
                events.append(f"program: {name} wrote nothing ({trial.note})")
                continue
            gate = self._gate(judged)
            if gate.passed:
                self._install(trial.source, name, problem, trial, gate, events)
                return
            written.append((name, trial))
            store.log_learning(name=DECISION, mechanism=name, version=None, episodes=len(problem.episodes),
                               writer_calls=trial.writer_calls, dollars=trial.dollars, gate=gate.to_dict(),
                               outcome="refused")
            events.append(f"program: {name} wrote a program the gate refused: {judged.accepted}/{judged.total}, "
                          f"{judged.wrong} wrong actions")
        state = store.state(_LOOP)
        if tier.search is None or state.get("searches", 0) >= tier.search.max_searches:
            return
        state["searches"] = state.get("searches", 0) + 1
        store.put_state(_LOOP, state)
        await self._search(problem, events, written)

    async def _best_written(self, written: Sequence[tuple[str, Any]]) -> tuple[str, str] | None:
        """The written program that does best on the search's own cases, so the gate's stay hidden."""
        cases = list(self.tier.search_cases) or None
        scored = [((await self.judge(trial.source, cases)).score(self.tier.search.penalty), i, name, trial.source)
                  for i, (name, trial) in enumerate(written)]
        if not scored:
            return None
        _, _, name, source = max(scored, key=lambda row: (row[0], -row[1]))
        return name, source

    def _edit_feedback(self, seen: Judgment) -> str:
        """The score, and each failure as the request itself and what Jev answered: what an edit needs to see."""
        shown = []
        for failure in seen.failures[:14]:
            case_id, reason = failure.split(": ", 1)
            request = (self.tier.search_cases.get(case_id) or self.tier.labeled.get(case_id) or {}).get("text", case_id)
            shown.append(f"- Request: {request}\n  Outcome: {reason}\n  Jev answered: "
                         f"{json.dumps(seen.answers.get(case_id, {}))}")
        return (f"{seen.accepted}/{seen.total} requests served right or rightly handed off, {seen.wrong} wrong "
                f"actions, {seen.handoffs} hand-offs of work it could have done, {seen.routed} routed to the language model "
                f"({seen.routed_right} right).\n" + "\n".join(shown))

    async def _search(self, problem: LearningProblem, events: list[str],
                      written: Sequence[tuple[str, Any]] = ()) -> None:
        store, tier, spec = self.meta.store, self.tier, self.tier.search
        editing = spec.target == "program"
        if editing:
            chosen = await self._best_written(written)
            if chosen is None:
                events.append("search: no mechanism wrote a program to edit")
                return
            name, seed_source = chosen
        else:
            name, seed_source = self._mechanisms()[0]
        best: dict[str, Any] = {}

        def logged(source: str, outcome: str, **fields: Any) -> None:
            self.search_log.append({"design": source_id(source), "outcome": outcome, "source": source, **fields})
            detail = "; ".join(f"{k}={v}" for k, v in fields.items() if k != "program")
            events.append(f"search: design {source_id(source)} {outcome} {detail}")

        async def evaluate(candidate: Any, questions: Any = None) -> GapMeasurement:
            source = candidate.content if hasattr(candidate, "content") else candidate
            if editing:
                trial = SimpleNamespace(source=source, writer_calls=0, dollars=0.0, note="")
                judged = await self.judge(source)
            else:
                try:
                    trial, judged = await self._learn(problem, f"design_{source_id(source)}", source)
                except UnsafeMechanism as exc:
                    logged(source, "refused by guard", reason=str(exc))
                    return GapMeasurement(score=0.0, signal_id="program_gate", signal_version=1, triggered=True,
                                          raw_value=0.0, feedback=f"refused by the mechanism guard: {exc}", metadata={})
            if judged is None:
                logged(source, "wrote nothing", note=trial.note, writer_calls=trial.writer_calls)
                return GapMeasurement(score=0.0, signal_id="program_gate", signal_version=1, triggered=True,
                                      raw_value=0.0, feedback=f"it wrote no program: {trial.note}", metadata={})
            # The proposer sees only the search's own cases. The gate's verdict decides admission and stays hidden.
            seen = await self.judge(trial.source, list(tier.search_cases)) if tier.search_cases else judged
            score = seen.score(spec.penalty)
            passed = self._gate(judged).passed
            logged(source, "admitted" if passed else "refused by gate", right=f"{judged.accepted}/{judged.total}",
                   wrong=judged.wrong, handoffs=judged.handoffs, routed=f"{judged.routed_right}/{judged.routed}",
                   seen=f"{seen.accepted}/{seen.total}", seen_wrong=seen.wrong, search_score=round(score, 2),
                   writer_calls=trial.writer_calls, program=trial.source)
            if passed and (not best or score > best["score"]):
                best.update(score=score, source=source, trial=trial, judged=judged)
            behavior = sorted(failure.split(":", 1)[0] for failure in seen.failures)
            return GapMeasurement(score=score, signal_id="program_gate", signal_version=1, triggered=not passed,
                                  raw_value=score,
                                  feedback=self._edit_feedback(seen) if editing else seen.feedback(),
                                  metadata={"judged": asdict(seen), "behavior": behavior})

        def validate(source: str) -> str | None:
            try:
                check_program(source) if editing else check_mechanism(source)
            except (UnsafeMechanism, UnsafeProgram) as exc:
                return str(exc)
            return frozen_change(seed_source, source) if editing else None

        calls = tier.budget.writer_calls if tier.budget.writer_calls is not None else "unlimited"
        run = f"\n\nThis is design search {spec.label}." if spec.label else ""
        rules = contract_text(tier.contract, GENERIC)
        mutation = (EDIT_PROMPT.replace("{contract}", rules) if editing
                    else MUTATION_PROMPT.replace("{writer_calls}", str(calls)))
        crossover = (COMBINE_PROMPT.replace("{contract}", rules) if editing
                     else CROSSOVER_PROMPT.replace("{writer_calls}", str(calls)))
        search = build_search(spec.method, proposer_model=spec.designer_model or tier.writer_model or DEFAULT_MODEL,
                              evaluator=evaluate, mutation_prompt=mutation + run, crossover_prompt=crossover + run,
                              validate=validate, proposer_params=dict(spec.designer_params),
                              population=spec.population, generations=spec.generations, rng=random.Random(spec.seed))
        archive = SQLiteArchive(spec.archive)
        seed = genesis(DECISION if editing else "program_mechanism", ArtifactKind.CODE, seed_source, created_by=name)
        limit = SearchBudget(max_candidates=spec.population * spec.generations)
        async for _ in search.propose(seed, None, archive, limit):
            pass
        archive.close()
        for rejected in search.discarded:
            self.search_log.append({"design": None, "outcome": "proposal failed validation", **rejected})
            events.append(f"search: a proposal failed validation: {rejected['reason']}")
        if not best:
            events.append("search: no design wrote a program the gate admits")
            return
        if editing:
            design = f"edited_{source_id(best['source'])}"
            events.append(f"search: {design} wrote a program the gate admits")
            self._install(best["source"], design, problem, best["trial"], self._gate(best["judged"]), events)
            return
        design = f"designed_{source_id(best['source'])}"
        found = [m for m in (store.setting("program_mechanisms") or []) if m["name"] != design]
        store.put_setting("program_mechanisms", [{"name": design, "source": best["source"]}, *found])
        events.append(f"search: {design} wrote a program the gate admits")
        self._install(best["trial"].source, design, problem, best["trial"], self._gate(best["judged"]), events)


__all__ = ["DECISION", "RUNG", "DecisionLoop", "ProgramSearch", "ProgramTier", "Judgment"]
