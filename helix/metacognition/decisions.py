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
             named them, and with no way to act if the program said it should not. Each action is a task class with
             its own detector window, so a class that starts failing is switched off alone and goes back to the LLM
             until a program passes.

Programs are kept as versions of `DECISION` in the store, beside the procedures. Runs a program served are
recorded with the reasoning setting "program". Jev's calls are charged as the run's decider calls.
"""

from __future__ import annotations

import asyncio
import inspect
import random
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field, replace
from types import SimpleNamespace
from typing import Any

from helix.archive import SQLiteArchive
from helix.artifact import ArtifactKind, genesis
from helix.memory.learning_mechanisms import Contract, Episode
from helix.memory.mechanism_runtime import (MECHANISM_BUILTINS, MechanismBudget, UnsafeMechanism, check_mechanism,
                                             check_runnable, pack_memories, unpack_memories)
from helix.scaffold.common import META_KEY
from helix.search.base import SearchBudget
from helix.search.methods import build_search
from helix.signal import Cost, GapMeasurement, SignalKind
from helix.trajectory import StepKind, Trajectory

from helix.deciders.base import Decider, MissingAnswer
from helix.metacognition.decide import RUNG, Desk, Kit, UnsafeProgram, check_program, frozen_change, run_program, serve
from helix.metacognition.events import emit
from helix.metacognition.exceptions import Unavailable
from helix.metacognition.api import SkillAPI
from helix.metacognition.gap import Attempt, Good, gap_of, report
from helix.metacognition.ladders import most_demanding
from helix.metacognition.mechanisms import contract_text
from helix.metacognition.parts import Detector, GateResult
from helix.metacognition.prompts import GENERIC
from helix.metacognition.selection import LearningProblem, run_tournament
from helix.metacognition.sources import SourceLearner, source_id
from helix.metacognition.model_settings import BUILDER_MODEL, DEFAULT_MODEL
from helix.metacognition.pricing import PRICES, run_cost
from helix.metacognition.result import RunResult
from helix.metacognition.run_context import RunContextWrapper

DECISION = "decision_program"
MEMORY = "memory:"  # a stored memory's versions are kept under this prefix, beside the program that retrieves them
HANDOFF_REWARD = 0.25  # better than a wrong action, far from serving the request
ROUTED_REWARD = 0.6  # the LLM got it right where the program sent it: served, but not for free
ROUTED_WRONG = 0.1  # the program's choice of route failed; the LLM's own mistake is not a program's wrong action
READ_ONLY_LOST = 0.15  # real work sent where nothing can do it: no wrong action, but the request is lost
_LOOP = "decision:loop"
_OUTCOMES = "decision:outcomes"

PROGRAM_LAB = """A learning component is Python source. It decides what is learned from experience, how that is stored as
memories, and how each request is served from them. It is given runs a language model already did, each judged by a
checker.

A memory has a name, a description, a body and a route, and the route says how a request that retrieves it is served:
- "program": the body is a decision program, `async def decide(request, kit)`, that serves the request with no language
  model. It calls tools and procedures and asks Jev, a small model that returns typed judgments, for what code can't
  decide, and it returns an action or a hand-off as the contract says.
- "none", "low", "medium" or "high": the language model serves the request at that reasoning, reading the body first.
  With read_only true it can read and answer but has no way to act.

The component's entry point, `async def learn(lab)`, stores memories with
`lab.remember(name, description, body, route=..., read_only=...)` and returns the source of the program that picks a
memory for each request. That program is also `async def decide(request, kit)`. It can call `await kit.recall(text, k)`
for the closest memories (each a dict with "name", "description", "route" and "read_only") and can ask Jev. It returns
{"serve": name}, or acts or hands off itself as the contract says.

Whatever a program decides about what a request asks for, it decides by asking Jev, never by looking for words in the
request: a list of words fits only the wordings it was built from, and the requests that score a component are worded
differently. Code is for what code can know exactly, such as finding an id or calling a tool.

The component is scored on held-out requests: an action must be right, a request outside the job must be handed off
read-only, a wrong action is penalized heavily, a request served by the language model earns less than one served
without it, and a plain hand-off of work the program could have done earns little. {admission} It must work for any job, so nothing in it may assume one domain; the job
is described by `lab.contract`.

Its form:
- top-level constants (prompts are plain strings with {placeholder} fields) and helper functions;
- one entry point, `async def learn(lab)`;
- no imports, no names that start with an underscore, no bare `except`, and no str.format; fill a template
  with `lab.fill(TEMPLATE, name=value, ...)`;
- the only builtins, here and in the programs it writes, are """ + ", ".join(sorted(MECHANISM_BUILTINS)) + """.

What `lab` offers, every call metered (a component over budget stops and scores nothing):
- `lab.episodes`: the judged runs, each with .task_id, .request, .steps (each a dict with "tool", "arguments" and
  "result"), .success and .reason;
- `lab.memories`: what this component's parent stored, as dicts whose keys are remember's arguments, so
  `lab.remember(**memory)` keeps one. It is empty on a first run, and nothing is kept unless it is remembered again;
- `lab.render(episode)`: a run as text; `lab.contract`: a decision program's contract as text;
- `await lab.ask(prompt)`: the writer's text reply; `lab.parse(reply)`: the JSON value in a reply;
- `await lab.practice(source, task_id)`: try a program, with what is remembered so far, on a run's request, returning
  (reward, reason, the name of the memory that served it, if one did). The reward is 1 if a program did the work or a
  request that should not be acted on went read-only, """ + str(ROUTED_REWARD) + """ if the language model got it right, """ + str(HANDOFF_REWARD) + """ or less if it
  wasn't done, and 0 for a wrong action;
- `lab.stream()`: the distinct task ids in order; `lab.request(task_id)`: a task's request;
- `lab.tests`: the labeled test requests the search shares with this component, each a dict with "id", "request" and
  "should_act", empty if it shares none. `lab.practice` and `lab.request` take a test's id, so `learn` can try a change on
  the very requests the gap report names before deciding to keep it;
- `lab.note(text)`: keep a note.

Writer calls are the cost: the budget is {writer_calls} per problem."""

_NEEDS_A_PROGRAM = "A component that serves nothing\nwithout the language model can't be admitted."
_OR_COSTS_LESS = ("A component that serves everything with the language model is admitted only if it gets as many test\n"
                  "requests right as what is running now and costs less.")

MUTATION_PROMPT = """You are improving a learning component by rewriting its code, in a search that keeps every variant.

""" + PROGRAM_LAB + """

You will see the current component and its gap: what good is, how far the component is from it, where that gap sits,
what happened on every request it was tried on, what it stored and the program it returned. You may also see a sibling
component. Build one variant that closes the largest part of the gap and keeps what works.

Fix wrong actions and requests left undone first. If the language model gets a memory's requests wrong, raise that
memory's reasoning level before anything else. Use the lowest level that stays right: `learn` can try each level on the
shared requests with `lab.practice` and store the lowest that works, and a memory that is right at its level may be
right at a lower one.

After that, if requests are right only because the language model served them, that is the gap to close. Choose one
memory that acts and is still served by the language model, and change the code so that memory is served by a decision
program: `learn` writes the program and stores it with route "program", and keeps the other memories. Move one memory at
a time.

A decision program has to work on requests you can't see, which may be worded differently or ask for things in forms you
haven't seen. If the report says a memory's program made wrong actions on those requests, that program doesn't
generalize: put the memory back on the language model at a reasoning level that gets it right, and keep the programs that
work. Some work needs the language model, and a memory that does should stay on it.

Output only the component's source."""

CROSSOVER_PROMPT = """You are combining two learning components into one offspring.

""" + PROGRAM_LAB + """

Take what each parent does well and drop what either does poorly.

Output only the component's source."""

PROGRAM_RULES = """A decision program serves requests with no language model, and asks Jev, a small model that returns typed
judgments, for what code can't decide. Its contract:

{contract}

What you may change is how the program judges a request, in the body of `decide`: the questions it asks Jev, the state
it gives Jev, the thresholds it applies to the answers, and its hand-offs, including any hand-off the contract lets it
route. You may add constants and helpers. In its MEMORIES list, you may change a memory's description, route and
read_only.

Everything else is frozen, and a proposal that changes it is rejected: every existing constant and helper, every call to
`kit.find`, `kit.read`, `kit.procedure` or a helper, and every action the program returns. You may add no new action.
A decision memory's body follows the same rules, and no memory may be removed.

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
    """A search for when no mechanism writes a program the gate admits: a model rewrites code, and each variant is kept.

    `method` is any of helix.search.METHODS. `archive` is a path to keep the search's history across searches: a
    later gap forks from anything earlier ones found instead of starting over. `target` is the code it changes:
    "mechanism" rewrites the whole learning component, and each variant starts from the memories its parent stored.
    "program" edits the best program the components wrote, and `frozen_change` keeps what already works.
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
    share_all: bool = False  # hand the rewriter every test run, the gate's too, try every variant on the gate, and
    # let a variant's `learn` practice on the gate's requests as well as the search's
    stop_when_admitted: bool = False  # end the search at the first variant the gate admits
    lower_reasoning: bool = False  # also admit a variant that stays on the LLM, if it gets as many test requests right
    # as what's running now does, on the same requests, and costs less


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
    long_running: bool = False  # what a deployment that serves for months needs, and a short study does not.
    # Attempts and searches are counted for each gap and start again when it closes, since counted over the tier's
    # whole life they would stop improvement for good after `max_attempts` gaps. And a request a program served
    # wrongly stops being a gate case once a procedure has been relearned, since its world is gone. And when a
    # program fails its re-test, or one of its actions is switched off, each procedure it stands on is put through
    # its own gate: one that fails is quarantined at once, so it stops serving the agent too, and no builder is
    # asked to rewrite the program until it has been relearned
    retry_after: int = 12
    writer_model: str | None = None
    writer_params: dict[str, Any] = field(default_factory=lambda: {"reasoning_effort": "medium"})
    budget: MechanismBudget = field(default_factory=lambda: MechanismBudget(writer_calls=8, practice_calls=1000,
                                                                            seconds=1800))
    prices: dict[str, tuple[float, ...]] = field(default_factory=lambda: dict(PRICES))
    gating_sources: Sequence[str] = ("verifier", "human", "harness")  # verdicts that can switch a class off
    routed_bar: float = 0.0  # the share of routed requests the LLM must get right, at the program's chosen route
    read_only_handoffs: bool = False  # a request that should not be acted on is right only if handed off read-only
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
    acted: int = 0  # requests the program served right with no language model
    attempts: list[Attempt] = field(default_factory=list)  # every request tried, and what happened

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


class ServiceGap:
    """The gap between a decision program and what good is, as a helix Signal.

    A program is tried on the search's cases and the gate's. Its gap is the mean distance from good over every request
    it was tried on, and the report of that gap, with what happened on each request, goes with the candidate to the model
    that builds the next variant. By default the gate's requests stay hidden and a variant reaches them only if it clears
    the bar on its own, and the report gives only counts for them. `share_all` hands over every test run instead, which
    leaves the held-out stream as the only requests no part of the search has seen.
    """

    kind = SignalKind.GROUND_TRUTH
    signal_id = "service_gap"
    signal_version = 1
    cost_estimate = Cost()

    def __init__(self, loop: DecisionLoop, good: Good, *, share_all: bool = False) -> None:
        self.loop = loop
        self.good = good
        self.share_all = share_all
        self.running_now: dict[str, tuple[bool, float]] | None = None  # each test request: right, and what it cost

    async def tried(self, program: str) -> tuple[Judgment | None, Judgment | None]:
        """The program on the search's cases, and on the gate's if everything is shared or it earned that."""
        loop = self.loop
        seen = await loop.judge(program, list(loop.tier.search_cases)) if loop.tier.search_cases else None
        reached = seen is None or self.share_all or self.passes(seen)
        return seen, await loop.judge(program) if reached else None

    def keeps_up(self, judged: Judgment) -> bool:
        """Whether a variant that stays on the LLM gets as many right as what's running now, and costs less."""
        if self.running_now is None or judged.acted or judged.wrong > self.loop.tier.max_wrong:
            return False
        known = [attempt for attempt in judged.attempts if attempt.case_id in self.running_now]
        right = sum(attempt.reward >= 1.0 or attempt.reward == ROUTED_REWARD for attempt in known)
        ran = [attempt for attempt in known if attempt.outcome in ("llm", "llm_wrong")]
        return (right >= sum(self.running_now[attempt.case_id][0] for attempt in known)
                and sum(attempt.dollars for attempt in ran) < sum(self.running_now[attempt.case_id][1] for attempt in ran))

    def passes(self, judged: Judgment) -> bool:
        return self.loop._gate(judged).passed or self.keeps_up(judged)

    def admits(self, seen: Judgment | None, judged: Judgment | None) -> bool:
        if judged is None or not self.passes(judged):
            return False
        return seen is None or not self.share_all or self.passes(seen)

    @staticmethod
    def _held_back(judged: Judgment) -> str:
        wrong = Counter(attempt.memory for attempt in judged.attempts if attempt.outcome == "wrong" and attempt.memory)
        named = ", ".join(f"{count} from {memory}" for memory, count in wrong.most_common())
        return (f"\n\nOn {judged.total} other requests, which you can't see and which are worded differently from those "
                f"above, it got {judged.accepted} right, with {judged.wrong} wrong actions and {judged.handoffs} hand-offs "
                "of work it could have done." + (f" The wrong actions came from: {named}." if named else ""))

    def measurement(self, seen: Judgment | None, judged: Judgment | None, *, passed: bool, notes: str = "") -> GapMeasurement:
        looked_at = seen.attempts if seen is not None else []
        everything = [*looked_at, *(judged.attempts if judged is not None else [])]
        shown = everything if self.share_all or seen is None else looked_at
        gap = gap_of(everything, self.good)
        feedback = report(shown, self.good)
        if seen is not None and judged is not None and not self.share_all:
            feedback += self._held_back(judged)
        behavior = sorted(a.case_id for a in shown if self.good.gap(a.outcome) > 0)
        return GapMeasurement(score=max(0.0, 1.0 - gap), signal_id=self.signal_id, signal_version=self.signal_version,
                              triggered=not passed, raw_value=gap, feedback=feedback + notes,
                              metadata={"attempts": [asdict(a) for a in shown], "behavior": behavior})

    async def measure(self, candidate: Any, trajectory: Any = None, reference: Any = None,
                      ground_truth: Any = None) -> GapMeasurement:
        program = candidate.content if hasattr(candidate, "content") else candidate
        seen, judged = await self.tried(program)
        return self.measurement(seen, judged, passed=self.admits(seen, judged))


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

    def _trusted_tools(self, ctx: Any, loop: Any) -> dict[str, Any]:
        """The procedures the agent has at this moment, as tools on this context."""
        api = SkillAPI(self.meta._api_tools(), ctx, loop)
        tools = {}
        for name, spec in self.meta.procedures.items():
            version = self.meta.store.trusted(name)
            if version is not None:
                entry = SimpleNamespace(content={"skill_name": name, "description": spec.description,
                                                 "body": version["source"], "body_kind": "python"},
                                        metadata={"created_by": version["created_by"], "version": version["version"]})
                tools[name] = self.meta._procedure_tool(entry, api)
        return tools

    def _kit(self, agent: Any, ctx: Any, loop: Any, trajectory: Trajectory, calls: list[dict[str, Any]]) -> Kit:
        def noted(name: str, arguments: dict[str, Any], result: Any) -> None:
            trajectory.append(StepKind.TOOL_CALL, {"name": name, "arguments": arguments})
            trajectory.append(StepKind.TOOL_RESULT, {"result": result})

        def read(tool: Any) -> Any:
            def call(**arguments: Any) -> Any:
                try:
                    result = tool.fn(ctx, **arguments) if tool.takes_context else tool.fn(**arguments)
                except Unavailable:
                    ctx.unavailable = True  # the program hands off, and the fault is not its own
                    raise
                if inspect.isawaitable(result):
                    getattr(result, "close", lambda: None)()
                    raise RuntimeError(f"{tool.name} is async; a decision program reads sync tools only")
                noted(tool.name, arguments, result)
                return result

            return call

        def procedure(name: str, tool: Any) -> Any:
            async def call(**arguments: Any) -> Any:
                result = await tool.call(**arguments)
                noted(name, arguments, result)
                return result

            return call

        kit = Kit(reads={t.name: read(t) for t in agent.tools if getattr(t, "read_only", False)},
                  procedures={name: procedure(name, tool) for name, tool in self._trusted_tools(ctx, loop).items()},
                  decider=self.tier.decider)
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
            decision = await run_program(self._packed(version), request, kit)
        except MissingAnswer:
            raise
        except Exception as exc:  # noqa: BLE001 - a program's own failure hands the request to the agent
            decision = {"handoff": f"the program failed: {type(exc).__name__}: {exc}"}
        plan.decider_calls = calls
        if kit.served_by:
            plan.report.memory = kit.served_by
            plan.report.events.append(f"program: used memory {kit.served_by}")
        action = decision.get("action")
        tool = next((t for t in agent.tools if t.name == action and not getattr(t, "read_only", False)), None)
        if "handoff" in decision or tool is None or self.off(str(action)):
            why = decision.get("handoff") or (f"{action} is switched off" if tool else f"no action {action!r}")
            route = decision.get("route")
            if route:
                plan.effort, plan.report.effort, plan.memory = route, route, decision.get("memory") or None
            if decision.get("read_only"):
                plan.program_outcome, plan.read_only = "read_only", True
                at = f" at {route}" if route else ""
                plan.report.events.append(f"program: handed off read-only{at} ({why})")
                return None
            if route:
                plan.program_outcome = "routed"
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

    def after_run(self, plan: Any, verdict: Any, events: list[str]) -> bool:
        """Count how the request went, and let a verdict on a program's action reach that action's detector.
        True when this verdict switched the action off."""
        store = self.meta.store
        outcome = plan.program_outcome or "llm"
        state = store.state(_OUTCOMES)
        state["recent"] = [*state.get("recent", []), outcome][-100:]
        store.put_state(_OUTCOMES, state)
        if outcome != "served" or verdict is None or verdict.source not in self.tier.gating_sources:
            return False
        key = f"program:{plan.program_action}"
        cell = store.state(key)
        window, quarantine = self.tier.detector.record(cell.get("window", []), plan.run_id, verdict.accepted,
                                                       source=verdict.source)
        cell["window"] = window
        if quarantine:
            cell.update(off=True, reason=verdict.reason, window=[])
            events.append(f"program: switched off {plan.program_action} ({verdict.reason})")
        store.put_state(key, cell)
        return bool(quarantine)

    # ------------------------------------------------------------ the signal and the gap

    def signal(self) -> dict[str, Any]:
        """What the store says about this work: the LLM's cost, and how the current program is doing."""
        store, tier = self.meta.store, self.tier
        since = store.state(_LOOP).get("since", 0)
        agent = [(dollars, seconds) for effort, dollars, seconds in store.judged(since) if effort != RUNG]
        cost = sum(dollars for dollars, _ in agent) / len(agent) if agent else 0.0
        seconds = sum(seconds for _, seconds in agent) / len(agent) if agent else 0.0
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
        used_up = state.get("attempts", 0) >= tier.max_attempts
        if count < state.get("next_attempt", 0) or (used_up and not tier.long_running):
            return
        if tier.long_running and self._repairing():
            return  # a procedure the program stands on is being relearned, and that repair comes first
        reason = self.gap(self.signal())
        if reason is None:
            if tier.long_running and (state.get("attempts") or state.get("searches")):
                self._gap_closed(state)  # it closed by itself, so the next gap starts with its attempts whole
                store.put_state(_LOOP, state)
            return
        if used_up:
            # Learning has used its attempts on this gap. Testing the current program again asks no builder, so
            # an action that was switched off can still come back once what broke it has been fixed.
            state["next_attempt"] = count + tier.retry_after
            store.put_state(_LOOP, state)
            await self._retest(events)
            return
        state.update(attempts=state.get("attempts", 0) + 1, next_attempt=count + tier.retry_after)
        store.put_state(_LOOP, state)
        events.append(f"gap: {reason}")
        await self.improve(events)

    # ------------------------------------------------------------ improvement

    async def practice(self, source: str, case_id: str) -> tuple[float, str, str]:
        attempt = await self._attempt(source, case_id)
        return attempt.reward, attempt.reason, attempt.memory or ""

    def _practice_desk(self) -> Desk:
        """The tier's desk, calling the procedures the agent has now, so the gate passes only what would work live."""
        loop = asyncio.get_running_loop()

        def procedures(world: Any) -> dict[str, Any]:
            tools = self._trusted_tools(RunContextWrapper(context=world), loop)
            return {name: tool.call for name, tool in tools.items()}

        return replace(self.tier.desk, procedures=procedures)

    async def _attempt(self, source: str, case_id: str) -> Attempt:
        """Try a program on one request, and say what happened."""
        task = self.tier.labeled.get(case_id) or self.tier.search_cases.get(case_id)
        if task is None:
            run = self.meta.store.case(case_id)
            if run is None or run["case_data"] is None:
                return Attempt(case_id, case_id, True, "wrong", 0.0, f"no recorded request for {case_id}")
            task = run["case_data"]
        expected = task.get("expect") == "handoff"
        try:
            row = await serve(source, task, self._practice_desk(), self.tier.decider)
        except UnsafeProgram as exc:
            return Attempt(case_id, task["text"], not expected, "refused", 0.0, f"the guard refused the program: {exc}")

        def ended(outcome: str, reward: float, reason: str, dollars: float = 0.0, seconds: float = 0.0) -> Attempt:
            return Attempt(case_id, task["text"], not expected, outcome, reward, reason, row.served_by, row.answers,
                           row.dollars + dollars, row.seconds + seconds)

        if row.read_only:
            # Known without running the agent: it has no way to act, so nothing can go wrong here or get done.
            if expected:
                return ended("read_only", 1.0, "handed off read-only")
            return ended("lost", READ_ONLY_LOST, f"it was handed off read-only, so nothing could serve it: {row.reason}")
        if expected and not row.handled and self.tier.read_only_handoffs:
            return ended("handoff", HANDOFF_REWARD, ("it was handed off with the tools to act; a request that should not "
                                                     "be acted on must be handed off read-only"))
        if row.route and (self.tier.route_practice is not None or self.tier.world is not None):
            attempt = self.tier.route_practice or self._try_route
            accepted, reason, *spent = await attempt(task, row.route, row.memory)
            reward, outcome = (ROUTED_REWARD, "llm") if accepted else (ROUTED_WRONG, "llm_wrong")
            return ended(outcome, reward, f"routed to the LLM at {row.route}: {reason}", *spent)
        if row.right:
            return ended("program" if row.handled else "read_only", 1.0, row.reason)
        if not row.handled:
            return ended("handoff", HANDOFF_REWARD, f"it was handed off, though it could be served: {row.reason}")
        return ended("wrong", 0.0, row.reason)

    async def _try_route(self, task: dict[str, Any], effort: str, memory: str | None) -> tuple[bool, str, float, float]:
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
            return False, f"the run failed: {exc}", 0.0, 0.0
        spent = float(result.cost.dollars or 0.0), float(result.cost.seconds or 0.0)
        verdict = self.meta.verifier(result.context_wrapper, result)
        verdict = Verdict.coerce(await verdict if inspect.isawaitable(verdict) else verdict)
        return (False, "no verdict", *spent) if verdict is None else (verdict.accepted, verdict.reason or "accepted", *spent)

    async def _running_now(self) -> dict[str, tuple[bool, float]]:
        """The agent as it runs now on each test request: what a variant that stays on the LLM has to match."""
        model = getattr(self.meta.agent, "model", None) or DEFAULT_MODEL
        effort = most_demanding(self.meta.setting(name, self.meta.store.trusted(name) is not None, model)
                                for name in self.meta.procedures)
        attempt = self.tier.route_practice or self._try_route
        measured: dict[str, tuple[bool, float]] = {}
        for case_id, task in {**self.tier.search_cases, **self.tier.labeled}.items():
            accepted, _, *spent = await attempt(task, effort, None)
            measured[case_id] = (bool(accepted), float(spent[0]) if spent else 0.0)
        return measured

    def _cases(self) -> list[str]:
        """The gate's labeled requests, and every request a program served wrongly since the last install."""
        store = self.meta.store
        since = store.state(_LOOP).get("since", 0)
        if self.tier.long_running:
            # A wrong action from before a procedure was last relearned was made in a world that has changed since.
            since = max([since, *(store.state(name).get("trusted_at", 0) for name in self.meta.procedures)])
        recent = store.rejected_cases(since, RUNG)
        return [*self.tier.labeled, *[c for c in dict.fromkeys(recent) if c not in self.tier.labeled]]

    async def judge(self, source: str, cases: Sequence[str] | None = None) -> Judgment:
        tried = [await self._attempt(source, case_id) for case_id in (self._cases() if cases is None else cases)]
        return Judgment(accepted=sum(a.reward >= 1.0 or a.reward == ROUTED_REWARD for a in tried), total=len(tried),
                        wrong=sum(a.outcome == "wrong" for a in tried),
                        handoffs=sum(a.reward in (HANDOFF_REWARD, READ_ONLY_LOST) for a in tried),
                        failures=[f"{a.case_id}: {a.reason}" for a in tried if a.reward < 1.0],
                        routed=sum(a.reward in (ROUTED_REWARD, ROUTED_WRONG) for a in tried),
                        routed_right=sum(a.reward == ROUTED_REWARD for a in tried),
                        acted=sum(a.outcome == "program" for a in tried), attempts=tried)

    def _gate(self, judged: Judgment) -> GateResult:
        # Each route answers for itself: what the program serves must clear the bar with next to no wrong actions,
        # and what it sends to the LLM must do as well as the tier asks of the LLM. A program that serves nothing
        # itself hasn't closed the gap, however well the LLM does.
        own = judged.accepted - judged.routed_right
        own_total = judged.total - judged.routed
        passed = (judged.acted > 0 and own_total > 0 and own / own_total >= self.tier.bar
                  and judged.wrong <= self.tier.max_wrong
                  and (judged.routed == 0 or judged.routed_right / judged.routed >= self.tier.routed_bar))
        return GateResult(judged.accepted, judged.total, passed, judged.failures)

    def _problem(self) -> LearningProblem:
        episodes: list[Episode] = self.meta.store.episodes(last=80)
        return LearningProblem(procedure=DECISION, contract=self.tier.contract, episodes=episodes,
                               practice=self.practice, cases=list(self.tier.labeled), score=self.practice)

    def _mechanisms(self) -> list[tuple[str, str]]:
        found = self.meta.store.setting("program_mechanisms") or []
        return [(m["name"], m["source"]) for m in found] + list(self.tier.mechanisms)

    def _stored(self) -> list[dict[str, Any]]:
        """The memories of the current program, from the versions the store keeps of each."""
        store, found = self.meta.store, []
        for name in store.state(_LOOP).get("memories", []):
            version = store.trusted(MEMORY + name)
            if version is not None:
                found.append({"name": name, **version["info"], "body": version["source"]})
        return found

    def memories(self) -> list[dict[str, Any]]:
        """What the current program stored: each memory's name, description, route, read_only and body."""
        return self._stored()

    def _packed(self, version: dict[str, Any]) -> str:
        return pack_memories(version["source"], self._stored())

    def _store_memories(self, memories: list[dict[str, Any]], created_by: str) -> None:
        """Keep each memory as its own version: one that didn't change keeps the version it has."""
        store = self.meta.store
        state = store.state(_LOOP)
        names = [memory["name"] for memory in memories]
        for name in state.get("memories", []):
            old = store.trusted(MEMORY + name)
            if name not in names and old is not None:
                store.set_state(MEMORY + name, old["version"], "retired")
        for memory in memories:
            key = MEMORY + memory["name"]
            info = {"description": memory["description"], "route": memory["route"],
                    "read_only": bool(memory.get("read_only"))}
            old = store.trusted(key)
            if old is not None and old["source"] == memory["body"] and old["info"] == info:
                continue
            if old is not None:
                store.set_state(key, old["version"], "retired")
            store.add_version(key, memory["body"], state="trusted", created_by=created_by, info=info)
        state["memories"] = names
        store.put_state(_LOOP, state)

    def _shared_tests(self, share_all: bool) -> list[dict[str, Any]]:
        """The labeled requests a variant can practice on: those its gap report shows, never a gate held back."""
        tier = self.tier
        cases = {**tier.search_cases, **(tier.labeled if share_all or not tier.search_cases else {})}
        return [{"id": case_id, "request": task["text"], "should_act": task.get("expect") != "handoff"}
                for case_id, task in cases.items()]

    async def _write(self, problem: LearningProblem, name: str, source: str, memories: Sequence[dict[str, Any]],
                     tests: Sequence[dict[str, Any]] = ()) -> Any:
        learner = SourceLearner(source, name, GENERIC, self.tier.budget, memories=memories, tests=tests)
        choice = await run_tournament(problem, [learner], writer_model=self.tier.writer_model or BUILDER_MODEL,
                                      writer_params=self.tier.writer_params, prices=self.tier.prices)
        return choice.winner

    async def _learn(self, problem: LearningProblem, name: str, source: str) -> tuple[Any, Judgment | None]:
        # A component run again gets the labeled requests a search could show it, as it had when it was admitted.
        # The gate's requests are never among them.
        tests = self._shared_tests(False) if self.tier.search_cases else []
        trial = await self._write(problem, name, source, self._stored(), tests)
        return trial, await self.judge(trial.source) if trial.source else None

    def _install(self, program: str, mechanism: str, problem: LearningProblem, trial: Any, gate: GateResult,
                 events: list[str]) -> None:
        store = self.meta.store
        current = self.current()
        if current is not None:
            store.set_state(DECISION, current["version"], "retired")
        router, memories = unpack_memories(program)
        version = store.add_version(DECISION, router, state="trusted", created_by=mechanism)
        self._store_memories(memories, mechanism)
        for key in self._classes():
            store.put_state(key, {})
        state = store.state(_LOOP)
        state["since"] = store.count()
        self._gap_closed(state)
        store.put_state(_LOOP, state)
        store.put_state(_OUTCOMES, {})
        store.log_learning(name=DECISION, mechanism=mechanism, version=version, episodes=len(problem.episodes),
                           writer_calls=trial.writer_calls, dollars=trial.dollars, gate=gate.to_dict(),
                           outcome="passed, installed")
        events.append(f"program: {mechanism} wrote {DECISION}@{version}; gate {gate.accepted}/{gate.total} "
                      "passed, installed")
        if memories:
            events.append("program: memories " + ", ".join(self._label(memory) for memory in memories))

    def _gap_closed(self, state: dict[str, Any]) -> None:
        """A gap that closed takes its attempts with it, when the tier counts them for each gap."""
        if self.tier.long_running:
            state.update(attempts=0, searches=0)

    @staticmethod
    def _label(memory: dict[str, Any]) -> str:
        return f"{memory['name']} ({memory['route']}{', read-only' if memory.get('read_only') else ''})"

    async def _retest(self, events: list[str]) -> bool:
        """Switch classes back on if the current program now passes: what broke may not have been the program."""
        store, current = self.meta.store, self.current()
        off = [key for key in self._classes() if store.state(key).get("off")]
        if current is None or not off:
            return False
        judged = await self.judge(self._packed(current))
        gate = self._gate(judged)
        label = f"{DECISION}@{current['version']}"
        if not gate.passed:
            events.append(f"program: re-tested {label}; still refused, {judged.accepted}/{judged.total}, "
                          f"{judged.wrong} wrong actions")
            if self.tier.long_running:
                await self.check_procedures(events)
            return False
        for key in off:
            store.put_state(key, {})
        state = store.state(_LOOP)
        state["since"] = store.count()
        self._gap_closed(state)
        store.put_state(_LOOP, state)
        store.put_state(_OUTCOMES, {})
        store.log_learning(name=DECISION, mechanism="retest", version=current["version"], episodes=0, writer_calls=0,
                           dollars=0.0, gate=gate.to_dict(), outcome="passed, switched back on")
        events.append(f"program: re-tested {label}; gate {gate.accepted}/{gate.total} passed, switched back on: "
                      f"{', '.join(key.split(':', 1)[1] for key in off)}")
        return True

    async def check_procedures(self, events: list[str]) -> None:
        """Put each procedure the program stands on through its own gate. One that fails is what broke the
        program, so it is quarantined for its own loop to relearn, and no builder rewrites the program."""
        store = self.meta.store
        for name in self.meta.procedures:
            checked = store.trusted(name)
            gate = await self.meta.recheck(name)
            current = store.trusted(name)
            if gate is None or gate.passed or current is None or current["version"] != checked["version"]:
                continue  # it holds, or it was quarantined or replaced while the gate ran
            reason = f"its own gate now passes {gate.accepted}/{gate.total}"
            self.meta.quarantine(name, reason)
            events.append(f"detector: quarantined {name}@{current['version']} ({reason})")
            emit(kind="quarantined", procedure=name, version=current["version"], detail=reason)

    def _repairing(self) -> list[str]:
        """Procedures that were trusted and are quarantined now, with no version trusted in their place."""
        store = self.meta.store
        return [name for name in self.meta.procedures
                if store.trusted(name) is None and "quarantined_at" in store.state(name)]

    async def improve(self, events: list[str]) -> None:
        store, tier = self.meta.store, self.tier
        if await self._retest(events):
            return
        if tier.long_running and self._repairing():
            # The re-test found a procedure at fault. No builder was asked, so the gap keeps this attempt.
            state = store.state(_LOOP)
            state["attempts"] = max(0, state.get("attempts", 0) - 1)
            store.put_state(_LOOP, state)
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

    def _stored_summary(self, program: str) -> str:
        """What a component stored and the program it returned: a rewrite doesn't see those in its own source."""
        router, memories = unpack_memories(program)
        stored = "\n".join(f"- {self._label(memory)}: {memory['description']}" for memory in memories)
        return f"\n\nIt stored these memories:\n{stored or '(none)'}\n\nThe program it returned:\n{router}"

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
        signal = ServiceGap(self, Good(wrong_action=1.0 + spec.penalty), share_all=spec.share_all)
        if spec.lower_reasoning and not editing:
            signal.running_now = await self._running_now()
            events.append(f"search: what's running now got {sum(right for right, _ in signal.running_now.values())} of "
                          f"{len(signal.running_now)} test requests right")
        built: dict[Any, list[dict[str, Any]]] = {}  # what each variant stored, so its children start from it
        starting = self._stored()
        learning = replace(problem, cases=[])  # a variant is measured below, not by the tournament
        shared = [] if editing else self._shared_tests(spec.share_all)

        def logged(source: str, outcome: str, **fields: Any) -> None:
            self.search_log.append({"design": source_id(source), "outcome": outcome, "source": source, **fields})
            detail = "; ".join(f"{k}={v}" for k, v in fields.items() if k != "program")
            events.append(f"search: design {source_id(source)} {outcome} {detail}")

        async def inherited(parent: Any) -> list[dict[str, Any]]:
            """What a variant's parent stored: from this search, or from the archive of an earlier one."""
            if parent in built or parent is None:
                return built.get(parent, starting)
            history = await archive.get_measurement_history(*parent)
            return history[-1].metadata.get("memories", starting) if history else starting

        async def evaluate(candidate: Any, questions: Any = None) -> GapMeasurement:
            source = candidate.content if hasattr(candidate, "content") else candidate
            if editing:
                trial = SimpleNamespace(source=source, writer_calls=0, dollars=0.0, note="")
            else:
                memories = await inherited(getattr(candidate, "parent_id", None))
                try:
                    trial = await self._write(learning, f"design_{source_id(source)}", source, memories, shared)
                except UnsafeMechanism as exc:
                    logged(source, "refused by guard", reason=str(exc))
                    return GapMeasurement(score=0.0, signal_id="program_gate", signal_version=1, triggered=True,
                                          raw_value=0.0, feedback=f"refused by the mechanism guard: {exc}", metadata={})
            if not trial.source:
                logged(source, "wrote nothing", note=trial.note, writer_calls=trial.writer_calls)
                return GapMeasurement(score=0.0, signal_id="program_gate", signal_version=1, triggered=True,
                                      raw_value=0.0, feedback=f"it wrote no program: {trial.note}", metadata={})
            if hasattr(candidate, "ref"):
                built[candidate.ref] = unpack_memories(trial.source)[1]
            seen, judged = await signal.tried(trial.source)
            passed = signal.admits(seen, judged)
            outcome = "admitted" if passed else "refused by gate" if judged is not None else "refused on its own cases"
            looked_at = seen if seen is not None else judged
            on_gate = {} if judged is None else {"right": f"{judged.accepted}/{judged.total}", "wrong": judged.wrong,
                                                 "handoffs": judged.handoffs,
                                                 "routed": f"{judged.routed_right}/{judged.routed}"}
            measurement = signal.measurement(seen, judged, passed=passed,
                                             notes="" if editing else self._stored_summary(trial.source))
            measurement.metadata["memories"] = unpack_memories(trial.source)[1]
            logged(source, outcome, **on_gate, seen=f"{looked_at.accepted}/{looked_at.total}", seen_wrong=looked_at.wrong,
                   seen_acted=looked_at.acted, seen_routed=f"{looked_at.routed_right}/{looked_at.routed}",
                   search_score=round(measurement.score, 2), writer_calls=trial.writer_calls, program=trial.source)
            if passed and (not best or measurement.score > best["score"]):
                best.update(score=measurement.score, source=source, trial=trial, judged=judged)
            return measurement

        def validate(source: str) -> str | None:
            try:
                if editing:
                    check_program(source)
                else:
                    check_mechanism(source)
                    check_runnable(source)
            except (UnsafeMechanism, UnsafeProgram) as exc:
                return str(exc)
            return frozen_change(seed_source, source) if editing else None

        calls = tier.budget.writer_calls if tier.budget.writer_calls is not None else "unlimited"
        run = f"\n\nThis is design search {spec.label}." if spec.label else ""
        rules = contract_text(tier.contract, GENERIC)
        admission = _OR_COSTS_LESS if spec.lower_reasoning else _NEEDS_A_PROGRAM
        mutation = (EDIT_PROMPT.replace("{contract}", rules) if editing
                    else MUTATION_PROMPT.replace("{writer_calls}", str(calls)).replace("{admission}", admission))
        crossover = (COMBINE_PROMPT.replace("{contract}", rules) if editing
                     else CROSSOVER_PROMPT.replace("{writer_calls}", str(calls)).replace("{admission}", admission))
        search = build_search(spec.method, proposer_model=spec.designer_model or tier.writer_model or BUILDER_MODEL,
                              evaluator=evaluate, mutation_prompt=mutation + run, crossover_prompt=crossover + run,
                              validate=validate, proposer_params=dict(spec.designer_params),
                              population=spec.population, generations=spec.generations, rng=random.Random(spec.seed))
        archive = SQLiteArchive(spec.archive)
        seed = genesis(DECISION if editing else "program_mechanism", ArtifactKind.CODE, seed_source, created_by=name)
        limit = SearchBudget(max_candidates=spec.population * spec.generations)
        proposals = search.propose(seed, signal, archive, limit)
        async for _ in proposals:
            if spec.stop_when_admitted and best:
                await proposals.aclose()
                break
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
        self._install(best["trial"].source, design, problem, best["trial"], replace(self._gate(best["judged"]), passed=True),
                      events)


__all__ = ["DECISION", "RUNG", "DecisionLoop", "Judgment", "ProgramSearch", "ProgramTier", "ServiceGap"]
