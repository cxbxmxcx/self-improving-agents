"""Metacognition: chapter 6's online loop, as a library.

    meta = Metacognition(procedures=[find_shared_starts], verifier=check)
    agent = Agent(name="...", instructions="...", tools=[...], metacognition=meta)
    result = await Runner.run(agent, request, context=world)
    result.metacognition   # effort, procedures offered, verdict, events

For each run the Runner asks `before_run` for a plan and hands the result to
`after_run`. Between the two, the agent loop is untouched; the agent sees two
effects only: a reasoning setting and a set of recalled procedure tools, each
with one line of context (study §3.7).

before_run   route the request to the procedures whose class it belongs to, recall
             the trusted ones among them (helix's own recall, so the tools the
             agent can call are the ones its context names), bind each to an `api`
             of read-only tools, and choose the reasoning setting by lookup
after_run    judge the run (verifier now, or `feedback` later), record it, let the
             detector quarantine a failing procedure (blaming the procedures the
             run called), and relearn an untrusted one every `relearn_after` runs;
             the gate admits a new version only when it clears the bar on held-back
             cases

Practice and frozen runs (the gate's, a learner's, or a calibration sweep's) are
never recorded, so a procedure never learns from its own trials.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import random
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from functools import wraps
from typing import Any

from helix.memory.base import MemoryContext, Scope, ScopeKey
from helix.memory.code_skills import tools_from_entries
from helix.memory.hooks import recall_skills
from helix.memory.learning_mechanisms import Episode, Writer
from helix.memory.procedural import ProceduralMemory
from helix.trajectory import StepKind

from helix.metacognition.ladders import ladder_for, most_demanding
from helix.metacognition.api import SkillAPI
from helix.metacognition.events import emit, publish
from helix.metacognition.judge import Judge
from helix.metacognition.mechanisms import Mechanism, resolve_mechanism, spent, writer_cost
from helix.metacognition.parts import Controller, Detector, Gate, GateResult, LearningBudget, Practice, Router, Verdict
from helix.metacognition.procedure import Procedure
from helix.metacognition.prompts import GENERIC, PACKS, PromptPack
from helix.metacognition.sandbox import sandboxed_tool
from helix.metacognition.selection import LearnerChoice, LearnerSelection, LearningProblem, design_rank, run_tournament
from helix.metacognition.sources import CONTROL as SEED_CONTROL
from helix.metacognition.sources import SEEDS, SourceLearner
from helix.metacognition.store import MetaStore
from helix.metacognition.model_settings import DEFAULT_MODEL
from helix.metacognition.lexical import embed

GLOBAL = ScopeKey(Scope.GLOBAL, "*")
MODES = ("online", "frozen", "practice")
ATTRIBUTIONS = ("called", "offered")
LEARNING = ("background", "inline", "worker")
EXECUTORS = ("auto", "guarded", "subprocess")
# How many judged runs per procedure version the detector remembers having counted.
COUNTED = 256
# Who wrote a version that may run in-process under "auto": people, not models.
HUMAN = "human"


async def _maybe(value: Any) -> Any:
    return await value if inspect.isawaitable(value) else value


@dataclass
class MetaReport:
    """What metacognition did for one run; `RunResult.metacognition`."""

    run_id: str
    mode: str
    case_id: str
    effort: str | None = None
    matched: list[str] = field(default_factory=list)
    offered: list[str] = field(default_factory=list)  # "name@version"
    verdict: Verdict | None = None
    order: int | None = None
    events: list[str] = field(default_factory=list)
    called: list[str] = field(default_factory=list)  # "name@version": the offered procedures the run used
    program: str | None = None  # what a decision program did with the request: "served", "routed", "read_only" or
    # "handoff"; None when no program had the first try
    memory: str | None = None  # the stored memory the program's router chose, if it chose one

    def __str__(self) -> str:
        verdict = "pending" if self.verdict is None else ("accepted" if self.verdict.accepted else "rejected")
        text = f"effort={self.effort} offered={self.offered or '[]'} verdict={verdict}"
        return text + "".join(f"\n  {event}" for event in self.events)


@dataclass
class RunPlan:
    run_id: str
    mode: str
    case_id: str
    case_data: Any
    request: str
    matched: list[str] = field(default_factory=list)
    offered: list[tuple[str, int]] = field(default_factory=list)
    tools: list[Any] = field(default_factory=list)
    tier: ProceduralMemory | None = None
    skill_k: int = 2
    effort: str | None = None
    report: MetaReport | None = None
    model: str | None = None
    scores: dict[str, float] = field(default_factory=dict)
    program: dict[str, Any] | None = None  # the trusted decision program that gets the first try at this request
    program_outcome: str | None = None  # "served", "routed", "read_only" or "handoff"
    memory: str | None = None  # what a program's route tells the agent to read first
    read_only: bool = False  # a program said this request should not be acted on: the agent gets no action tools
    program_action: str | None = None
    decider_calls: list[dict[str, Any]] = field(default_factory=list)  # Jev's calls, charged to the run


def called_procedures(trajectory: Any, offered: Sequence[tuple[str, int]]) -> list[tuple[str, int]]:
    """The offered procedures a run actually called (a call stopped by the tool-call cap doesn't count)."""
    used = {s.payload.get("name") for s in getattr(trajectory, "steps", [])
            if s.kind == StepKind.TOOL_CALL and not s.payload.get("cap_hit")}
    return [(name, version) for name, version in offered if name in used]


@dataclass
class AgentPractice:
    """Practice through the agent: run it on a recorded case with the candidate installed, then ask the verifier.

    `world(case_data)` rebuilds the case's world (its run context), where
    `case_data` is what the run was given as `RunConfig(case_data=...)`. This
    measures a procedure the way the agent uses it (study §5.3, attempt B).
    """

    world: Callable[[Any], Any]
    max_turns: int = 10
    run_config: Any = None

    def bind(self, meta: Metacognition, name: str) -> Practice:
        async def practice(source: str, case_id: str) -> tuple[float, str, str]:
            from helix.metacognition.exceptions import AgentsException
            from helix.metacognition.run import RunConfig, Runner

            run = meta.store.case(case_id)
            if run is None:
                return 0.0, f"no recorded run for case {case_id}", ""
            config = replace(self.run_config or RunConfig(), metacognition_mode="practice",
                             practice_candidate=(name, source), case_id=case_id, case_data=run["case_data"])
            context = self.world(run["case_data"] if run["case_data"] is not None else case_id)
            try:
                result = await Runner.run(meta.agent, run["request"], context=context, max_turns=self.max_turns,
                                          run_config=config)
            except AgentsException as exc:
                return 0.0, f"the run failed: {exc}", ""
            verdict = Verdict.coerce(await _maybe(meta.verifier(result.context_wrapper, result)))
            if verdict is None:
                return 0.0, "the verifier gave no verdict", ""
            return (1.0 if verdict.accepted else 0.0), verdict.reason or "accepted", ""

        return practice


class Metacognition:
    """The online metacognition loop for one agent deployment.

    procedures      the procedures worth learning (`@procedure` stubs or Procedure objects)
    verifier        `verifier(context, result)` -> Verdict, bool, (bool, reason), checker dict, or None
                    (None: wait for `feedback`); sync or async. A `Judge` is a model-based verifier
                    whose verdicts never install code on their own
    learner         a mechanism name ("value_learning", ...) or instance, such as a SourceLearner
                    (chapter 7's mechanisms as source); `prompts` is its PromptPack
    learners        a learner per procedure name, for the procedures that learn better another way
    selection       a LearnerSelection: when learning keeps failing for a procedure, hold a tournament
                    of mechanisms on its own judged runs and keep the winner as its learner (A5, L2).
                    A learner chosen this way, kept in the store, comes before `learners` and `learner`
    practice        how a candidate is tried on a case: an AgentPractice (through the agent) or a
                    function (source, case_id) -> (reward, reason, observations)
    gate, detector, controller
                    the policy parts; study defaults
    store           a MetaStore, a directory path, or None for memory
    router          which procedures a request belongs to (a Router; `calibrate_routing` sets its
                    thresholds). Without one: the `recall_k` best procedures with score >= `relevance`
    recall_k        procedures offered per run (k=2: E4 found k=1 picked the wrong one 4 of 16 times)
    attribution     whose fault a rejected run is: "called" (the procedures the run called) or
                    "offered" (every procedure it was offered, as the study's detector counted)
    relearn_after   runs between learning attempts while a procedure is untrusted
    learning        "background" (an asyncio task beside serving), "inline" (finish before the run
                    returns), or "worker" (serving only records; `metacog worker` learns, see learn_due)
    budget          a LearningBudget: attempts or dollars per procedure per 24 hours
    approval        "auto" installs on a passing gate; "human" proposes; None picks "auto" when a
                    programmatic verifier is set and "human" otherwise (no verifier, or a Judge)
    effort_param    send the controller's setting under this one parameter name instead of through
                    the model family's effort ladder (helix.metacognition.ladders)
    executor        "guarded" runs procedure code in-process under helix's guard (fast; not a security
                    boundary); "subprocess" runs each call in a fresh child process (see sandbox.py);
                    "auto" (the default) runs versions people wrote guarded and model-written ones in
                    the subprocess

    Measured floors in the store (`MetaStore.put_floors`, `Floors.record`) join the controller's
    per-model floors; floors given in code win.
    """

    def __init__(
        self,
        procedures: Sequence[Procedure],
        *,
        verifier: Callable[..., Any] | None = None,
        learner: str | Mechanism = "value_learning",
        learners: Mapping[str, str | Mechanism] | None = None,
        selection: LearnerSelection | None = None,
        prompts: PromptPack | str = GENERIC,
        writer_model: str | None = None,
        writer_params: dict[str, Any] | None = None,
        practice: AgentPractice | Practice | None = None,
        gate: Gate | None = None,
        detector: Detector | None = None,
        controller: Controller | None = None,
        store: MetaStore | str | None = None,
        router: Router | None = None,
        recall_k: int = 2,
        relevance: float = 0.0,
        attribution: str = "called",
        relearn_after: int = 4,
        experience_limit: int | None = None,
        learning: str = "background",
        budget: LearningBudget | None = None,
        approval: str | None = None,
        gating_sources: Sequence[str] = ("verifier", "human", "harness"),
        effort_param: str | None = None,
        skill_timeout: float = 5.0,
        prices: dict[str, tuple[float, ...]] | None = None,
        executor: str = "auto",
        programs: Any = None,
    ) -> None:
        if learning not in LEARNING:
            raise ValueError(f"learning must be one of {LEARNING}")
        if executor not in EXECUTORS:
            raise ValueError(f"executor must be one of {EXECUTORS}")
        if attribution not in ATTRIBUTIONS:
            raise ValueError(f"attribution must be one of {ATTRIBUTIONS}")
        self.executor = executor
        self.programs = None
        if programs is not None:
            from helix.metacognition.decisions import DecisionLoop

            self.programs = DecisionLoop(self, programs)  # the decision tier (helix.metacognition.decisions)
        self.budget = budget
        self.attribution = attribution
        self.procedures = {p.name: p for p in procedures}
        self.verifier = verifier
        self.learner = resolve_mechanism(learner, prompts)
        unknown = sorted(set(learners or {}) - set(self.procedures))
        if unknown:
            raise ValueError(f"learners name no procedure: {unknown}")
        self.learners = {name: resolve_mechanism(each, prompts) for name, each in (learners or {}).items()}
        self.prompts = PACKS[prompts] if isinstance(prompts, str) else prompts
        self.selection = selection
        self._candidates = [self._as_learner(each) for each in (selection.candidates if selection else ())]
        self._chosen: dict[str, tuple[str, Mechanism]] = {}  # procedure -> (choice id, learner) from the store
        self.writer_model = writer_model
        self.writer_params = dict(writer_params or {"reasoning_effort": "medium"})
        from helix.metacognition.recording import install

        install()  # learning can run in a worker that serves no run: route its Claude calls too
        self.practice = practice
        self.gate = gate or Gate(holdout=3)
        self.detector = detector or Detector()
        self.controller = controller or Controller()
        self.store = store if isinstance(store, MetaStore) else MetaStore(store)
        # A router given in code wins; else the one `metacog calibrate` saved with the deployment.
        saved = self.store.setting("router") if router is None else None
        self.router = router or (Router.from_dict(saved) if saved else Router(k=recall_k, default=relevance))
        self.recall_k = self.router.k
        self.relearn_after = relearn_after
        # The most runs a learner is handed. Without a limit an attempt that comes late gets every run since the
        # quarantine, and a learner that writes a candidate for each run costs more the longer it waited.
        self.experience_limit = experience_limit
        self.learning = learning
        self.approval = approval or ("auto" if verifier is not None and not isinstance(verifier, Judge) else "human")
        self.gating_sources = tuple(gating_sources)
        self.effort_param = effort_param
        self.skill_timeout = skill_timeout
        self.prices = prices
        self.agent: Any = None
        self._descriptions = {name: p.description for name, p in self.procedures.items()}
        # Recall tiers by the trusted versions they hold; a change of trust makes a new key.
        self._tiers: dict[tuple[tuple[str, int], ...], ProceduralMemory] = {}
        self._tasks: dict[str, asyncio.Task[Any]] = {}
        for name, proc in self.procedures.items():
            if proc.reference is not None and not self.store.versions(name):
                self.store.add_version(name, proc.reference, state="trusted", created_by="human")

    # ------------------------------------------------------------ managing procedures

    def install(self, name: str, source: str, *, created_by: str = "human") -> int:
        """Trust `source` as the procedure's live version now (a person's promotion)."""
        from helix.metacognition.procedure import _check_source

        _check_source(source, name)
        return self._trust(name, self.store.add_version(name, source, state="candidate", created_by=created_by))

    def approve(self, name: str, version: int) -> int:
        """Trust a version the gate proposed when approval is "human"."""
        approved = approve_version(self.store, name, version)
        emit(kind="approved", procedure=name, version=version)
        return approved

    def quarantine(self, name: str, reason: str = "by hand") -> None:
        quarantine_procedure(self.store, name, reason, relearn_after=self.relearn_after)

    async def recheck(self, name: str) -> GateResult | None:
        """Put the trusted version of `name` through its own gate again, and say how it did.

        A recheck costs what the gate's practice costs, which is nothing when that practice is a function. None
        when there is nothing to check: no trusted version, or a gate that holds no cases of its own. The result
        says something about today only if the gate's cases are kept current.
        """
        current = self.store.trusted(name)
        cases = list(self.gate.cases or [])
        if current is None or not cases or (self.practice is None and self.gate.practice is None):
            return None
        return await self.gate.check(current["source"], self._practice_for(name, self.gate.practice), cases)

    def _trust(self, name: str, version: int) -> int:
        return trust_version(self.store, name, version)

    def proposals(self) -> list[dict[str, Any]]:
        return [v for v in self.store.versions() if v["state"] == "proposed"]

    def effort_params(self, model: str, effort: str) -> dict[str, Any]:
        """The call parameters that set `effort` on `model`: its family's ladder, or `effort_param` when set."""
        if self.effort_param is not None:
            return {self.effort_param: effort}
        return ladder_for(model).params(effort)

    # ------------------------------------------------------------ before the run

    def _match(self, request: str) -> list[str]:
        """The procedures whose class the request belongs to, best first."""
        return self.router.match(request, self._descriptions)[0]

    async def _tier_for(self, entries: list[tuple[str, int, str, str]]) -> ProceduralMemory | None:
        if not entries:
            return None
        tier = ProceduralMemory(embedding_fn=embed)
        for name, version, source, created_by in entries:
            await tier.save_skill(name, self.procedures[name].description, source, body_kind="python",
                                  scope_key=GLOBAL, metadata={"created_by": created_by, "version": version})
        return tier

    async def _trusted_tier(self, names: Sequence[str]) -> ProceduralMemory | None:
        """A recall tier of the trusted versions of `names`: what a request in their classes may be offered."""
        trusted = [(name, v) for name in self.procedures if name in names
                   and (v := self.store.trusted(name)) is not None]
        key = tuple((name, v["version"]) for name, v in trusted)
        if key not in self._tiers:
            entries = [(name, v["version"], v["source"], v["created_by"]) for name, v in trusted]
            self._tiers[key] = await self._tier_for(entries)
        return self._tiers[key]

    def _api_tools(self) -> list[Any]:
        tools: dict[str, Any] = {}
        for proc in self.procedures.values():
            tools.update({tool.name: tool for tool in proc.api})
        return list(tools.values())

    def _procedure_tool(self, entry: Any, api: SkillAPI, *, candidate: bool = False) -> Any:
        """The tool for one recalled version: guarded in-process, or in a child process for model-written
        code under "auto" (and for everything under "subprocess").

        A trusted version answers only while it is still the trusted one. A request can be in flight when its
        procedure is quarantined, and its call then gets an error to work around, not a stale answer. A
        `candidate` is a version being tried in practice, which nothing has trusted yet.
        """
        name, body = entry.content["skill_name"], entry.content["body"]
        by_person = entry.metadata.get("created_by") == HUMAN
        if self.executor == "subprocess" or (self.executor == "auto" and not by_person):
            tool = sandboxed_tool(name, entry.content.get("description", ""), body, api, timeout=self.skill_timeout)
        else:
            tool = tools_from_entries([entry], bind={"api": api}, timeout=self.skill_timeout)[0]
        if candidate:
            return tool
        version, run = int(entry.metadata.get("version", 0)), tool.fn

        @wraps(run)  # it reads as the tool it guards: its name, and where the code runs
        async def while_trusted(**arguments: Any) -> Any:
            current = self.store.trusted(name)
            if current is None or current["version"] != version:
                return {"error": f"{name} was withdrawn while this request was being served. "
                                 "Work the answer out with the other tools."}
            return await run(**arguments)

        return replace(tool, fn=while_trusted)

    def _controller(self) -> Controller:
        """The controller, with the floors measured into the store joining its own (its own win)."""
        measured = self.store.floors()
        if not measured:
            return self.controller
        return replace(self.controller, per_model={**measured, **self.controller.per_model})

    def setting(self, procedure: str | None, trusted: bool, model: str) -> str | None:
        """The reasoning setting a run on `model` gets for `procedure`, trusted or not, from the
        controller's own floors and those recorded in the store."""
        return self._controller().effort(procedure, trusted, model)

    async def before_run(self, agent: Any, ctx: Any, request: str, config: Any, memory_context: MemoryContext,
                         model: str | None = None) -> RunPlan:
        mode = config.metacognition_mode
        if mode not in MODES:
            raise ValueError(f"metacognition_mode must be one of {MODES} or 'off', not {mode!r}")
        if mode == "online":
            self.agent = agent
        model = model or config.model or getattr(agent, "model", None) or DEFAULT_MODEL
        case_id = config.case_id or "case-" + hashlib.sha1(request.encode("utf-8")).hexdigest()[:12]
        matched, scores = self.router.match(request, self._descriptions)
        plan = RunPlan(f"run_{uuid.uuid4().hex}", mode, case_id, config.case_data, request, matched=matched,
                       skill_k=self.router.k, model=model, scores=scores)
        if mode == "practice":
            name, source = config.practice_candidate
            tier = await self._tier_for([(name, 0, source, "candidate")])
        else:
            tier = await self._trusted_tier(matched)
        if tier is not None:
            entries = await recall_skills(tier, request, memory_context, self.recall_k)
            api = SkillAPI(self._api_tools(), ctx, asyncio.get_running_loop(), executor=config.tool_executor)
            plan.tools = [self._procedure_tool(entry, api, candidate=mode == "practice") for entry in entries
                          if entry.content.get("body_kind") == "python"]
            plan.tier = tier
            plan.offered = [(e.content["skill_name"], int(e.metadata.get("version", 0))) for e in entries]
        if mode == "online" and self.programs is not None:
            plan.program = self.programs.current()
        if mode != "frozen":
            # The most demanding floor among the procedures in play, so no procedure runs below its own.
            controller = self._controller()
            if plan.offered:
                plan.effort = most_demanding(controller.effort(name, True, model) for name, _ in plan.offered)
            elif plan.matched:
                plan.effort = most_demanding(controller.effort(name, False, model) for name in plan.matched)
        if config.effort is not None:
            plan.effort = config.effort
        plan.memory = config.memory
        plan.report = MetaReport(plan.run_id, mode, case_id, plan.effort, list(plan.matched),
                                 [f"{name}@{version}" for name, version in plan.offered])
        return plan

    # ------------------------------------------------------------ after the run

    async def after_run(self, plan: RunPlan, ctx: Any, result: Any, *, trajectory: Any,
                        failed: str | None = None) -> None:
        if plan.mode != "online":
            return
        report = plan.report
        if failed is not None:
            verdict: Verdict | None = Verdict(False, failed, "harness")
        elif self.verifier is not None:
            verdict = Verdict.coerce(await _maybe(self.verifier(ctx, result)))
        else:
            verdict = None
        episode = Episode.from_trajectory(plan.case_id, trajectory, bool(verdict and verdict.accepted),
                                          verdict.reason if verdict else "awaiting a verdict")
        called = called_procedures(trajectory, plan.offered)
        report.called = [f"{name}@{version}" for name, version in called]
        report.program = plan.program_outcome
        cost = getattr(result, "cost", None)
        usage = getattr(result, "usage", None)
        report.order = self.store.record(
            run_id=plan.run_id, case_id=plan.case_id, case_data=plan.case_data, request=plan.request, episode=episode,
            verdict=verdict, matched=plan.matched, offered=report.offered, effort=plan.effort,
            dollars=float(getattr(cost, "dollars", 0.0) or 0.0),
            reasoning_tokens=int(getattr(usage, "reasoning_tokens", 0) or 0),
            seconds=float(getattr(cost, "seconds", 0.0) or 0.0),
            answer=None if result is None else result.final_output, model=plan.model, called=report.called,
            scores=plan.scores)
        report.verdict = verdict
        await publish(kind="run", run_id=plan.run_id, effort=plan.effort, offered=list(report.offered),
                      detail="pending" if verdict is None else ("accepted" if verdict.accepted else "rejected"))
        # A run a decision program sent on read-only had no way to act, and a run in which a tool could not reach
        # its service failed for that reason. Neither verdict says anything about a procedure.
        blameless = plan.read_only or getattr(ctx, "unavailable", False)
        if verdict is not None and verdict.source in self.gating_sources and not blameless:
            self._observe(called if self.attribution == "called" else plan.offered, verdict, report.events,
                          plan.run_id)
        await self._maybe_learn(report.events)
        if self.programs is not None:
            if self.programs.after_run(plan, verdict, report.events) and self.programs.tier.long_running:
                # What made the program act wrongly may be a procedure it stands on. Checked now, a stale one
                # stops serving the agent as well, before the agent's own wrong answers have to show it.
                await self.programs.check_procedures(report.events)
            await self.programs.maybe_improve(report.events)

    async def feedback(self, run_id: str, accepted: bool, reason: str = "", *, source: str = "human"
                       ) -> list[str]:
        """A verdict that arrives after the run, from a person or a later check. It replaces the run's earlier
        verdict, in the store and in the detector's window. Returns what it set off: a quarantine, learning."""
        verdict = Verdict(accepted, reason, source)
        run = self.store.set_verdict(run_id, verdict)
        if run is None:
            raise KeyError(f"no recorded run {run_id}")
        events: list[str] = []
        if source in self.gating_sources:
            # A run recorded before metacog kept `called` has only `offered` to go on.
            blamed = run["offered"] if self.attribution == "offered" or run["called"] is None else run["called"]
            self._observe([(item.split("@")[0], int(item.split("@")[1])) for item in blamed], verdict, events,
                          run_id)
        await self._maybe_learn(events)
        return events

    def _observe(self, offered: list[tuple[str, int]], verdict: Verdict, events: list[str],
                 run_id: str | None = None) -> None:
        for name, version in offered:
            current = self.store.trusted(name)
            if current is None or current["version"] != version:
                continue  # replaced or quarantined since this run
            state = self.store.state(name)
            window = state.get("window") or [[None, ok] for ok in state.get("recent", [])]
            counted = state.get("counted", [])
            window, quarantine = self.detector.record(window, run_id, verdict.accepted, counted,
                                                      source=verdict.source)
            if run_id is not None and run_id not in counted:
                counted = [*counted, run_id][-COUNTED:]
            state.update(window=window, recent=[entry[1] for entry in window], counted=counted)
            if quarantine:
                self.store.set_state(name, version, "quarantined")
                count = self.store.count()
                state.update(recent=[], window=[], counted=[], quarantined_at=count,
                             next_attempt=count + self.relearn_after, reason=verdict.reason)
                if current.get("created_by") not in (None, "human") and "trusted_at" in state:
                    state["lasted"] = count - state["trusted_at"]  # runs a learned version served
                events.append(f"detector: quarantined {name}@{version} ({verdict.reason})")
                emit(kind="quarantined", procedure=name, version=version, detail=verdict.reason)
            self.store.put_state(name, state)

    def _due(self, name: str, count: int) -> bool:
        """Whether an attempt to learn `name` is due: untrusted, nothing awaiting a person, and its turn."""
        if self.store.trusted(name) is not None or name in self._tasks:
            return False
        if any(v["state"] == "proposed" for v in self.store.versions(name)):
            return False  # a passing version awaits a person; do not stack up more
        return count >= self.store.state(name).get("next_attempt", self.relearn_after)

    def _schedule(self, name: str, count: int) -> None:
        """Set the next attempt `relearn_after` runs from `count`."""
        state = self.store.state(name)
        state["next_attempt"] = count + self.relearn_after
        state.pop("attempt", None)
        self.store.put_state(name, state)

    def _experience(self, name: str) -> list[Episode] | None:
        """The judged runs since quarantine to learn from, or None while none of them was accepted."""
        episodes = self.store.episodes(since=self.store.state(name).get("quarantined_at", 0), procedure=name,
                                       last=self.experience_limit)
        return episodes if any(e.success for e in episodes) else None

    async def _over_budget(self, name: str, events: list[str]) -> bool:
        """Log and report a skipped attempt when the budget refuses one."""
        if self.budget is None:
            return False
        if self.budget.runs_per_day is None:
            recent = self.store.learning(name, within=timedelta(days=1))
        else:  # the stream's own clock: a day is that many recorded runs
            recent = self.store.learning(name, since_run=self.store.count() - self.budget.runs_per_day)
        refusal = self.budget.refusal(recent)
        if refusal is None:
            return False
        self.store.log_learning(name=name, mechanism=self.learner_for(name).name, version=None, episodes=0,
                                writer_calls=0,
                                dollars=0.0, gate=None, outcome=f"skipped: {refusal}")
        events.append(f"learner: skipped {name}: {refusal}")
        await publish(kind="skipped", procedure=name, detail=refusal)
        return True

    async def _maybe_learn(self, events: list[str]) -> None:
        if self.learning == "worker":
            return  # serving only records; a worker learns (learn_due)
        count = self.store.count()
        for name in self.procedures:
            if not self._due(name, count):
                continue
            self._schedule(name, count)  # serving advances first, so a slow attempt is never stacked
            episodes = self._experience(name)
            if episodes is None or await self._over_budget(name, events):
                continue
            if self.learning == "inline":
                await self._learn(name, episodes, events)
            else:
                # The run's report is returned before the attempt ends, so the attempt reports to the
                # store and the event bus instead.
                events.append(f"learner: learning {name} in the background")
                task = asyncio.create_task(self._learn(name, episodes, []))
                self._tasks[name] = task
                task.add_done_callback(lambda _t, name=name: self._tasks.pop(name, None))

    async def learn_due(self, *, holder: str = "worker") -> list[str]:
        """One pass of a learning worker: run each attempt that is due, inline, and say what happened.

        The schedule advances only when an attempt ends (installed, proposed, refused or failed), so
        an attempt cut short by a crash or a restart is due again on the next pass. While one runs,
        the procedure's state names its holder and start time.
        """
        events: list[str] = []
        count = self.store.count()
        for name in self.procedures:
            if not self._due(name, count):
                continue
            episodes = self._experience(name)
            if episodes is None or await self._over_budget(name, events):
                self._schedule(name, count)
                continue
            state = self.store.state(name)
            interrupted = state.get("attempt")
            if interrupted:
                events.append(f"learner: retrying {name}; the attempt {interrupted['holder']} started at "
                              f"{interrupted['started']} never finished")
            state["attempt"] = {"holder": holder, "started": datetime.now(UTC).isoformat()}
            self.store.put_state(name, state)
            await self._learn(name, episodes, events)
            self._schedule(name, count)
        return events

    async def wait_for_learning(self) -> None:
        """Wait for background learning attempts started so far."""
        while self._tasks:
            await asyncio.gather(*list(self._tasks.values()), return_exceptions=True)

    def learner_for(self, name: str) -> Mechanism:
        """The mechanism that learns procedure `name`: the one a tournament chose (kept in the store, so
        every process sees it), else its own (`learners`), else the deployment's."""
        chosen = (self.store.setting("learners") or {}).get(name)
        if not chosen:
            return self.learners.get(name, self.learner)
        cached = self._chosen.get(name)
        if cached is None or cached[0] != chosen["id"]:
            learner = (SourceLearner(chosen["source"], chosen["mechanism"], self.prompts) if chosen.get("source")
                       else resolve_mechanism(chosen["mechanism"], self.prompts))
            cached = self._chosen[name] = (chosen["id"], learner)
        return cached[1]

    def _as_learner(self, candidate: str | Mechanism) -> Mechanism:
        if isinstance(candidate, Mechanism):
            return candidate
        if candidate in (*SEEDS, SEED_CONTROL):
            return SourceLearner.seed(candidate, self.prompts)  # chapter 7's form, so designs compete alike
        return resolve_mechanism(candidate, self.prompts)

    def _tournament_reason(self, name: str) -> str | None:
        """Why learning `name` should start with a learner tournament now, or None."""
        if self.selection is None:
            return None
        refused = 0
        for attempt in reversed(self.store.learning(name)):
            if attempt["mechanism"].startswith("tournament:"):
                break  # a tournament starts the count again
            if attempt["outcome"].startswith("skipped"):
                continue
            if attempt["outcome"] != "refused":
                break
            refused += 1
        if refused >= self.selection.after_refusals:
            return f"{refused} learned versions in a row failed the gate"
        lasted = self.store.state(name).get("lasted")
        within = self.selection.quarantined_within
        if lasted is not None and within is not None and lasted <= within:
            return f"the last learned version was quarantined after {lasted} run{'' if lasted == 1 else 's'}"
        return None

    def _practice_for(self, name: str, practice: Any = None) -> Practice:
        practice = practice if practice is not None else self.practice
        if hasattr(practice, "bind"):  # AgentPractice or ReplayPractice: through the agent
            return practice.bind(self, name)
        if practice is not None:
            return practice
        raise ValueError("learning needs a practice: AgentPractice(world=...) or a function (source, case_id)")

    async def _learn(self, name: str, episodes: list[Episode], events: list[str]) -> None:
        proc = self.procedures[name]
        learn_from, cases = episodes, list(self.gate.cases or [])
        if not cases and self.gate.holdout:
            accepted = [e for e in episodes if e.success]
            if len(accepted) <= self.gate.holdout:
                events.append(f"learner: {name} waits for more accepted runs ({len(accepted)} of "
                              f"{self.gate.holdout + 1})")
                return
            held = {e.task_id for e in accepted[-self.gate.holdout:]}
            learn_from = [e for e in episodes if e.task_id not in held]
            cases = list(dict.fromkeys(e.task_id for e in accepted[-self.gate.holdout:]))
        model = self.writer_model or getattr(self.agent, "model", None) or DEFAULT_MODEL
        writer = Writer(model=model, params=dict(self.writer_params))
        practice = self._practice_for(name)
        reason = self._tournament_reason(name)
        if reason is not None:
            await self._tournament(name, learn_from, cases, practice, model, reason, events)
            return
        learner = self.learner_for(name)
        mechanism = learner.name
        learned = None
        try:
            learned = await learner.learn(learn_from, practice, writer, proc.contract)
            gate_practice = practice if self.gate.practice is None else self._practice_for(name, self.gate.practice)
            gate = await self.gate.check(learned.source, gate_practice, cases)
        except Exception as exc:  # noqa: BLE001 - a failed attempt is logged; serving goes on
            # A mechanism run in helix's lab writes through the lab's own metered writer; what it spent comes
            # back on the learned memory, or on the MechanismStopped that ended it.
            calls, usage = spent(learned or exc, writer)
            dollars = writer_cost(usage, model, self.prices)
            self.store.log_learning(name=name, mechanism=mechanism, version=None, episodes=len(learn_from),
                                    writer_calls=calls, dollars=dollars, gate=None,
                                    outcome=f"failed: {type(exc).__name__}: {exc}")
            events.append(f"learner: {mechanism} failed for {name}: {type(exc).__name__}: {exc}")
            await publish(kind="learning_failed", procedure=name, detail=f"{type(exc).__name__}: {exc}")
            return
        calls, usage = spent(learned, writer)
        dollars = writer_cost(usage, model, self.prices)
        parent = (self.store.versions(name) or [{}])[-1].get("version")
        info = {"mechanism": mechanism, "writer_calls": calls, "dollars": dollars, "gate": gate.to_dict(),
                "learned_from": [e.task_id for e in learn_from], "notes": learned.notes[-20:]}
        outcome = self._decide(name, learned.source, gate, parent, info)
        self.store.log_learning(name=name, mechanism=mechanism, version=outcome[1], episodes=len(learn_from),
                                writer_calls=calls, dollars=dollars, gate=gate.to_dict(), outcome=outcome[0])
        events.append(f"learner: {mechanism} wrote {name} from {len(learn_from)} runs; gate {gate.accepted}/"
                      f"{gate.total} {outcome[0]}")
        kind = {"refused": "refused", "proposed for approval": "proposed"}.get(outcome[0], "installed")
        await publish(kind=kind, procedure=name, version=outcome[1],
                      detail=f"{mechanism}, gate {gate.accepted}/{gate.total}, ${dollars:.4f}")
        if kind == "installed" and self.programs is not None:
            await self.programs.relearned(name, events)

    async def _tournament(self, name: str, learn_from: list[Episode], cases: list[str], practice: Practice,
                          model: str, reason: str, events: list[str]) -> None:
        """Let every candidate learn `name` from the same runs, keep the winner as its learner, and take the
        winner's function as this attempt's candidate, gated on the cases it was scored on."""
        gate_practice = practice if self.gate.practice is None else self._practice_for(name, self.gate.practice)
        problem = LearningProblem(name, self.procedures[name].contract, learn_from, practice, cases, gate_practice)
        choice = await run_tournament(problem, self._tournament_candidates(), writer_model=model,
                                      writer_params=self.writer_params, prices=self.prices)
        winner = choice.winner
        self._keep_learner(name, choice, reason)
        state = self.store.state(name)
        state.pop("lasted", None)
        self.store.put_state(name, state)
        events.append(f"learner: tournament for {name} ({reason}): {choice.line()}; chose {winner.mechanism}")
        await publish(kind="learner_chosen", procedure=name, detail=f"{winner.mechanism}: {choice.line()}")
        gate = self.gate.verdict(winner.accepted, winner.total, winner.failures)
        if winner.source is None:  # every candidate failed to learn anything
            self.store.log_learning(name=name, mechanism=f"tournament:{winner.mechanism}", version=None,
                                    episodes=len(learn_from), writer_calls=choice.writer_calls,
                                    dollars=choice.dollars, gate=None, outcome="failed: no candidate learned")
            await self._design(name, problem, model, events)
            return
        parent = (self.store.versions(name) or [{}])[-1].get("version")
        info = {"mechanism": winner.mechanism, "writer_calls": choice.writer_calls, "dollars": choice.dollars,
                "gate": gate.to_dict(), "learned_from": [e.task_id for e in learn_from],
                "tournament": choice.to_dict(), "notes": [reason]}
        outcome = self._decide(name, winner.source, gate, parent, info)
        self.store.log_learning(name=name, mechanism=f"tournament:{winner.mechanism}", version=outcome[1],
                                episodes=len(learn_from), writer_calls=choice.writer_calls, dollars=choice.dollars,
                                gate=gate.to_dict(), outcome=outcome[0])
        events.append(f"learner: {winner.mechanism} wrote {name} from {len(learn_from)} runs; gate {gate.accepted}/"
                      f"{gate.total} {outcome[0]}")
        kind = {"refused": "refused", "proposed for approval": "proposed"}.get(outcome[0], "installed")
        await publish(kind=kind, procedure=name, version=outcome[1],
                      detail=f"{winner.mechanism} (tournament), gate {gate.accepted}/{gate.total}, "
                             f"${choice.dollars:.4f}")
        if kind == "installed" and self.programs is not None:
            await self.programs.relearned(name, events)
        if outcome[0] == "refused":  # no candidate learns this procedure well enough
            await self._design(name, problem, model, events)

    async def _design(self, name: str, problem: LearningProblem, model: str, events: list[str]) -> None:
        """No mechanism learns `name`: design one on its own learning problem, in the worker, and propose it."""
        policy = self.selection.design if self.selection is not None else None
        if policy is None or self.learning != "worker":
            return  # a design search never runs in a serving process
        from helix.metacognition.design import design_learners, register_mechanism

        tests = list(await policy.tests()) if policy.tests is not None else []
        design = await design_learners(
            [problem], tests, prompts=self.prompts, designer_model=policy.designer_model or model,
            designer_params=policy.designer_params, writer_model=model, writer_params=self.writer_params,
            budget=policy.budget, fixed_effort=policy.fixed_effort, population=policy.population,
            generations=policy.generations, rng=random.Random(self.store.count()), prices=self.prices)
        outcome = "design proposed" if design.accepted else f"design refused: {design.reason}"
        self.store.log_learning(name=name, mechanism=f"design:{design.name}", version=None,
                                episodes=len(problem.episodes), writer_calls=0, dollars=design.dollars, gate=None,
                                outcome=outcome)
        if not design.accepted:
            events.append(f"learner: no design learns {name} better: {design.reason}")
            return
        register_mechanism(self.store, design)
        events.append(f"learner: designed {design.name} for {name} ({design.reason}); it competes once a person "
                      f"approves it")
        await publish(kind="mechanism_proposed", procedure=name, detail=f"{design.name}: {design.reason}")

    def _tournament_candidates(self) -> list[Mechanism]:
        """The selection's candidates, and the designed mechanisms a person approved (helix.metacognition.design)."""
        candidates = list(self._candidates)
        if self.selection is not None and self.selection.designed:
            names = {c.name for c in candidates}
            designs = sorted((self.store.setting("mechanisms") or {}).items(), key=lambda item: design_rank(item[1]))
            for name, design in designs:  # the best standing first: one that transfers before one that ties
                if design.get("state") == "approved" and name not in names:
                    candidates.append(SourceLearner(design["source"], name, self.prompts))
        return candidates

    def _keep_learner(self, name: str, choice: LearnerChoice, reason: str) -> None:
        learner = choice.winner.learner
        source = learner.source if isinstance(learner, SourceLearner) else None
        chosen = dict(self.store.setting("learners") or {})
        chosen[name] = {"mechanism": choice.winner.mechanism, "source": source,
                        "id": learner.id if isinstance(learner, SourceLearner) else choice.winner.mechanism,
                        "reason": reason, "chosen_at": datetime.now(UTC).isoformat(), **choice.to_dict()}
        self.store.put_setting("learners", chosen)

    def _decide(self, name: str, source: str, gate: GateResult, parent: int | None, info: dict[str, Any]
                ) -> tuple[str, int]:
        if not gate.passed:
            return "refused", self.store.add_version(name, source, state="rejected", created_by=info["mechanism"],
                                                     parent=parent, info=info)
        if self.approval == "human":
            return "proposed for approval", self.store.add_version(
                name, source, state="proposed", created_by=info["mechanism"], parent=parent, info=info)
        version = self.store.add_version(name, source, state="candidate", created_by=info["mechanism"],
                                         parent=parent, info=info)
        return "passed, installed", self._trust(name, version)

    # ------------------------------------------------------------ reporting

    def summary(self) -> dict[str, Any]:
        """The study's economics for this deployment: runs and cost by reasoning setting and with or
        without a procedure, learning spend, and how many requests with a procedure pay it back."""
        from helix.metacognition.report import summarize

        return summarize(self.store)


# ------------------------------------------------------------ store-level operations (also used by the CLI)

def trust_version(store: MetaStore, name: str, version: int) -> int:
    """Make `version` the procedure's trusted version, retiring the one before, with a fresh detector window."""
    current = store.trusted(name)
    if current is not None and current["version"] != version:
        store.set_state(name, current["version"], "retired")
    store.set_state(name, version, "trusted")
    state = store.state(name)
    state.update(recent=[], window=[], counted=[], trusted_at=store.count())
    store.put_state(name, state)
    return version


def approve_version(store: MetaStore, name: str, version: int) -> int:
    found = store.version(name, version)
    if found is None or found["state"] != "proposed":
        raise ValueError(f"{name}@{version} is not awaiting approval")
    return trust_version(store, name, version)


def quarantine_procedure(store: MetaStore, name: str, reason: str = "by hand", *, relearn_after: int = 4) -> bool:
    current = store.trusted(name)
    if current is None:
        return False
    store.set_state(name, current["version"], "quarantined")
    count = store.count()
    store.put_state(name, {"recent": [], "window": [], "counted": [], "quarantined_at": count,
                           "next_attempt": count + relearn_after, "reason": reason})
    return True
