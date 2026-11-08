"""The offline toolkit: measure reasoning floors, search a procedure's code, and search learning mechanisms.

These run between deployments, off the serving path (study §3.1's offline loop):

calibrate          effort sweeps per arm (no memory; trusted memory, recalled but frozen) -> Floors,
                   whose table is the controller's lookup for the model measured (study E1)
calibrate_routing  per-procedure thresholds for the router, from requests labeled with the
                   procedure they belong to (or none)
calibrate_deployment  both, for an agent's deployment, recorded into its store (`metacog calibrate`);
                   cases come from a JSONL case file (load_cases)
search_procedure   GEPA over a procedure's source, scored by practice on recorded cases, every case
                   tried under each input variant, with the candidate's own outcome in the feedback
                   (study E2, including both of its measurement fixes)
search_learners    every learning mechanism on the same experience, scored by behavior match to a
                   reference function, kept only on strict improvement (study E3)
"""

from __future__ import annotations

import inspect
import json
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from helix.archive import SQLiteArchive
from helix.artifact import ArtifactKind, genesis
from helix.eval.effort import EffortSweep, ReasoningFloorSignal, sweep
from helix.memory.code_skills import UnsafeSkill, check_source
from helix.memory.learning_mechanisms import Episode, Writer
from helix.search.base import SearchBudget
from helix.search.candidates import CandidateSearch
from helix.search.methods import build_search
from helix.signal import GapMeasurement
from helix.signals.behavior_match import BehaviorMatchSignal

from helix.metacognition.exceptions import AgentsException
from helix.metacognition.ladders import bare
from helix.metacognition.mechanisms import MECHANISMS, contract_text, resolve_mechanism, writer_cost
from helix.metacognition.parts import ADEQUATE, EFFORTS, SOLID, Controller, Practice, Router, Verdict
from helix.metacognition.procedure import Procedure
from helix.metacognition.prompts import GENERIC, PromptPack
from helix.metacognition.model_settings import DEFAULT_MODEL, ModelSettings, Reasoning


@dataclass(frozen=True)
class Case:
    """One request to measure on: a stable id, the input, and what rebuilds its world.

    `procedure` labels the request's class for routing calibration: a procedure's name, or None for a
    request in no procedure's class. `labeled` says whether the case carries a label at all.
    """

    id: str
    input: str
    data: Any = None
    procedure: str | None = None
    labeled: bool = False


def load_cases(path: str | Path) -> list[Case]:
    """Cases from a JSONL file, one object a line: {"id", "input", "data"?, "procedure"?}.

    "data" is what the world factory rebuilds the case from. A "procedure" key (a procedure's name, or
    null for a request in no procedure's class) labels the case for routing calibration.
    """
    cases: list[Case] = []
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError as exc:
            raise ValueError(f"{path}, line {n}: not JSON ({exc})") from exc
        if not isinstance(row, dict) or "id" not in row or "input" not in row:
            raise ValueError(f'{path}, line {n}: a case needs "id" and "input"')
        cases.append(Case(str(row["id"]), str(row["input"]), row.get("data"), row.get("procedure"),
                          "procedure" in row))
    seen: set[str] = set()
    for case in cases:
        if case.id in seen:
            raise ValueError(f"{path}: case id {case.id!r} appears more than once")
        seen.add(case.id)
    return cases


def save_cases(path: str | Path, cases: Sequence[Case]) -> Path:
    """Write cases as a JSONL case file that `load_cases` reads back."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for case in cases:
        row: dict[str, Any] = {"id": case.id, "input": case.input}
        if case.data is not None:
            row["data"] = case.data
        if case.labeled:
            row["procedure"] = case.procedure
        lines.append(json.dumps(row, default=str))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return path


# ------------------------------------------------------------------ calibrate

@dataclass
class Floors:
    """Effort sweeps per arm on one model, and the floors and controller table they imply."""

    sweeps: dict[str, EffortSweep] = field(default_factory=dict)
    model: str | None = None

    def floor(self, arm: str, threshold: float = SOLID) -> str | None:
        return self.sweeps[arm].floor(threshold)

    def setting(self, arm: str, threshold: float = ADEQUATE) -> str:
        """The cheapest setting that clears the bar, or, when none does, the one with the best record
        (spending less would only fail more often), as chapter 6's Controller.from_floors chooses."""
        sweep_ = self.sweeps[arm]
        measured = [e for e in EFFORTS if e in sweep_.cells]
        return sweep_.floor(threshold) or max(measured, key=lambda e: (sweep_.cells[e].rate, EFFORTS.index(e)))

    def table(self, threshold: float = ADEQUATE) -> dict[str, str]:
        return {arm: self.setting(arm, threshold) for arm in ("memory", "no_memory") if arm in self.sweeps}

    def controller(self, threshold: float = ADEQUATE) -> Controller:
        """A controller that uses these floors for the model they were measured on (any model, if unknown)."""
        if self.model is None:
            return Controller(table=self.table(threshold))
        return Controller(per_model={bare(self.model): {"table": self.table(threshold)}})

    def signal(self, arm: str, *, target: str = "low", threshold: float = SOLID) -> GapMeasurement:
        return ReasoningFloorSignal(target=target, threshold=threshold).measure_sweep(self.sweeps[arm])

    def rows(self) -> list[dict[str, Any]]:
        return [row for sweep_ in self.sweeps.values() for row in sweep_.rows()]

    def to_dict(self, threshold: float = ADEQUATE) -> dict[str, Any]:
        return {**self.table(threshold), "threshold": threshold, "model": self.model,
                "sweeps": [s.to_dict() for s in self.sweeps.values()]}

    def save(self, path: str | Path, threshold: float = ADEQUATE) -> Path:
        """Write the floors; `Controller.from_floors(path)` reads the table back."""
        path = Path(path)
        if path.is_dir() or not path.suffix:
            path = path / "floors.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(threshold), indent=2), encoding="utf-8")
        return path

    def record(self, store: Any, *, procedure: str | None = None, threshold: float = ADEQUATE) -> None:
        """Write these floors into a metacognition store (a MetaStore or a Metacognition), where the
        controller reads them for this model. `procedure` scopes them to one procedure."""
        from helix.metacognition.store import ANY_PROCEDURE

        if self.model is None:
            raise ValueError("these floors name no model; calibrate records the model it measured")
        store = getattr(store, "store", store)
        cases = max((cell.total for sweep_ in self.sweeps.values() for cell in sweep_.cells.values()), default=0)
        store.put_floors(self.model, self.table(threshold), procedure=procedure or ANY_PROCEDURE, bar=threshold,
                         cases=cases)

    @classmethod
    def load(cls, path: str | Path) -> Floors:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls({s["arm"]: EffortSweep.from_dict(s) for s in data["sweeps"]}, data.get("model"))


async def calibrate(agent: Any, cases: Sequence[Case], *, verifier: Callable[..., Any],
                    world: Callable[[Any], Any] | None = None, efforts: Sequence[str] = EFFORTS,
                    arms: Sequence[str] = ("no_memory", "memory"), repeats: int = 1, concurrency: int = 4,
                    max_turns: int = 10, run_config: Any = None,
                    prices: dict[str, tuple[float, ...]] | None = None) -> Floors:
    """Run every case at every setting in each arm and collect the floors.

    The "memory" arm recalls the agent's trusted procedures with metacognition
    frozen (nothing recorded, no controller); "no_memory" runs without them.
    `world(case.data)` builds each run's context; `verifier(context, result)`
    judges it. Repeats run each case more than once, live, to see variance.
    """
    from helix.metacognition.pricing import PRICES
    from helix.metacognition.run import RunConfig, Runner

    base = run_config or RunConfig()
    tasks = [(case, r) for case in cases for r in range(repeats)]
    floors = Floors(model=base.model or getattr(agent, "model", None) or DEFAULT_MODEL)
    for arm in arms:
        if arm == "memory" and getattr(agent, "metacognition", None) is None:
            raise ValueError("the memory arm needs an agent with metacognition and a trusted procedure")
        mode = "frozen" if arm == "memory" else "off"

        async def run(task: tuple[Case, int], effort: str, mode: str = mode):
            case, _ = task
            settings = (base.model_settings or ModelSettings()).resolve(ModelSettings(reasoning=Reasoning(effort)))
            config = replace(base, model_settings=settings, metacognition_mode=mode, case_id=case.id,
                             case_data=case.data)
            context = world(case.data if case.data is not None else case.id) if world else case.data
            try:
                result = await Runner.run(agent, case.input, context=context, max_turns=max_turns, run_config=config)
            except AgentsException as exc:
                return False, f"the run failed: {exc}", getattr(exc, "trajectory", None)
            judged = verifier(result.context_wrapper, result)
            verdict = Verdict.coerce(await judged if inspect.isawaitable(judged) else judged)
            return bool(verdict and verdict.accepted), verdict.reason if verdict else "no verdict", result.trajectory

        def task_id(task: tuple[Case, int]) -> str:
            return task[0].id if repeats == 1 else f"{task[0].id}#{task[1]}"

        floors.sweeps[arm] = await sweep(arm, tasks, run, efforts=efforts, task_id=task_id, concurrency=concurrency,
                                         prices=prices or PRICES)
    return floors


# ------------------------------------------------------------------ calibrate_routing

@dataclass
class RoutingCalibration:
    """A router with calibrated thresholds, and how well it separated the labeled requests."""

    router: Router
    procedures: dict[str, dict[str, Any]]
    accuracy: float  # labeled requests routed to their own procedure (or to none), scored leave-one-out


def calibrate_routing(procedures: Any, labeled: Sequence[tuple[str, str | None]], *, k: int = 2,
                      examples: bool = True) -> RoutingCalibration:
    """Set the router's threshold for each procedure from requests labeled with the procedure they belong to.

    `procedures` is a Metacognition or a list of procedures. `labeled` pairs each request with its
    procedure's name, or None for a request in no procedure's class. With `examples`, the labeled requests
    also become each procedure's example requests, and every request is scored without itself (leave one
    out), so thresholds are set on the scores the router would really give a new request.

    Each threshold sits halfway between two observed scores, where balanced accuracy (the share of the
    procedure's requests matched, averaged with the share of other requests not matched) is highest. Ties go
    to the higher threshold, because a request offered a procedure outside its class runs at that procedure's
    low setting. A procedure without labeled requests both in and out of its class keeps the default.
    """
    procs = list(procedures.procedures.values()) if hasattr(procedures, "procedures") else list(procedures)
    descriptions = {p.name: p.description for p in procs}
    unknown = sorted({label for _, label in labeled if label is not None and label not in descriptions})
    if unknown:
        raise ValueError(f"labels name no procedure: {unknown}")
    router = Router(k=k, examples={name: [r for r, label in labeled if label == name] for name in descriptions}
                    if examples else {})
    rows = [(router.scores(request, descriptions, leave_out=request if examples else None), label)
            for request, label in labeled]
    report: dict[str, dict[str, Any]] = {}
    for name in descriptions:
        inside = [scores[name] for scores, label in rows if label == name]
        outside = [scores[name] for scores, label in rows if label != name]
        if not inside or not outside:
            report[name] = {"threshold": router.default, "inside": len(inside), "outside": len(outside),
                            "note": "needs labeled requests both in and out of its class; kept the default"}
            continue
        values = sorted(set(inside + outside))
        cuts = [values[0] - 1e-9, *((a + b) / 2 for a, b in zip(values, values[1:], strict=False)), values[-1] + 1e-9]

        def balanced(t: float, inside: list[float] = inside, outside: list[float] = outside) -> float:
            return (sum(s >= t for s in inside) / len(inside) + sum(s < t for s in outside) / len(outside)) / 2

        best = max(cuts, key=lambda t: (balanced(t), t))
        router.thresholds[name] = best
        report[name] = {"threshold": round(best, 4), "inside": len(inside), "outside": len(outside),
                        "matched_inside": sum(s >= best for s in inside),
                        "matched_outside": sum(s >= best for s in outside)}
    correct = 0
    for scores, label in rows:
        ranked = sorted(scores.items(), key=lambda pair: -pair[1])[:k]
        routed = [n for n, s in ranked if s >= router.thresholds.get(n, router.default)]
        correct += (routed[0] if routed else None) == label
    return RoutingCalibration(router, report, correct / len(rows) if rows else 0.0)


@dataclass
class Calibration:
    """What `calibrate_deployment` measured and recorded."""

    floors: Floors
    routing: RoutingCalibration | None
    skipped: list[str] = field(default_factory=list)  # arms not measured, and why


async def calibrate_deployment(agent: Any, cases: Sequence[Case], *, world: Callable[[Any], Any] | None = None,
                               efforts: Sequence[str] = EFFORTS, arms: Sequence[str] = ("no_memory", "memory"),
                               repeats: int = 1, concurrency: int = 4, max_turns: int = 10, run_config: Any = None,
                               prices: dict[str, tuple[float, ...]] | None = None,
                               threshold: float = ADEQUATE) -> Calibration:
    """Calibrate an agent's deployment in place, as `metacog calibrate` does.

    Measures the reasoning floors on `cases` with the deployment's own verifier, records them in its store
    for the agent's model (the controller reads them from there), and, when cases are labeled, calibrates
    the router and saves it with the store. The world factory defaults to an AgentPractice's. The memory
    arm is skipped while no procedure is trusted, because it would measure nothing a procedure does.
    """
    meta = getattr(agent, "metacognition", None)
    if meta is None or not hasattr(meta, "store"):
        raise ValueError("calibrate_deployment needs an agent with Metacognition (one store)")
    if meta.verifier is None:
        raise ValueError("calibration needs verdicts: give the Metacognition a verifier")
    world = world or getattr(meta.practice, "world", None)
    skipped = []
    if "memory" in arms and not any(meta.store.trusted(name) for name in meta.procedures):
        arms = [arm for arm in arms if arm != "memory"]
        skipped.append("memory: no procedure is trusted yet")
    floors = await calibrate(agent, cases, verifier=meta.verifier, world=world, efforts=efforts, arms=arms,
                             repeats=repeats, concurrency=concurrency, max_turns=max_turns, run_config=run_config,
                             prices=prices)
    floors.record(meta, threshold=threshold)
    routing = None
    labeled = [(case.input, case.procedure) for case in cases if case.labeled]
    if labeled:
        routing = calibrate_routing(meta, labeled, k=meta.router.k)
        if routing.router.thresholds:
            meta.store.put_setting("router", routing.router.to_dict())
            meta.router = routing.router
        else:
            skipped.append("routing: every procedure needs labeled requests both in and out of its class")
    return Calibration(floors, routing, skipped)


# ------------------------------------------------------------------ search_procedure

MUTATION_PROMPT = """You are improving a Python function by reflective mutation in a
population-based search. An assistant calls the function instead of working
the answer out itself. You will see the current function, feedback from
running it on recorded cases (which outcomes were rejected and why, and what
the function observed and returned on one of them), and sometimes a sibling
function from the same population.

Produce an improved function that fixes what the feedback shows, keeps what
already works, and keeps this contract exactly:

{contract}

Output only the function source."""

CROSSOVER_PROMPT = """You are combining two Python functions into one offspring. An assistant
calls the function instead of working the answer out itself. Take what each
parent does well, drop what either does poorly, and keep this contract
exactly:

{contract}

Output only the function source."""


def practice_evaluator(practice: Practice, cases: Sequence[str], *, variants: Sequence[Practice] = (),
                       partial: Callable[[str, str], Awaitable[float]] | None = None, shown: int = 6
                       ) -> Callable[..., Awaitable[GapMeasurement]]:
    """An evaluator for searching a procedure's code, built from practice.

    A case counts as accepted only when it is accepted under `practice` and
    under every variant (the same case with its inputs written another way),
    the fix for the study's attempt B, where a search overfit the evaluator's
    calling convention. Feedback names up to `shown` rejected cases with their
    reasons and, for the first, what practice observed, which should include
    the call and the function's output: attempt C stalled without it. With
    `partial`, 0.8 of the score is acceptance and 0.2 the mean partial credit.
    """
    tries = [practice, *variants]

    async def evaluate(candidate: Any, questions: Any = None) -> GapMeasurement:
        source = candidate.content if hasattr(candidate, "content") else candidate
        accepted, rejected = 0, []
        for case_id in cases:
            for index, attempt in enumerate(tries):
                reward, reason, observed = await attempt(source, case_id)
                if reward < 1.0:
                    label = "" if index == 0 else f" (input variant {index})"
                    rejected.append((case_id, f"{reason}{label}", observed))
                    break
            else:
                accepted += 1
        total = len(cases)
        score = accepted / total if total else 0.0
        if partial is not None and total:
            credit = sum([await partial(source, case_id) for case_id in cases]) / total
            score = 0.8 * score + 0.2 * credit
        if rejected:
            feedback = (f"{accepted} of {total} cases accepted. Rejected:\n"
                        + "\n".join(f"- {case_id}: {reason}" for case_id, reason, _ in rejected[:shown])
                        + f"\n\nWhat the function observed and returned on {rejected[0][0]}:\n{rejected[0][2]}")
        else:
            feedback = f"all {total} cases accepted"
        return GapMeasurement(score=score, signal_id="practice_acceptance", signal_version=1,
                              triggered=accepted < total, raw_value=float(accepted), feedback=feedback,
                              metadata={"accepted": accepted, "total": total})

    return evaluate


@dataclass
class ProcedureSearch:
    source: str
    version: int
    seed_score: float
    lineage: list[dict[str, Any]]


async def search_procedure(procedure: Procedure, *, seed_source: str | None = None,
                           evaluator: Callable[..., Awaitable[GapMeasurement]] | None = None,
                           practice: Practice | None = None, cases: Sequence[str] = (),
                           variants: Sequence[Practice] = (), proposer_model: str | None = None,
                           proposer_params: dict[str, Any] | None = None, population: int = 4, generations: int = 4,
                           rng: Any = None, prompts: PromptPack = GENERIC, mutation_prompt: str | None = None,
                           crossover_prompt: str | None = None, archive: str | Path = ":memory:",
                           method: str = "gepa") -> ProcedureSearch:
    """Search a procedure's source with GEPA (reflective mutation and crossover over a population).

    Give either `evaluator` or `practice` with `cases` (and `variants`). The
    seed defaults to the procedure's hand-written body. Pass a seeded
    `random.Random` as `rng`: GEPA's choices then replay (study §7).
    """
    seed_text = seed_source or procedure.reference
    if seed_text is None:
        raise ValueError("search_procedure needs a seed: seed_source=, or a procedure with a body")
    if evaluator is None:
        if practice is None or not cases:
            raise ValueError("search_procedure needs evaluator=, or practice= with cases=")
        evaluator = practice_evaluator(practice, cases, variants=variants)
    contract = contract_text(procedure.contract, prompts)

    def validate(source: str) -> str | None:
        try:
            check_source(source, procedure.name)
        except UnsafeSkill as exc:
            return str(exc)
        return None

    search = build_search(method, proposer_model=proposer_model or DEFAULT_MODEL, evaluator=evaluator,
                          mutation_prompt=mutation_prompt or MUTATION_PROMPT.format(contract=contract),
                          crossover_prompt=crossover_prompt or CROSSOVER_PROMPT.format(contract=contract),
                          validate=validate, population=population, generations=generations,
                          proposer_params=dict(proposer_params or {"reasoning_effort": "medium"}), rng=rng,
                          distinct_proposals=False)
    store = SQLiteArchive(archive)
    seed = genesis(f"memory.{procedure.name}", ArtifactKind.CODE, seed_text, created_by="seed")
    seed_score = (await evaluator(seed)).score or 0.0
    budget = SearchBudget(max_candidates=population * generations)
    winners = [v async for v in search.propose(seed, None, store, budget)]
    lineage = []
    for variant in sorted(await store.all_artifacts_with_measurements(), key=lambda v: v.artifact.version):
        if variant.measurement is None:
            continue
        art = variant.artifact
        lineage.append({"version": art.version, "parent": art.parent_id[1] if art.parent_id else None,
                        "created_by": art.created_by, "score": variant.measurement.score or 0.0,
                        "metadata": dict(variant.measurement.metadata)})
    store.close()
    kept = winners[0].artifact if winners else seed
    return ProcedureSearch(kept.content, kept.version, seed_score, lineage)


# ------------------------------------------------------------------ search_learners

@dataclass
class LearnerTrial:
    mechanism: str
    matched: int
    cases: int
    writer_calls: int
    dollars: float
    source: str
    notes: list[str] = field(default_factory=list)


@dataclass
class LearnerSearch:
    control: LearnerTrial | None
    trials: list[LearnerTrial]
    kept: str
    test: tuple[int, int] | None = None

    @property
    def kept_source(self) -> str:
        return next(t.source for t in self.trials if t.mechanism == self.kept)


async def search_learners(procedure: Procedure, episodes: list[Episode], *, practice: Practice, reference: str,
                          run_case: Callable[[str, Any], Awaitable[Any]], cases: Sequence[Any],
                          test_cases: Sequence[Any] | None = None, mechanisms: Sequence[str] | None = None,
                          prompts: PromptPack = GENERIC, writer_model: str | None = None,
                          writer_params: dict[str, Any] | None = None, case_id: Callable[[Any], str] = str,
                          include_control: bool = True, archive: str | Path = ":memory:",
                          prices: dict[str, tuple[float, ...]] | None = None) -> LearnerSearch:
    """Run each mechanism on the same experience and keep the one whose function best matches the reference.

    `run_case(source, case)` runs a function on one case and returns its
    output; a mechanism's function matches when every case's output equals the
    reference's. The search proposes mechanisms in order and replaces the
    incumbent only on a strictly higher match, so ties go to the first.
    """
    signal = BehaviorMatchSignal(reference, run_case, case_id=case_id)
    writer_model = writer_model or DEFAULT_MODEL
    params = dict(writer_params or {"reasoning_effort": "medium"})
    trials: dict[str, LearnerTrial] = {}

    async def trial(name: str) -> tuple[LearnerTrial, GapMeasurement]:
        writer = Writer(model=writer_model, params=dict(params))
        learned = await resolve_mechanism(name, prompts).learn(episodes, practice, writer, procedure.contract)
        measurement = await signal.measure(learned.source, ground_truth=list(cases))
        return LearnerTrial(name, int(measurement.metadata.get("matched", 0)), len(cases), learned.writer_calls,
                            writer_cost(writer.usage, writer_model, prices), learned.source,
                            list(learned.notes)), measurement

    control = (await trial("no_experience"))[0] if include_control else None
    names = list(mechanisms or MECHANISMS)
    config = {"writer": {"model": writer_model, "params": params}}

    async def evaluate(artifact: Any) -> tuple[GapMeasurement, dict[str, Any]]:
        result, measurement = await trial(artifact.content["mechanism"])
        trials[result.mechanism] = result
        return measurement, {"writer_calls": result.writer_calls, "notes": result.notes}

    store = SQLiteArchive(archive)
    seed = genesis(f"learning_mechanism.{procedure.name}", ArtifactKind.CODE, {"mechanism": names[0], **config})
    search = CandidateSearch([{"mechanism": name, **config} for name in names], evaluate)
    variants = [v async for v in search.propose(seed, signal, store, None)]
    winner = await search.select(variants, signal, store)
    store.close()
    kept = winner.content["mechanism"]
    test = None
    if test_cases:
        measured = await signal.measure(trials[kept].source, ground_truth=list(test_cases))
        test = (int(measured.metadata.get("matched", 0)), len(test_cases))
    return LearnerSearch(control, [trials[name] for name in names if name in trials], kept, test)
