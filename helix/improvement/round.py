"""One round of improvement: the orchestration math.

Pure-ish function (it does I/O) that:

  1. Reads the current-best artifact from the Archive (or accepts a seed).
  2. Runs a reference pass: agent backed by current-best against the eval set.
  3. Asks the Search to propose one candidate mutation.
  4. Runs a candidate pass: agent backed by candidate against the same set.
  5. Pairwise-judges each question; aggregates verdicts.
  6. Records BOTH candidate and reference measurements to the archive.
  7. Emits structured events at every step for observability.
  8. Returns a RoundResult.

The Improver class drives this in a loop. Chapter scripts can also call it
directly when they want a one-shot round outside the Improver's schedule.
"""

from __future__ import annotations

import asyncio
import time
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

# Per-round concurrency setting flowed into _judge_pair via a contextvar so
# we don't have to thread it through every internal call signature.
_judge_pair_concurrency: ContextVar[int] = ContextVar(
    "_judge_pair_concurrency", default=5
)
# Budget the judge phase charges into. Flowed through the same way.
_judge_budget: ContextVar = ContextVar("_judge_budget", default=None)

from helix.agent import Agent
from helix.artifact import Artifact
from helix.archive import SQLiteArchive
from helix.archive.archive import Archive
from helix.eval.dataset import EvalQuestion
from helix.eval.source import EvalSource
from helix.observability.bus import EventBus, get_bus
from helix.observability.events import (
    BudgetExceeded,
    ImproverRoundCompleted,
    ImproverRoundStarted,
    JudgeQuestionCompleted,
    PairPassQuestionCompleted,
    PairPassQuestionStarted,
    SignalIdentified,
)
from helix.search.base import SearchBudget, Variant
from helix.signal import Cost, GapMeasurement, Preference, Signal
from helix.trajectory import StepKind, Trajectory


@dataclass
class RoundResult:
    """The outcome of one improvement round."""

    round_index: int
    target_artifact_id: str
    reference_version: int
    candidate_version: int
    candidate_score: float | None
    reference_score: float | None
    candidate_measurement: GapMeasurement
    reference_measurement: GapMeasurement
    promoted: bool
    per_question: list[dict[str, Any]] = field(default_factory=list)
    elapsed_sec: float = 0.0


async def _run_one_question(
    q: EvalQuestion,
    agent_factory,
    *,
    bus: EventBus,
    improver_id: str,
    role: str,
    archive_for_cache,
    cache_artifact: Artifact | None,
) -> tuple[str, str | None, Trajectory]:
    """Run a single question, with cache check and event emission. Returns
    (question_id, answer, trajectory)."""
    # Cache check first.
    if archive_for_cache is not None and cache_artifact is not None and hasattr(archive_for_cache, "get_trajectory"):
        cached = await archive_for_cache.get_trajectory(cache_artifact, q.id)
        if cached is not None:
            cached_answer, cached_traj = cached
            num_tool = sum(1 for s in cached_traj.steps if s.kind == StepKind.TOOL_CALL)
            await bus.publish(
                PairPassQuestionCompleted(
                    improver_id=improver_id,
                    role=f"{role} (cache hit)",
                    question_id=q.id,
                    band=q.band,
                    elapsed_sec=0.0,
                    num_tool_calls=num_tool,
                )
            )
            return q.id, cached_answer, cached_traj

    await bus.publish(
        PairPassQuestionStarted(
            improver_id=improver_id,
            role=role,
            question_id=q.id,
            band=q.band,
        )
    )
    agent = await _maybe_await(agent_factory())
    t0 = time.perf_counter()
    try:
        answer, trajectory = await agent.run(q.question)
    except Exception as exc:  # noqa: BLE001
        trajectory = Trajectory(task=q.question)
        trajectory.complete(None)
        answer = None
        trajectory.metadata["error"] = f"{type(exc).__name__}: {exc}"
    elapsed = time.perf_counter() - t0
    num_tool = sum(1 for s in trajectory.steps if s.kind == StepKind.TOOL_CALL)
    await bus.publish(
        PairPassQuestionCompleted(
            improver_id=improver_id,
            role=role,
            question_id=q.id,
            band=q.band,
            elapsed_sec=elapsed,
            num_tool_calls=num_tool,
        )
    )

    # Cache fresh results for next round.
    if archive_for_cache is not None and cache_artifact is not None and hasattr(archive_for_cache, "store_trajectory"):
        try:
            await archive_for_cache.store_trajectory(cache_artifact, q.id, trajectory, answer)
        except Exception:
            pass  # cache writes never break the round

    return q.id, answer, trajectory


async def _run_against_questions(
    agent_factory,
    questions: list[EvalQuestion],
    *,
    bus: EventBus,
    improver_id: str,
    role: str,
    archive_for_cache=None,
    cache_artifact: Artifact | None = None,
    max_concurrent: int = 5,
) -> dict[str, tuple[str | None, Trajectory]]:
    """Run questions concurrently with a bounded semaphore.

    max_concurrent caps parallel agent runs to stay inside provider rate
    limits. Events still emit per-question; concurrency does not break their
    pairing because each task emits its own Started/Completed pair.

    The trajectory cache (when archive_for_cache + cache_artifact are
    provided) skips agent runs entirely for cached (artifact, question) pairs.
    """
    sem = asyncio.Semaphore(max_concurrent)

    async def _gated(q: EvalQuestion) -> tuple[str, str | None, Trajectory]:
        async with sem:
            return await _run_one_question(
                q,
                agent_factory,
                bus=bus,
                improver_id=improver_id,
                role=role,
                archive_for_cache=archive_for_cache,
                cache_artifact=cache_artifact,
            )

    results = await asyncio.gather(*[_gated(q) for q in questions])
    return {qid: (ans, traj) for qid, ans, traj in results}


async def _maybe_await(value):
    """Allow agent_factory to be sync or async."""
    import inspect
    if inspect.isawaitable(value):
        return await value
    return value


def _describe_signal(signal: Signal) -> tuple[str, str, str]:
    """Walk a Signal's wrapper chain and return (innermost_class, model, wrappers).

    SwapAndAgree(PairwiseJudge(...)) returns ("PairwiseJudge", "claude-sonnet-4-6",
    "SwapAndAgree"). CompositeSignal returns its own class name + a summary of
    its component count. Bare Signal returns (class, model, "").
    """
    wrappers: list[str] = []
    current = signal
    while hasattr(current, "inner") and getattr(current, "inner", None) is not None:
        wrappers.append(type(current).__name__)
        current = current.inner
    inner_class = type(current).__name__
    model = getattr(current, "model", "n/a")
    return inner_class, model, " -> ".join(wrappers)


async def _judge_pair(
    candidate_prompt: Artifact,
    reference_prompt: Artifact,
    candidate_runs: dict[str, tuple[str | None, Trajectory]],
    reference_runs: dict[str, tuple[str | None, Trajectory]],
    questions: list[EvalQuestion],
    signal: Signal,
    *,
    bus: EventBus,
    improver_id: str,
) -> tuple[GapMeasurement, GapMeasurement, list[dict[str, Any]]]:
    """Pairwise-judge each question; return candidate aggregate + reference mirror."""
    # Identify the signal (class, model, wrapper chain) to the bus before we
    # start judging. This is what makes the '[judge]' tag pedagogically
    # meaningful: readers see which Signal primitive is actually acting.
    sig_class, sig_model, wrapper_chain = _describe_signal(signal)
    await bus.publish(
        SignalIdentified(
            improver_id=improver_id,
            signal_class=sig_class,
            signal_kind=signal.kind.value,
            model=sig_model,
            wrapper_chain=wrapper_chain,
        )
    )

    # Concurrent per-question judging with bounded semaphore. Judge calls
    # are independent across questions, so we gather them with parallelism =
    # max_concurrent (passed by caller). Results are aggregated after.
    sem = asyncio.Semaphore(_judge_pair_concurrency.get())

    judge_budget = _judge_budget.get()

    async def _judge_one(q: EvalQuestion):
        async with sem:
            cand_ans, cand_traj = candidate_runs[q.id]
            ref_ans, ref_traj = reference_runs[q.id]
            measurement = await signal.measure(
                candidate=candidate_prompt,
                trajectory=cand_traj,
                reference=reference_prompt,
                ground_truth={
                    "question": q.question,
                    "reference_answer": q.reference_answer,
                    "candidate_trajectory": cand_traj,
                    "reference_trajectory": ref_traj,
                },
            )
            # Charge the budget for this judge call.
            if judge_budget is not None:
                judge_budget.charge(measurement.cost)
            await bus.publish(
                JudgeQuestionCompleted(
                    improver_id=improver_id,
                    question_id=q.id,
                    band=q.band,
                    preference=measurement.preference.value,
                    confidence=measurement.confidence,
                )
            )
            return q, measurement

    judged = await asyncio.gather(*[_judge_one(q) for q in questions])

    per_question: list[dict[str, Any]] = []
    total_cost = Cost()
    n_left = n_right = n_tie = 0
    score_sum = 0.0
    feedback_bits: list[str] = []

    for q, measurement in judged:
        total_cost = total_cost + measurement.cost

        if measurement.preference == Preference.LEFT:
            n_left += 1
            score_sum += 1.0
        elif measurement.preference == Preference.RIGHT:
            n_right += 1
            score_sum += 0.0
        else:
            n_tie += 1
            score_sum += 0.5

        per_question.append({
            "question_id": q.id,
            "band": q.band,
            "preference": measurement.preference.value,
            "confidence": measurement.confidence,
            "feedback": (measurement.feedback or "")[:300],
        })
        if measurement.feedback:
            feedback_bits.append(f"[{q.id}] {measurement.feedback[:200]}")

    n = len(questions)
    mean_score = score_sum / n if n else 0.0
    if n_left > n_right:
        majority = Preference.LEFT
    elif n_right > n_left:
        majority = Preference.RIGHT
    else:
        majority = Preference.TIE
    win_rate = n_left / n if n else 0.0

    candidate_agg = GapMeasurement(
        score=mean_score,
        preference=majority,
        feedback="\n".join(feedback_bits[:5]),
        confidence=win_rate,
        cost=total_cost,
        metadata={
            "role": "candidate",
            "n_questions": n,
            "n_candidate_wins": n_left,
            "n_reference_wins": n_right,
            "n_ties": n_tie,
            "win_rate": win_rate,
            "per_question": per_question,
        },
    )
    ref_pref = (
        Preference.LEFT if majority == Preference.RIGHT
        else Preference.RIGHT if majority == Preference.LEFT
        else Preference.TIE
    )
    ref_win_rate = n_right / n if n else 0.0
    reference_agg = GapMeasurement(
        score=1.0 - mean_score,
        preference=ref_pref,
        feedback="(mirror of candidate measurement)",
        confidence=ref_win_rate,
        cost=Cost(),
        metadata={
            "role": "reference",
            "n_questions": n,
            "n_candidate_wins": n_left,
            "n_reference_wins": n_right,
            "n_ties": n_tie,
            "win_rate": ref_win_rate,
            "per_question": per_question,
        },
    )
    return candidate_agg, reference_agg, per_question


async def run_improvement_round(
    *,
    round_index: int,
    seed_artifact: Artifact,
    agent_factory,            # async () -> Agent, given a system_prompt artifact
    build_agent_with_prompt,  # async (prompt: Artifact) -> Agent
    search,                   # Search instance
    signal: Signal,           # pairwise Signal (typically SwapAndAgree wrapped)
    archive: Archive,
    eval_source: EvalSource,
    budget: SearchBudget,
    improver_id: str,
    promote_threshold_win_rate: float = 0.5,
    questions_per_round: int | None = None,
    max_concurrent_questions: int = 5,
    bus: EventBus | None = None,
) -> RoundResult:
    """One self-contained improvement round.

    Emits the full event sequence for visibility. Returns a RoundResult with
    the per-question verdicts, both measurements, and the promotion outcome.
    """
    bus = bus or get_bus()
    questions = await eval_source.get_questions(n=questions_per_round)
    if not questions:
        raise RuntimeError(f"eval source {eval_source.label} returned no questions")

    # Set the judge concurrency and budget for this round (read by _judge_pair).
    _judge_pair_concurrency.set(max_concurrent_questions)
    _judge_budget.set(budget)

    t_round_start = time.perf_counter()
    current_best = seed_artifact

    await bus.publish(
        ImproverRoundStarted(
            improver_id=improver_id,
            target_artifact_id=current_best.id,
            target_version=current_best.version,
            round_index=round_index,
            n_questions=len(questions),
        )
    )

    async def _check_budget(stage: str) -> bool:
        """Returns True if the budget is exhausted; publishes a BudgetExceeded event."""
        if budget.exhausted():
            rem = budget.remaining()
            # Find which axis tripped first.
            tripped_axis = "unknown"
            tripped_cap = 0.0
            tripped_spent = 0.0
            for axis, value in rem.items():
                if value is not None and value <= 0:
                    tripped_axis = axis
                    if axis == "tokens":
                        tripped_cap = float(budget.max_tokens or 0)
                        tripped_spent = float(budget._spent_tokens)
                    elif axis == "dollars":
                        tripped_cap = float(budget.max_dollars or 0)
                        tripped_spent = float(budget._spent_dollars)
                    break
            await bus.publish(
                BudgetExceeded(
                    improver_id=improver_id,
                    budget_axis=tripped_axis,
                    spent=tripped_spent,
                    cap=tripped_cap,
                )
            )
            return True
        return False

    if await _check_budget("before_reference_pass"):
        raise RuntimeError("budget exhausted before reference pass started")

    # 1) Reference pass. The reference artifact is immutable so we can serve
    #    cached trajectories from prior rounds. First round caches; later
    #    rounds use the cache, skipping all reference LLM calls.
    reference_runs = await _run_against_questions(
        agent_factory=lambda: build_agent_with_prompt(current_best),
        questions=questions,
        bus=bus,
        improver_id=improver_id,
        role="reference",
        archive_for_cache=archive,
        cache_artifact=current_best,
        max_concurrent=max_concurrent_questions,
    )

    if await _check_budget("before_propose"):
        raise RuntimeError("budget exhausted before SPO proposed a candidate")

    # 2) Propose one candidate
    candidate_variant: Variant | None = None
    async for v in search.propose(seed=current_best, signal=signal, archive=archive, budget=budget):
        candidate_variant = v
        break
    if candidate_variant is None:
        raise RuntimeError(f"search {search.kind.value} did not propose a candidate")
    candidate_prompt = candidate_variant.artifact

    if await _check_budget("before_candidate_pass"):
        raise RuntimeError("budget exhausted before candidate pass started")

    # 3) Candidate pass. Also caches its trajectories so that if this
    #    candidate becomes a future champion, its reference pass next round
    #    is free.
    candidate_runs = await _run_against_questions(
        agent_factory=lambda: build_agent_with_prompt(candidate_prompt),
        questions=questions,
        bus=bus,
        improver_id=improver_id,
        role="candidate",
        archive_for_cache=archive,
        cache_artifact=candidate_prompt,
        max_concurrent=max_concurrent_questions,
    )

    if await _check_budget("before_judge"):
        raise RuntimeError("budget exhausted before judge phase started")

    # 4) Judge
    candidate_agg, reference_agg, per_q = await _judge_pair(
        candidate_prompt=candidate_prompt,
        reference_prompt=current_best,
        candidate_runs=candidate_runs,
        reference_runs=reference_runs,
        questions=questions,
        signal=signal,
        bus=bus,
        improver_id=improver_id,
    )

    # 5) Record both measurements
    await archive.record(candidate_variant, candidate_agg)
    reference_variant = Variant(
        artifact=current_best,
        parent=current_best,
        search_method=current_best.created_by,
        metadata={"role": "reference_in_round", "round": round_index},
    )
    await archive.record(reference_variant, reference_agg)

    # 6) Decide promotion: candidate wins outright if its win rate > threshold
    # Promotion is whatever archive.best() will return next round. The
    # archive orders by latest measurement score, so the candidate is
    # promoted iff its score beats the reference's. The win-rate threshold
    # is a separate confidence signal exposed in the metadata for callers
    # who want a stricter check, but it does NOT gate promotion (because
    # archive.best() doesn't know about it).
    cand_score = candidate_agg.score or 0.0
    ref_score = reference_agg.score or 0.0
    promoted = cand_score > ref_score
    confidence_above_threshold = (
        (candidate_agg.confidence or 0.0) >= promote_threshold_win_rate
    )

    elapsed = time.perf_counter() - t_round_start
    await bus.publish(
        ImproverRoundCompleted(
            improver_id=improver_id,
            target_artifact_id=current_best.id,
            candidate_version=candidate_prompt.version,
            candidate_score=candidate_agg.score,
            reference_score=reference_agg.score,
            n_candidate_wins=candidate_agg.metadata["n_candidate_wins"],
            n_reference_wins=candidate_agg.metadata["n_reference_wins"],
            n_ties=candidate_agg.metadata["n_ties"],
            promoted=promoted,
            elapsed_sec=elapsed,
        )
    )

    return RoundResult(
        round_index=round_index,
        target_artifact_id=current_best.id,
        reference_version=current_best.version,
        candidate_version=candidate_prompt.version,
        candidate_score=candidate_agg.score,
        reference_score=reference_agg.score,
        candidate_measurement=candidate_agg,
        reference_measurement=reference_agg,
        promoted=promoted,
        per_question=per_q,
        elapsed_sec=elapsed,
    )
