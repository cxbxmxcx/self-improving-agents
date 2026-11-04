"""The group round: N candidates, one absolute signal, advantage over the mean.

This is the framework's third round shape after the pairwise round in
round.py and the archive-evolutionary round inside DGMSearch (SPEC §4.2.1).
It is the selection mechanic of Group Relative Policy Optimization with the
gradient removed: each candidate's advantage is its score minus the group
mean, and the "policy update" is promotion of the above-average artifacts.

Two scoring paths share one arithmetic. When questions and an agent factory
are given, each candidate's agent runs the questions and the per-question
scores are averaged, as the pairwise round does. When they are absent, the
signal measures the candidate artifact directly, which is how Chapter 5
scores candidate memory insights by replay utility without an agent.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field, replace
from typing import Any

from helix.archive.archive import Archive
from helix.artifact import Artifact
from helix.eval.dataset import EvalQuestion
from helix.eval.source import EvalSource
from helix.improvement.round import _describe_signal, _run_against_questions
from helix.observability.bus import EventBus, get_bus
from helix.observability.events import (
    CandidateWins,
    ImproverRoundCompleted,
    ImproverRoundStarted,
    SignalIdentified,
)
from helix.search.base import SearchBudget, Variant
from helix.signal import Cost, GapMeasurement, Signal


@dataclass
class GroupRoundResult:
    """The outcome of one group round.

    `scores` pairs every candidate with its recorded measurement, in the
    order the candidates were scored; `selected` is the subset that survived
    the selection rule; `best` is the top scorer whether or not it beat the
    mean.
    """

    round_index: int
    group_mean: float
    group_std: float
    scores: list[tuple[Variant, GapMeasurement]]
    selected: list[Variant]
    best: Variant | None
    n_candidates: int
    cost: Cost = field(default_factory=Cost)
    elapsed_sec: float = 0.0


def group_advantages(scores: list[float]) -> tuple[float, float, list[float], list[float]]:
    """The GRPO arithmetic: (mean, population std, advantages, normalized).

    The population standard deviation is used because the group is the whole
    population we are ranking within, not a sample of something larger. The
    normalized advantage falls back to 0.0 when every candidate tied, since
    there is no spread to scale by.
    """
    if not scores:
        return 0.0, 0.0, [], []
    mean = statistics.fmean(scores)
    std = statistics.pstdev(scores) if len(scores) > 1 else 0.0
    advantages = [s - mean for s in scores]
    normalized = [a / std if std > 0 else 0.0 for a in advantages]
    return mean, std, advantages, normalized


async def _pull_candidates(
    proposer,
    seed: Artifact,
    signal: Signal,
    archive: Archive,
    budget: SearchBudget,
    n: int,
) -> list[Variant]:
    """Take exactly n variants from a Search's propose iterator, then stop."""
    out: list[Variant] = []
    if n <= 0:
        return out
    async for variant in proposer.propose(seed=seed, signal=signal, archive=archive, budget=budget):
        out.append(variant)
        if len(out) >= n:
            break
    return out


async def _score_direct(signal: Signal, variant: Variant) -> GapMeasurement:
    """Score an artifact on its own; replay-utility signals need no agent."""
    return await signal.measure(candidate=variant.artifact, trajectory=None)


async def _score_via_agent(
    signal: Signal,
    variant: Variant,
    questions: list[EvalQuestion],
    agent_factory,
    *,
    bus: EventBus,
    improver_id: str,
    role: str,
    archive: Archive,
    max_concurrent: int,
) -> GapMeasurement:
    """Run the candidate's agent over the questions and average the scores."""
    runs = await _run_against_questions(
        agent_factory=lambda: agent_factory(variant.artifact),
        questions=questions,
        bus=bus,
        improver_id=improver_id,
        role=role,
        archive_for_cache=archive,
        cache_artifact=variant.artifact,
        max_concurrent=max_concurrent,
    )
    total_cost = Cost()
    per_question: list[dict[str, Any]] = []
    score_sum = 0.0
    n_scored = 0
    for q in questions:
        _answer, traj = runs[q.id]
        m = await signal.measure(
            candidate=variant.artifact,
            trajectory=traj,
            ground_truth={"question": q.question, "reference_answer": q.reference_answer},
        )
        total_cost = total_cost + m.cost
        if m.score is not None:
            score_sum += m.score
            n_scored += 1
        per_question.append({
            "question_id": q.id,
            "band": q.band,
            "score": m.score,
            "feedback": (m.feedback or "")[:300],
        })
    mean = score_sum / n_scored if n_scored else 0.0
    return GapMeasurement(
        score=mean,
        confidence=n_scored / len(questions) if questions else 0.0,
        signal_id=signal.signal_id,
        signal_version=signal.signal_version,
        cost=total_cost,
        metadata={"n_questions": len(questions), "per_question": per_question},
    )


async def run_group_round(
    candidates: list[Variant] | None = None,
    *,
    proposer=None,
    signal: Signal,
    archive: Archive,
    eval_source: EvalSource | None = None,
    questions: list[EvalQuestion] | None = None,
    agent_factory=None,
    seed: Artifact | None = None,
    group_size: int | None = None,
    round_index: int = 0,
    bus: EventBus | None = None,
    select: str = "positive",
    k: int = 1,
    record: bool = True,
    budget: SearchBudget | None = None,
    improver_id: str = "group_round",
    mode: str = "offline",
    auto_promote: bool = False,
    max_concurrent_questions: int = 5,
) -> GroupRoundResult:
    """One group round: sample, score, cull.

    Candidates come from the list or from `proposer.propose` seeded with
    `seed`; `group_size` caps how many are pulled (it defaults to the
    proposer's `group_size`, then 4). `select` is "positive" (keep every
    candidate with advantage > 0) or "top_k" (keep the k highest scores).
    """
    if select not in ("positive", "top_k"):
        raise ValueError(f"select must be 'positive' or 'top_k', got {select!r}")
    if candidates is None and proposer is None:
        raise ValueError("run_group_round needs candidates or a proposer")
    if (questions is not None or eval_source is not None) and agent_factory is None:
        raise ValueError("scoring across questions needs an agent_factory")

    bus = bus or get_bus()
    budget = budget or SearchBudget()
    t0 = time.perf_counter()

    if candidates is None:
        if seed is None:
            raise ValueError("pulling from a proposer needs a seed artifact")
        n = group_size or getattr(proposer, "group_size", None) or 4
        candidates = await _pull_candidates(proposer, seed, signal, archive, budget, n)
    if not candidates:
        raise RuntimeError("group round has no candidates to score")

    if questions is None and eval_source is not None:
        questions = await eval_source.get_questions()
        if not questions:
            raise RuntimeError(f"eval source {eval_source.label} returned no questions")

    target = seed or candidates[0].parent
    await bus.publish(
        ImproverRoundStarted(
            improver_id=improver_id,
            target_artifact_id=target.id,
            target_version=target.version,
            round_index=round_index,
            n_questions=len(questions or []),
        )
    )
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

    # Score every candidate with the same absolute signal. A signal that
    # returns no score counts as 0.0 so the group arithmetic stays defined.
    raw: list[GapMeasurement] = []
    for i, variant in enumerate(candidates):
        if questions is not None:
            m = await _score_via_agent(
                signal, variant, questions, agent_factory,
                bus=bus, improver_id=improver_id,
                role=f"candidate {i + 1}/{len(candidates)}",
                archive=archive, max_concurrent=max_concurrent_questions,
            )
        else:
            m = await _score_direct(signal, variant)
        budget.charge(m.cost)
        raw.append(m)

    score_values = [m.score if m.score is not None else 0.0 for m in raw]
    group_mean, group_std, advantages, normalized = group_advantages(score_values)

    total_cost = Cost()
    scored: list[tuple[Variant, GapMeasurement]] = []
    for variant, m, s, adv, adv_n in zip(candidates, raw, score_values, advantages, normalized):
        enriched = replace(
            m,
            score=s,
            metadata={
                **m.metadata,
                "group_round": round_index,
                "group_mean": group_mean,
                "advantage": adv,
                "advantage_norm": adv_n,
                "group_size": len(candidates),
            },
        )
        variant.measurement = enriched
        total_cost = total_cost + enriched.cost
        if record:
            await archive.record(variant, enriched)
        scored.append((variant, enriched))

    ranked = sorted(scored, key=lambda pair: pair[1].score or 0.0, reverse=True)
    if select == "positive":
        selected = [v for v, m in ranked if m.metadata["advantage"] > 0]
    else:
        selected = [v for v, _ in ranked[: max(k, 0)]]
    best = ranked[0][0] if ranked else None
    best_score = ranked[0][1].score if ranked else None

    n_above = sum(1 for _, m in scored if m.metadata["advantage"] > 0)
    n_below = sum(1 for _, m in scored if m.metadata["advantage"] < 0)
    n_tied = len(scored) - n_above - n_below

    # The group mean is the reference here: a best that clears it is the
    # round's found improvement, and promotion stays a handler decision.
    beat_mean = best is not None and (best_score or 0.0) > group_mean
    if beat_mean:
        await bus.publish(
            CandidateWins(
                improver_id=improver_id,
                target_artifact_id=best.artifact.id,
                candidate_version=best.artifact.version,
                reference_version=best.parent.version,
                candidate_score=best_score,
                reference_score=group_mean,
                mode=mode,
                auto_promote=auto_promote,
            )
        )

    elapsed = time.perf_counter() - t0
    await bus.publish(
        ImproverRoundCompleted(
            improver_id=improver_id,
            target_artifact_id=target.id,
            candidate_version=best.artifact.version if best else 0,
            candidate_score=best_score,
            reference_score=group_mean,
            n_candidate_wins=n_above,
            n_reference_wins=n_below,
            n_ties=n_tied,
            promoted=beat_mean,
            elapsed_sec=elapsed,
        )
    )

    return GroupRoundResult(
        round_index=round_index,
        group_mean=group_mean,
        group_std=group_std,
        scores=scored,
        selected=selected,
        best=best,
        n_candidates=len(candidates),
        cost=total_cost,
        elapsed_sec=elapsed,
    )
