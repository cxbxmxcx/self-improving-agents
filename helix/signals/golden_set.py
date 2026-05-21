"""GoldenSetCalibrator: anchor live signals against ground truth.

Periodically runs the candidate artifact against a small fixed eval set with
known reference answers. Returns the candidate's pass rate as a
GapMeasurement.

This is the drift anchor of Chapter 8 in concrete form. When LiveTrajectoryJudge
or ImplicitFeedbackSignal start drifting (rubric decay; user-satisfaction
hacking), the golden set's pass rate is the truth check that catches it.

Caching: the calibrator caches the candidate's score against the golden set
for `cache_ttl_sec`. Repeated measure() calls within the TTL return the
cached score (no LLM cost). When TTL expires or the artifact's (id, version)
changes, the calibrator re-runs.

In a CompositeSignal, weight the calibrator lower than the live judge (so
it doesn't dominate every round) but not zero (so it still anchors trends).
A common ratio is live=0.5, implicit=0.3, golden=0.2.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from helix.artifact import Artifact
from helix.eval.dataset import EvalQuestion, EvalSet
from helix.signal import (
    Cost,
    GapMeasurement,
    Preference,
    Signal,
    SignalKind,
)
from helix.trajectory import Trajectory


@dataclass
class _CacheEntry:
    score: float
    n_questions: int
    n_correct: int
    detail: list[dict[str, Any]]
    cost: Cost
    measured_at: float


AgentFactory = Callable[[Artifact], Awaitable[Any]]
ScoreFn = Callable[[str, str, EvalQuestion], Awaitable[bool]]


async def _default_score_fn(
    answer: str | None,
    reference: str,
    question: EvalQuestion,
) -> bool:
    """Default scorer: case-insensitive substring containment of reference.

    Crude but deterministic. Production callers should replace this with a
    proper LLM-as-judge or string-similarity scorer. The signature is async
    so callers CAN use an LLM judge if they want.
    """
    if not answer:
        return False
    ref = (reference or "").strip().lower()
    ans = answer.strip().lower()
    if not ref:
        return False
    # Match on the reference's most distinctive content. If reference is
    # short, require substring match. If long, require >=50% of words
    # appear in the answer.
    ref_words = [w for w in ref.split() if len(w) > 3]
    if len(ref_words) < 8:
        return ref in ans
    matched = sum(1 for w in ref_words if w in ans)
    return matched / len(ref_words) >= 0.5


class GoldenSetCalibrator:
    """Signal that scores a candidate against a fixed golden set.

    Construction:
        calibrator = GoldenSetCalibrator(
            golden_set=load_eval_set("golden.json"),
            agent_factory=lambda art: build_helpdesk_agent(system_prompt=art),
            score_fn=my_judge_fn,         # optional; defaults to substring match
            cache_ttl_sec=3600,            # re-run hourly per artifact version
            max_questions=20,              # cap to limit cost per check
        )

    The measure() call:
      1. Looks up cached score for (candidate.id, candidate.version)
      2. If fresh, returns cached GapMeasurement
      3. If stale or missing, builds an agent from the candidate, runs it
         against the golden set, scores each answer, caches and returns
    """

    def __init__(
        self,
        golden_set: EvalSet,
        agent_factory: AgentFactory,
        score_fn: ScoreFn | None = None,
        cache_ttl_sec: float = 3600.0,
        max_questions: int = 20,
    ) -> None:
        if len(golden_set) == 0:
            raise ValueError("GoldenSetCalibrator requires a non-empty golden_set")
        self.golden_set = golden_set
        self.agent_factory = agent_factory
        self.score_fn = score_fn or _default_score_fn
        self.cache_ttl_sec = cache_ttl_sec
        self.max_questions = max_questions
        self._cache: dict[tuple[str, int], _CacheEntry] = {}

    @property
    def kind(self) -> SignalKind:
        return SignalKind.GROUND_TRUTH

    @property
    def cost_estimate(self) -> Cost:
        # N agent runs (each many tokens) + N score evaluations.
        per_q = 4000
        return Cost(
            tokens=per_q * min(self.max_questions, len(self.golden_set)),
            dollars=0.0,
            wall_clock_sec=10.0 * min(self.max_questions, len(self.golden_set)),
        )

    async def measure(
        self,
        candidate: Artifact,
        trajectory: Trajectory | None = None,  # ignored
        reference: Artifact | None = None,      # ignored
        ground_truth: Any | None = None,        # ignored
    ) -> GapMeasurement:
        cache_key = (candidate.id, candidate.version)
        now = time.monotonic()

        # Cache hit
        cached = self._cache.get(cache_key)
        if cached is not None and (now - cached.measured_at) < self.cache_ttl_sec:
            return GapMeasurement(
                score=cached.score,
                preference=Preference.NONE,
                feedback=(
                    f"golden-set: {cached.n_correct}/{cached.n_questions} correct (cached)"
                ),
                confidence=1.0,
                cost=Cost(),  # cached = no fresh cost
                metadata={
                    "signal_kind": "golden_set",
                    "cached": True,
                    "n_questions": cached.n_questions,
                    "n_correct": cached.n_correct,
                    "detail": cached.detail,
                },
            )

        # Cache miss: run the candidate against the golden set
        questions = list(self.golden_set.questions)
        if self.max_questions and self.max_questions < len(questions):
            questions = self.golden_set.sample(n=self.max_questions, stratified=True, seed=42)

        total_tokens = 0
        detail: list[dict[str, Any]] = []
        n_correct = 0

        for q in questions:
            try:
                agent = await self.agent_factory(candidate)
                answer, sub_traj = await agent.run(q.question)
            except Exception as exc:
                detail.append({
                    "id": q.id,
                    "correct": False,
                    "error": f"{type(exc).__name__}: {exc}",
                })
                continue
            try:
                ok = bool(await self.score_fn(answer, q.reference_answer, q))
            except Exception:
                ok = False
            if ok:
                n_correct += 1
            detail.append({
                "id": q.id,
                "correct": ok,
                "answer_excerpt": (answer or "")[:200],
            })
            # Add up token usage from each step's payload, if present
            for step in sub_traj.steps:
                payload = step.payload or {}
                usage = payload.get("usage") or {}
                if isinstance(usage, dict):
                    total_tokens += int(usage.get("total_tokens", 0) or 0)

        score = n_correct / len(questions) if questions else 0.0
        cost = Cost(tokens=total_tokens)
        entry = _CacheEntry(
            score=score,
            n_questions=len(questions),
            n_correct=n_correct,
            detail=detail,
            cost=cost,
            measured_at=now,
        )
        self._cache[cache_key] = entry

        return GapMeasurement(
            score=score,
            preference=Preference.NONE,
            feedback=(
                f"golden-set: {n_correct}/{len(questions)} correct on the calibration set"
            ),
            confidence=1.0,
            cost=cost,
            metadata={
                "signal_kind": "golden_set",
                "cached": False,
                "n_questions": len(questions),
                "n_correct": n_correct,
                "detail": detail,
            },
        )

    def clear_cache(self) -> None:
        """Drop all cached golden-set scores. Use after the golden set changes."""
        self._cache.clear()

    def status(self) -> dict[str, Any]:
        """Diagnostic snapshot of the calibrator's cache."""
        return {
            "golden_set_size": len(self.golden_set),
            "max_questions": self.max_questions,
            "cache_entries": len(self._cache),
            "cache_keys": [f"{aid}:v{v}" for aid, v in self._cache.keys()],
        }
