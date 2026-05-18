"""GEPA: Genetic-Pareto Reflective Mutation.

Population-based search with Pareto-front selection and reflective mutation
guided by textual feedback from a Reflection Signal. ICLR 2026 Oral.

Architecture: GEPA runs the full population x generations evolution internally.
Each generation:
  1. Mutate each parent reflectively (or crossover) to produce offspring
  2. For each offspring, build an agent and run it against the eval questions
  3. Pairwise-judge each offspring against the current best
  4. Pareto-select the next generation's parents from the archive
Yields only the winner at the end of evolution.

This is "Option Y" from the design discussion: GEPA's complexity stays
internal, so the Improver still sees one candidate per round but that
candidate is the result of a full evolutionary run.

Constructor requires:
  - proposer_model: the LLM doing reflective mutation and crossover
  - reflector: a Reflection Signal that critiques trajectories
  - agent_factory: builds an Agent given a prompt artifact (so GEPA can run
                   its candidates against the eval set)
  - eval_source: where GEPA gets its questions from

GEPA's signal (the pairwise judge) and archive arrive via propose().
"""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

import litellm

from helix.artifact import Artifact
from helix.eval.dataset import EvalQuestion
from helix.eval.source import EvalSource
from helix.observability.bus import EventBus, get_bus
from helix.observability.events import (
    MutationProposed,
    PairPassQuestionCompleted,
    PairPassQuestionStarted,
    SearchCompleted,
    SearchStarted,
)
from helix.search.base import (
    Search,
    SearchBudget,
    SearchCostModel,
    SearchKind,
    Variant,
)
from helix.signal import Cost, Preference, Signal
from helix.signals.reflection import Reflection
from helix.trajectory import StepKind, Trajectory

if TYPE_CHECKING:
    from helix.archive import Archive


DEFAULT_GEPA_MUTATION_PROMPT = """You are a prompt engineer applying reflective mutation in a population-based
search. You will see a candidate, the reflective feedback from a Reflection
Signal, and (optionally) a sibling candidate from the same population.

Produce a mutated version of the candidate that addresses what the feedback
identified as weakness. Where useful, borrow specific strengths from the
sibling (this is the crossover move). Avoid drifting away from what the
original candidate already does well.

Output only the new prompt text. No preamble or commentary."""


DEFAULT_GEPA_CROSSOVER_PROMPT = """You are combining two parent prompts into a single offspring.

You will see Parent A and Parent B. Produce one offspring that takes Parent
A's strengths and Parent B's strengths and discards what either does poorly.

Output only the new prompt text."""


class GEPA:
    """Genetic-Pareto reflective mutation with internal population evaluation.

    Parameters:
      proposer_model: LLM that produces mutations / crossovers
      reflector: Reflection Signal used to critique trajectories
      agent_factory: async (prompt) -> Agent — used to build agents for
                     each candidate's eval pass
      eval_source: EvalSource — supplies the questions GEPA scores against
      population_size: candidates per generation (default 4)
      generations: number of generations to run (default 2)
      mutation_rate: probability of reflective mutation vs crossover per
                     offspring (default 0.7)
      objectives: Pareto objectives (default ['score']; single-objective)
      max_concurrent_questions: parallelism for per-question agent runs
    """

    def __init__(
        self,
        proposer_model: str = "claude-sonnet-4-6",
        reflector: Reflection | None = None,
        agent_factory=None,
        eval_source: EvalSource | None = None,
        population_size: int = 4,
        generations: int = 2,
        mutation_rate: float = 0.7,
        objectives: list[str] | None = None,
        mutation_prompt: str = DEFAULT_GEPA_MUTATION_PROMPT,
        crossover_prompt: str = DEFAULT_GEPA_CROSSOVER_PROMPT,
        max_concurrent_questions: int = 5,
        bus: EventBus | None = None,
    ) -> None:
        if agent_factory is None:
            raise ValueError("GEPA requires an agent_factory to build agents per candidate")
        if eval_source is None:
            raise ValueError("GEPA requires an eval_source to know which questions to score against")
        self.proposer_model = proposer_model
        self.reflector = reflector or Reflection(model=proposer_model)
        self.agent_factory = agent_factory
        self.eval_source = eval_source
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.objectives = objectives or ["score"]
        self.mutation_prompt = mutation_prompt
        self.crossover_prompt = crossover_prompt
        self.max_concurrent_questions = max_concurrent_questions
        self.bus = bus or get_bus()

    @property
    def kind(self) -> SearchKind:
        return SearchKind.GENETIC_PARETO

    @property
    def cost_model(self) -> SearchCostModel:
        # Per propose() invocation: population_size * generations candidates,
        # each evaluated against N questions plus 1 reflection plus 1 judge.
        return SearchCostModel(
            proposes_per_round=self.population_size * self.generations,
            rounds_typical=1,  # one Improver round = one full GEPA evolution
            signal_calls_per_round=self.population_size * self.generations * 2,
        )

    async def propose(
        self,
        seed: Artifact,
        signal: Signal,
        archive: "Archive",
        budget: SearchBudget,
    ) -> AsyncIterator[Variant]:
        """Run a full population x generations evolution; yield only the winner.

        GEPA records every intermediate candidate to the archive with its
        measurement. After all generations complete, GEPA selects the
        highest-scoring candidate from the archive's Pareto front and yields
        it as the round's variant.
        """
        await self.bus.publish(
            SearchStarted(
                search_kind=self.kind.value,
                seed_artifact_id=seed.id,
                seed_version=seed.version,
                budget={"max_candidates": budget.max_candidates, "max_tokens": budget.max_tokens},
            )
        )
        total_tokens = 0
        candidates_evaluated = 0
        winner_variant: Variant | None = None
        # Track every candidate across every generation so we can pick the
        # all-time best at the end. Selecting only from the final generation
        # would miss earlier breakthroughs.
        all_evaluated: list[Variant] = []

        # The reference: cache this seed's trajectories so candidates judge
        # against a stable baseline.
        questions = await self.eval_source.get_questions(n=None)
        reference_runs = await self._run_against_questions(
            prompt=seed, questions=questions, role="reference"
        )

        try:
            # ---- Generation 0: seed-derived population ----
            current_generation: list[tuple[Variant, GapMeasurementHolder]] = []
            for _ in range(self.population_size):
                if budget.exhausted():
                    break
                child, mut_tokens = await self._mutate_from(seed, parent_feedback=None, sibling=None, archive=archive)
                total_tokens += mut_tokens
                measurement, cand_tokens = await self._evaluate_candidate(
                    candidate=child,
                    reference=seed,
                    questions=questions,
                    reference_runs=reference_runs,
                    signal=signal,
                )
                total_tokens += cand_tokens
                variant = Variant(
                    artifact=child,
                    parent=seed,
                    search_method=self.kind.value,
                    measurement=measurement,
                    metadata={"generation": 0, "operator": "init_mutation"},
                )
                await archive.record(variant, measurement)
                candidates_evaluated += 1
                budget.saw_candidate()
                current_generation.append((variant, _MeasurementHolder(measurement)))
                all_evaluated.append(variant)

            # ---- Subsequent generations ----
            for gen_idx in range(1, self.generations):
                if budget.exhausted():
                    break
                # Select parents from current generation by score.
                parents = sorted(
                    current_generation,
                    key=lambda vm: (vm[1].measurement.score or 0.0),
                    reverse=True,
                )[: max(2, self.population_size // 2)]

                next_generation: list[tuple[Variant, _MeasurementHolder]] = []
                for _ in range(self.population_size):
                    if budget.exhausted():
                        break
                    # Choose mutation vs crossover.
                    if random.random() < self.mutation_rate or len(parents) < 2:
                        parent_variant, parent_meas = random.choice(parents)
                        parent_art = parent_variant.artifact
                        # Reflect on parent's trajectory to feed the mutation.
                        # Use any one of the parent's per-question trajectories.
                        sibling_art = None
                        if len(parents) >= 2:
                            others = [p for p, _ in parents if p.artifact.ref != parent_variant.artifact.ref]
                            if others:
                                sibling_art = random.choice(others).artifact
                        feedback = parent_meas.measurement.feedback
                        child, mut_tokens = await self._mutate_from(
                            parent_art, parent_feedback=feedback, sibling=sibling_art, archive=archive
                        )
                        operator = "reflective_mutation"
                    else:
                        a, b = random.sample([p for p, _ in parents], 2)
                        child, mut_tokens = await self._crossover(a.artifact, b.artifact, archive=archive)
                        parent_art = a.artifact
                        operator = "crossover"
                    total_tokens += mut_tokens

                    measurement, cand_tokens = await self._evaluate_candidate(
                        candidate=child,
                        reference=seed,  # always judge against the seed (the round's incoming current_best)
                        questions=questions,
                        reference_runs=reference_runs,
                        signal=signal,
                    )
                    total_tokens += cand_tokens
                    variant = Variant(
                        artifact=child,
                        parent=parent_art,
                        search_method=self.kind.value,
                        measurement=measurement,
                        metadata={"generation": gen_idx, "operator": operator},
                    )
                    await archive.record(variant, measurement)
                    candidates_evaluated += 1
                    budget.saw_candidate()
                    next_generation.append((variant, _MeasurementHolder(measurement)))
                    all_evaluated.append(variant)

                if next_generation:
                    current_generation = next_generation

            # ---- Pick the overall winner ----
            # Best across ALL generations, not just the final one. An earlier
            # generation may have produced a stronger candidate that a later
            # mutation drifted away from.
            if all_evaluated:
                winner_variant = max(
                    all_evaluated,
                    key=lambda v: (v.measurement.score or 0.0) if v.measurement else 0.0,
                )

            if winner_variant is not None:
                yield winner_variant

        finally:
            winner_id = winner_variant.artifact.id if winner_variant else seed.id
            winner_version = winner_variant.artifact.version if winner_variant else seed.version
            await self.bus.publish(
                SearchCompleted(
                    search_kind=self.kind.value,
                    winner_artifact_id=winner_id,
                    winner_version=winner_version,
                    candidates_evaluated=candidates_evaluated,
                    cost_tokens=total_tokens,
                )
            )

    async def select(
        self,
        candidates: list[Variant],
        signal: Signal,
        archive: "Archive",
    ) -> Artifact:
        # The propose() already selected the winner; just return its artifact.
        if candidates:
            return candidates[0].artifact
        best = await archive.best(k=1)
        return best[0].artifact if best else None  # type: ignore[return-value]

    # ---------------- internal helpers ----------------

    async def _mutate_from(
        self,
        parent: Artifact,
        parent_feedback: str | None,
        sibling: Artifact | None,
        archive: "Archive",
    ) -> tuple[Artifact, int]:
        if not isinstance(parent.content, str):
            raise TypeError("GEPA v1 only supports text-content artifacts")
        sibling_section = ""
        if sibling is not None and isinstance(sibling.content, str):
            sibling_section = f"\n\nSibling candidate (consider borrowing strengths):\n---\n{sibling.content}\n---"

        user_msg = f"""Parent candidate:
---
{parent.content}
---

Reflective feedback from the most recent measurement: {parent_feedback or 'none'}
{sibling_section}

Produce a single mutated offspring."""

        response = await litellm.acompletion(
            model=self.proposer_model,
            messages=[
                {"role": "system", "content": self.mutation_prompt},
                {"role": "user", "content": user_msg},
            ],
        )
        new_content = (response.choices[0].message.content or "").strip()
        usage = getattr(response, "usage", None)
        tokens = getattr(usage, "total_tokens", 0) if usage else 0

        next_v = await archive.next_version(parent.id) if hasattr(archive, "next_version") else parent.version + 1
        child = parent.mutate(
            new_content=new_content,
            created_by=f"gepa_mutation",
            metadata={"proposer_model": self.proposer_model},
            version=next_v,
        )
        if hasattr(archive, "_store_artifact"):
            archive._store_artifact(child)
            archive._conn.commit()

        await self.bus.publish(
            MutationProposed(
                search_kind=self.kind.value,
                parent_artifact_id=parent.id,
                parent_version=parent.version,
                candidate_version=child.version,
                proposer_model=self.proposer_model,
                rationale=(parent_feedback or "(no prior feedback)")[:200],
                candidate_excerpt=new_content[:200],
            )
        )
        return child, tokens

    async def _crossover(
        self,
        a: Artifact,
        b: Artifact,
        archive: "Archive",
    ) -> tuple[Artifact, int]:
        if not (isinstance(a.content, str) and isinstance(b.content, str)):
            raise TypeError("GEPA crossover requires text-content parents")
        user_msg = f"""Parent A:
---
{a.content}
---

Parent B:
---
{b.content}
---

Produce a single offspring combining A's strengths and B's strengths."""

        response = await litellm.acompletion(
            model=self.proposer_model,
            messages=[
                {"role": "system", "content": self.crossover_prompt},
                {"role": "user", "content": user_msg},
            ],
        )
        new_content = (response.choices[0].message.content or "").strip()
        usage = getattr(response, "usage", None)
        tokens = getattr(usage, "total_tokens", 0) if usage else 0

        next_v = await archive.next_version(a.id) if hasattr(archive, "next_version") else a.version + 1
        child = a.mutate(
            new_content=new_content,
            created_by="gepa_crossover",
            metadata={"proposer_model": self.proposer_model, "other_parent_version": b.version},
            version=next_v,
        )
        if hasattr(archive, "_store_artifact"):
            archive._store_artifact(child)
            archive._conn.commit()

        await self.bus.publish(
            MutationProposed(
                search_kind=self.kind.value,
                parent_artifact_id=a.id,
                parent_version=a.version,
                candidate_version=child.version,
                proposer_model=self.proposer_model,
                rationale=f"crossover with parent v{b.version}",
                candidate_excerpt=new_content[:200],
            )
        )
        return child, tokens

    async def _run_against_questions(
        self,
        prompt: Artifact,
        questions: list[EvalQuestion],
        role: str,
    ) -> dict[str, tuple[str | None, Trajectory]]:
        """Build a fresh agent per question, run concurrently."""
        sem = asyncio.Semaphore(self.max_concurrent_questions)

        async def _one(q: EvalQuestion):
            async with sem:
                await self.bus.publish(
                    PairPassQuestionStarted(
                        improver_id="gepa-internal",
                        role=role,
                        question_id=q.id,
                        band=q.band,
                    )
                )
                agent = await _maybe_await(self.agent_factory(prompt))
                t0 = time.perf_counter()
                try:
                    answer, trajectory = await agent.run(q.question)
                except Exception:
                    trajectory = Trajectory(task=q.question)
                    trajectory.complete(None)
                    answer = None
                elapsed = time.perf_counter() - t0
                num_tool = sum(1 for s in trajectory.steps if s.kind == StepKind.TOOL_CALL)
                await self.bus.publish(
                    PairPassQuestionCompleted(
                        improver_id="gepa-internal",
                        role=role,
                        question_id=q.id,
                        band=q.band,
                        elapsed_sec=elapsed,
                        num_tool_calls=num_tool,
                    )
                )
                return q.id, answer, trajectory

        results = await asyncio.gather(*[_one(q) for q in questions])
        return {qid: (ans, traj) for qid, ans, traj in results}

    async def _evaluate_candidate(
        self,
        candidate: Artifact,
        reference: Artifact,
        questions: list[EvalQuestion],
        reference_runs: dict[str, tuple[str | None, Trajectory]],
        signal: Signal,
    ) -> tuple["GapMeasurement", int]:
        """Run the candidate against all questions, judge against reference,
        return aggregate measurement and tokens used."""
        candidate_runs = await self._run_against_questions(
            prompt=candidate, questions=questions, role="candidate"
        )

        n_left = n_right = n_tie = 0
        score_sum = 0.0
        feedback_bits: list[str] = []
        total_tokens = 0
        sem = asyncio.Semaphore(self.max_concurrent_questions)

        async def _judge_one(q: EvalQuestion):
            async with sem:
                cand_ans, cand_traj = candidate_runs[q.id]
                ref_ans, ref_traj = reference_runs[q.id]
                m = await signal.measure(
                    candidate=candidate,
                    trajectory=cand_traj,
                    reference=reference,
                    ground_truth={
                        "question": q.question,
                        "reference_answer": q.reference_answer,
                        "candidate_trajectory": cand_traj,
                        "reference_trajectory": ref_traj,
                    },
                )
                return q, m

        judged = await asyncio.gather(*[_judge_one(q) for q in questions])
        for q, m in judged:
            total_tokens += m.cost.tokens
            if m.preference == Preference.LEFT:
                n_left += 1
                score_sum += 1.0
            elif m.preference == Preference.RIGHT:
                n_right += 1
                score_sum += 0.0
            else:
                n_tie += 1
                score_sum += 0.5
            if m.feedback:
                feedback_bits.append(f"[{q.id}] {m.feedback[:200]}")

        n = len(questions)
        mean_score = score_sum / n if n else 0.0
        if n_left > n_right:
            majority = Preference.LEFT
        elif n_right > n_left:
            majority = Preference.RIGHT
        else:
            majority = Preference.TIE
        win_rate = n_left / n if n else 0.0

        from helix.signal import GapMeasurement
        aggregate = GapMeasurement(
            score=mean_score,
            preference=majority,
            feedback="\n".join(feedback_bits[:5]),
            confidence=win_rate,
            cost=Cost(tokens=total_tokens),
            metadata={
                "role": "candidate",
                "n_questions": n,
                "n_candidate_wins": n_left,
                "n_reference_wins": n_right,
                "n_ties": n_tie,
                "win_rate": win_rate,
                "search_kind": self.kind.value,
            },
        )
        return aggregate, total_tokens


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _maybe_await(value):
    import inspect
    if inspect.isawaitable(value):
        return await value
    return value


class _MeasurementHolder:
    """Tiny wrapper to carry a GapMeasurement alongside a Variant tuple."""
    def __init__(self, measurement) -> None:
        self.measurement = measurement


# Re-export GapMeasurement type alias for the helper above
from helix.signal import GapMeasurement as GapMeasurementHolder  # noqa: E402
