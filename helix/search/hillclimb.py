"""Hill-climbing Search.

The Karpathy-autoresearch ratchet: one candidate per round, accept if better,
otherwise reject. Cheap, monotonic, captured by local optima. SPEC section 4.2.

Uses any absolute Signal (one that fills the `score` field). For pairwise
signals, use SPO instead.

The mutation operator is a separate LLM call (the "proposer"): given the
current best and any feedback from the latest measurement, propose a refined
variant. Splitting proposer from judge per SPEC's design lets the book teach
"separate model roles" without restructuring later.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from helix.artifact import Artifact
from helix.llm_call import acompletion as helix_acompletion
from helix.observability.bus import EventBus, get_bus
from helix.observability.events import (
    MutationProposed,
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
from helix.signal import Cost, Signal

if TYPE_CHECKING:
    from helix.archive import Archive


DEFAULT_PROPOSER_PROMPT = """You are a prompt engineer producing the next iteration of a system prompt.

You will see the current candidate, its score, and optional feedback. Produce
a single improved version of the candidate. Keep what works; change only what
the feedback suggests. Output the new prompt directly, with no preamble, no
commentary, and no markdown fences."""


class HillClimb:
    """Single-candidate-per-round monotonic search."""

    def __init__(
        self,
        proposer_model: str = "gpt-4o",
        proposer_prompt: str = DEFAULT_PROPOSER_PROMPT,
        rounds: int = 5,
        bus: EventBus | None = None,
    ) -> None:
        self.proposer_model = proposer_model
        self.proposer_prompt = proposer_prompt
        self.rounds = rounds
        self.bus = bus or get_bus()

    @property
    def kind(self) -> SearchKind:
        return SearchKind.HILL_CLIMB

    @property
    def cost_model(self) -> SearchCostModel:
        return SearchCostModel(
            proposes_per_round=1,
            rounds_typical=self.rounds,
            signal_calls_per_round=1,
        )

    async def propose(
        self,
        seed: Artifact,
        signal: Signal,
        archive: Archive,
        budget: SearchBudget,
    ) -> AsyncIterator[Variant]:
        current_best = seed
        current_score: float | None = None
        current_feedback: str | None = None

        await self.bus.publish(
            SearchStarted(
                search_kind=self.kind.value,
                seed_artifact_id=seed.id,
                seed_version=seed.version,
                budget={"max_candidates": budget.max_candidates, "max_tokens": budget.max_tokens},
            )
        )
        candidates_yielded = 0
        total_cost_tokens = 0
        winner_id = seed.id
        winner_version = seed.version

        try:
            for round_idx in range(self.rounds):
                if budget.exhausted():
                    break

                new_content, gen_cost = await self._generate(current_best, current_score, current_feedback)
                budget.charge(gen_cost)
                total_cost_tokens += gen_cost.tokens
                next_v = await archive.next_version(current_best.id) if hasattr(archive, "next_version") else current_best.version + 1
                candidate_artifact = current_best.mutate(
                    new_content=new_content,
                    created_by=f"hillclimb_round_{round_idx}",
                    metadata={"round": round_idx, "proposer_model": self.proposer_model},
                    version=next_v,
                )
                await self.bus.publish(
                    MutationProposed(
                        search_kind=self.kind.value,
                        parent_artifact_id=current_best.id,
                        parent_version=current_best.version,
                        candidate_version=candidate_artifact.version,
                        proposer_model=self.proposer_model,
                        rationale=(current_feedback or "(first round; no prior feedback)")[:200],
                        candidate_excerpt=new_content[:200],
                    )
                )
                variant = Variant(
                    artifact=candidate_artifact,
                    parent=current_best,
                    search_method=self.kind.value,
                    metadata={"round": round_idx},
                )
                budget.saw_candidate()
                candidates_yielded += 1
                yield variant

                if variant.measurement is not None:
                    score = variant.measurement.score
                    if score is not None and (current_score is None or score > current_score):
                        current_best = variant.artifact
                        current_score = score
                        current_feedback = variant.measurement.feedback
                        winner_id = current_best.id
                        winner_version = current_best.version
        finally:
            await self.bus.publish(
                SearchCompleted(
                    search_kind=self.kind.value,
                    winner_artifact_id=winner_id,
                    winner_version=winner_version,
                    candidates_evaluated=candidates_yielded,
                    cost_tokens=total_cost_tokens,
                )
            )

    async def select(
        self,
        candidates: list[Variant],
        signal: Signal,
        archive: Archive,
    ) -> Artifact:
        scored = [c for c in candidates if c.measurement and c.measurement.score is not None]
        if not scored:
            if candidates:
                return candidates[0].artifact
            best = await archive.best(k=1)
            if best:
                return best[0].artifact
            raise ValueError("HillClimb.select: no scored candidates, no candidates, and empty archive")
        scored.sort(key=lambda v: v.measurement.score, reverse=True)
        return scored[0].artifact

    async def _generate(
        self,
        current: Artifact,
        score: float | None,
        feedback: str | None,
    ) -> tuple[str, Cost]:
        if not isinstance(current.content, str):
            raise TypeError("HillClimb v1 only supports text-content artifacts")
        user_msg = f"""Current candidate:
---
{current.content}
---

Latest score: {score if score is not None else 'unknown'}
Latest feedback: {feedback or 'none'}

Produce a single improved candidate."""

        response = await helix_acompletion(
            model=self.proposer_model,
            messages=[
                {"role": "system", "content": self.proposer_prompt},
                {"role": "user", "content": user_msg},
            ],
        )
        text = (response.choices[0].message.content or "").strip()
        usage = getattr(response, "usage", None)
        cost = Cost(tokens=getattr(usage, "total_tokens", 0) if usage else 0)
        return text, cost
