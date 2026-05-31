"""SPO: Self-Supervised Prompt Optimization.

Pairwise mutation. Generate a variant, judge it against the current best via a
pairwise Signal, accept on win. Pairwise self-supervision escapes APE's
labeled-data wall: no ground-truth labels are required because the judge
produces preference signal directly. SPEC section 4.2.

SPO works with any pairwise Signal (SignalKind.LLM_JUDGE_PAIRWISE or
SignalKind.CONTRASTIVE). The position-swap-and-agree wrapper from
helix.signals.pairwise_judge.SwapAndAgree composes cleanly here.

The caller is responsible for filling each Variant's measurement before the
next iteration: the typical pattern is `async for variant in search.propose(...):
    variant.measurement = await judge(variant, current_best)`, then yielding
back control to SPO which reads measurement on the next iteration.
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
from helix.signal import Cost, Preference, Signal

if TYPE_CHECKING:
    from helix.archive import Archive


DEFAULT_MUTATION_PROMPT = """You are a prompt engineer producing the next iteration of a system prompt.

You will see the current best prompt and (optionally) feedback from the most
recent pairwise judge. The judge compared a previous candidate against the
current best and explained which won and why.

Propose a single improved candidate. Make changes that address what the
feedback identified as the loser's weakness. If the previous candidate won,
extend its winning move. If it lost, change strategy.

Output only the new prompt. No preamble, no commentary, no markdown."""


class SPO:
    """Self-Supervised Prompt Optimization via pairwise mutation."""

    def __init__(
        self,
        proposer_model: str = "gpt-4o",
        proposer_prompt: str = DEFAULT_MUTATION_PROMPT,
        rounds: int = 5,
        first_round_model: str | None = None,
        bus: EventBus | None = None,
    ) -> None:
        self.proposer_model = proposer_model
        # The first round has no feedback to incorporate; a cheaper model
        # usually suffices for the initial "make this prompt clearer" pass.
        # Falls back to proposer_model when None.
        self.first_round_model = first_round_model or proposer_model
        self.proposer_prompt = proposer_prompt
        self.rounds = rounds
        self.bus = bus or get_bus()

    @property
    def kind(self) -> SearchKind:
        return SearchKind.PAIRWISE

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
        # Seed last_feedback and last_outcome from the archive. When the
        # Improver calls propose() round after round, this lets SPO see the
        # most recent judge verdict on a sibling of the current best, so
        # mutation can act on that feedback. SPEC's "search consumes signal"
        # composition: the archive is the carrier.
        last_feedback, last_outcome = await self._read_recent_feedback(archive, seed)

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

                model_for_this_round = self.first_round_model if round_idx == 0 else self.proposer_model
                new_content, gen_cost = await self._mutate(
                    current_best, last_outcome, last_feedback, model=model_for_this_round
                )
                budget.charge(gen_cost)
                total_cost_tokens += gen_cost.tokens
                # Ask the archive for the next unused version number for this
                # artifact id so siblings don't collide. Other Searches may have
                # produced earlier mutations of the same parent.
                next_v = await archive.next_version(current_best.id) if hasattr(archive, "next_version") else current_best.version + 1
                candidate_artifact = current_best.mutate(
                    new_content=new_content,
                    created_by=f"spo_round_{round_idx}",
                    metadata={
                        "round": round_idx,
                        "proposer_model": model_for_this_round,
                        "parent_was_best": True,
                    },
                    version=next_v,
                )
                await self.bus.publish(
                    MutationProposed(
                        search_kind=self.kind.value,
                        parent_artifact_id=current_best.id,
                        parent_version=current_best.version,
                        candidate_version=candidate_artifact.version,
                        proposer_model=model_for_this_round,
                        rationale=(last_feedback or "(first round; no prior feedback)")[:200],
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
                    last_feedback = variant.measurement.feedback
                    # Preference.LEFT means the candidate (LEFT slot) won; promote it.
                    if variant.measurement.preference == Preference.LEFT:
                        current_best = variant.artifact
                        winner_id = current_best.id
                        winner_version = current_best.version
                        last_outcome = "candidate_won"
                    elif variant.measurement.preference == Preference.RIGHT:
                        last_outcome = "candidate_lost"
                    else:
                        last_outcome = "tie"
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
        wins = [c for c in candidates if c.measurement and c.measurement.preference == Preference.LEFT]
        if wins:
            wins.sort(key=lambda v: v.measurement.confidence, reverse=True)
            return wins[0].artifact
        best = await archive.best(k=1)
        if best:
            return best[0].artifact
        if candidates:
            return candidates[0].artifact
        raise ValueError("SPO.select: no winning candidate, no candidates, and empty archive")

    async def _read_recent_feedback(
        self,
        archive: Archive,
        seed: Artifact,
    ) -> tuple[str | None, str | None]:
        """Look at the most recent measurement of a sibling/descendant of the
        seed; return (feedback, outcome_label) for SPO's mutation prompt.

        This is how SPO carries state across Improver rounds without holding
        Python state. The archive is the single source of truth.
        """
        try:
            descendants = await archive.descendants(seed) if hasattr(archive, "descendants") else []
        except Exception:
            descendants = []
        # Find the descendant with the most recent measurement.
        candidates_with_measure = []
        for d in descendants:
            try:
                variant = await archive.by_id(d.id, d.version)  # type: ignore[attr-defined]
            except Exception:
                continue
            if variant is None or variant.measurement is None:
                continue
            candidates_with_measure.append(variant)
        if not candidates_with_measure:
            return None, None
        # Sort by recorded_at (proxy: higher version number among same-parent).
        candidates_with_measure.sort(key=lambda v: v.artifact.version, reverse=True)
        latest = candidates_with_measure[0]
        feedback = latest.measurement.feedback if latest.measurement else None
        pref = latest.measurement.preference if latest.measurement else None
        if pref is not None and pref.value == "left":
            outcome = "candidate_won"
        elif pref is not None and pref.value == "right":
            outcome = "candidate_lost"
        else:
            outcome = "tie"
        return feedback, outcome

    async def _mutate(
        self,
        current: Artifact,
        last_outcome: str | None,
        last_feedback: str | None,
        model: str | None = None,
    ) -> tuple[str, Cost]:
        if not isinstance(current.content, str):
            raise TypeError("SPO v1 only supports text-content artifacts")
        use_model = model or self.proposer_model
        user_msg = f"""Current best prompt:
---
{current.content}
---

Outcome of last round: {last_outcome or 'this is the first round'}
Judge feedback from last round: {last_feedback or 'none'}

Propose a single improved candidate."""

        response = await helix_acompletion(
            model=use_model,
            messages=[
                {"role": "system", "content": self.proposer_prompt},
                {"role": "user", "content": user_msg},
            ],
        )
        text = (response.choices[0].message.content or "").strip()
        usage = getattr(response, "usage", None)
        cost = Cost(tokens=getattr(usage, "total_tokens", 0) if usage else 0)
        return text, cost
