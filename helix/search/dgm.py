"""DGMSearch: archive-evolutionary search.

The Darwin Goedel Machine pattern (Zhang et al. 2025) in framework form.
Where SPO mutates the current best one candidate at a time, and GEPA
evolves a fixed-size population, DGM searches an *archive*: every variant
ever produced stays, and each round samples a seed from the whole history
with quality-diversity weighting. A round-20 mutation may fork from a
round-3 candidate that nothing has touched in a while.

Three ideas distinguish DGM (SPEC section 4.2):

  1. Archive, not population. Nothing is discarded; selection draws from
     all of history.
  2. Quality-diversity sampling. The next seed is chosen by score *and*
     by distance from recent picks, so the search keeps exploring branches
     instead of collapsing onto the latest winner.
  3. Pluggable mutator. The operator that turns a seed into a candidate is
     supplied by the caller. Blind LLM rewrite, SPO-style feedback-
     conditioned mutation, and GEPA-style reflective mutation all satisfy
     the Mutator protocol. The search shape is fixed; the mutation strategy
     varies.

The paper applies this to code (the agent edits its own implementation).
The framework reuses the same class for L1 text artifacts (Chapter 3, tool
descriptions) and L4 code artifacts (Chapters 11/12). The artifact kind and
the mutator change; the archive-evolutionary loop does not.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from helix.artifact import Artifact
from helix.llm_call import acompletion as helix_acompletion
from helix.observability.bus import EventBus, get_bus
from helix.observability.events import (
    MutationProposed,
    SearchCompleted,
    SearchStarted,
)
from helix.search.base import (
    SearchBudget,
    SearchCostModel,
    SearchKind,
    Variant,
)
from helix.signal import Cost, Signal

if TYPE_CHECKING:
    from helix.archive import Archive
    from helix.archive.archive import SamplingStrategy


@runtime_checkable
class Mutator(Protocol):
    """Turns a seed artifact into a new candidate artifact.

    Implementations may read the seed's most recent measurement (passed as
    `last_measurement`) to condition the mutation, or ignore it for a blind
    rewrite. The mutator returns (new_content, cost): the content for the
    child artifact and the LLM cost of producing it. The DGMSearch loop
    handles versioning, lineage, and archiving.
    """

    async def mutate(
        self,
        seed: Artifact,
        last_feedback: str | None,
        signal: Signal,
    ) -> tuple[str | dict, Cost]: ...


DEFAULT_DGM_MUTATION_PROMPT = """You are improving an artifact under archive-evolutionary search.

You will see one artifact sampled from the search history, along with any
feedback recorded against it. Propose a single improved variant. You are not
restricted to incremental edits: because the archive keeps every variant,
bold rewrites that explore a new direction are valuable even if they risk
scoring lower, because the archive can always fork from a better ancestor
later.

Output only the new artifact content. No preamble, no commentary, no markdown."""


class BlindLLMMutator:
    """The simplest mutator: ask an LLM to rewrite the seed's content.

    Reads the seed's feedback if present (for light conditioning) but does
    not require it. This is the default DGM mutator; swap in a feedback- or
    reflection-conditioned mutator for stronger guidance.
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        prompt: str = DEFAULT_DGM_MUTATION_PROMPT,
    ) -> None:
        self.model = model
        self.prompt = prompt

    async def mutate(
        self,
        seed: Artifact,
        last_feedback: str | None,
        signal: Signal,
    ) -> tuple[str | dict, Cost]:
        seed_text = seed.content if isinstance(seed.content, str) else str(seed.content)
        user_prompt = (
            f"Artifact sampled from the search archive:\n---\n{seed_text}\n---\n\n"
            f"Feedback on this artifact:\n{last_feedback or '(none recorded)'}\n\n"
            f"Propose a single improved variant. Output only the new content."
        )
        response = await helix_acompletion(
            model=self.model,
            messages=[
                {"role": "system", "content": self.prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        new_content = (response.choices[0].message.content or "").strip()
        usage = getattr(response, "usage", None)
        cost = Cost(tokens=getattr(usage, "total_tokens", 0) if usage else 0)
        return new_content, cost


class DGMSearch:
    """Archive-evolutionary search with a pluggable mutator. SPEC section 4.2.

    Construction:
        search = DGMSearch(
            mutator=BlindLLMMutator(model="claude-sonnet-4-6"),
            rounds=1,                  # candidates proposed per propose() call
        )

    Each round:
      1. Sample a seed from the archive with quality-diversity weighting.
         (Falls back to the propose() seed argument when the archive has
         nothing measured yet.)
      2. Call the mutator to produce candidate content.
      3. Mutate the sampled seed into a new versioned artifact and yield it.
         The caller fills the Variant's measurement; the archive records it.

    The mutator is the only artifact-shape-specific piece. Everything else —
    sampling, versioning, lineage, observability — is generic.
    """

    def __init__(
        self,
        mutator: Mutator | None = None,
        rounds: int = 1,
        sampling_strategy: "SamplingStrategy | None" = None,
        bus: EventBus | None = None,
    ) -> None:
        self.mutator = mutator or BlindLLMMutator()
        self.rounds = rounds
        self._sampling_strategy = sampling_strategy
        self.bus = bus or get_bus()

    @property
    def kind(self) -> SearchKind:
        return SearchKind.EVOLUTIONARY_ARCHIVE

    @property
    def cost_model(self) -> SearchCostModel:
        return SearchCostModel(
            proposes_per_round=1,
            rounds_typical=self.rounds,
            signal_calls_per_round=1,
        )

    async def _sample_seed(self, archive: "Archive", fallback: Artifact) -> Artifact:
        """Pick the seed for the next mutation from the archive's history.

        Uses quality-diversity sampling when the archive has measured
        variants; otherwise returns the fallback (the propose() seed). This
        is the DGM-distinctive step: the seed comes from anywhere in history,
        not just from the latest winner.
        """
        from helix.archive.archive import SamplingStrategy

        strategy = self._sampling_strategy or SamplingStrategy.QUALITY_DIVERSITY
        try:
            variant = await archive.sample(strategy)
            return variant.artifact
        except (ValueError, NotImplementedError):
            # Empty archive, or strategy unsupported: fall back to the seed.
            return fallback

    async def _read_recent_feedback(
        self, archive: "Archive", artifact: Artifact
    ) -> str | None:
        """Pull the most recent measurement feedback for the sampled seed."""
        get_hist = getattr(archive, "get_measurement_history", None)
        if get_hist is None:
            return None
        try:
            history = await get_hist(artifact.id, artifact.version)
        except Exception:
            return None
        if not history:
            return None
        return history[-1].feedback

    async def propose(
        self,
        seed: Artifact,
        signal: Signal,
        archive: "Archive",
        budget: SearchBudget,
    ) -> AsyncIterator[Variant]:
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

                # 1. Sample a seed from the archive's history.
                sampled = await self._sample_seed(archive, fallback=seed)
                last_feedback = await self._read_recent_feedback(archive, sampled)

                # 2. Mutate.
                new_content, gen_cost = await self.mutator.mutate(
                    sampled, last_feedback, signal
                )
                budget.charge(gen_cost)
                total_cost_tokens += gen_cost.tokens

                # 3. Version + lineage. The candidate's parent is the *sampled*
                #    seed, not necessarily the propose() seed — that's the
                #    archive-evolutionary point.
                next_v = (
                    await archive.next_version(sampled.id)
                    if hasattr(archive, "next_version")
                    else sampled.version + 1
                )
                candidate_artifact = sampled.mutate(
                    new_content=new_content,
                    created_by=f"dgm_round_{round_idx}",
                    metadata={
                        "round": round_idx,
                        "sampled_parent_version": sampled.version,
                        "mutator": type(self.mutator).__name__,
                    },
                    version=next_v,
                )
                await self.bus.publish(
                    MutationProposed(
                        search_kind=self.kind.value,
                        parent_artifact_id=sampled.id,
                        parent_version=sampled.version,
                        candidate_version=candidate_artifact.version,
                        proposer_model=getattr(self.mutator, "model", "unknown"),
                        rationale=(last_feedback or "(no feedback on sampled seed)")[:200],
                        candidate_excerpt=str(new_content)[:200],
                    )
                )
                variant = Variant(
                    artifact=candidate_artifact,
                    parent=sampled,
                    search_method=self.kind.value,
                    metadata={"round": round_idx, "sampled_from_version": sampled.version},
                )
                budget.saw_candidate()
                candidates_yielded += 1
                yield variant

                if variant.measurement is not None and variant.measurement.score is not None:
                    # Track the best-scoring candidate this run for the
                    # SearchCompleted event.
                    winner_id = candidate_artifact.id
                    winner_version = candidate_artifact.version
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
        archive: "Archive",
    ) -> Artifact:
        """Return the highest-scoring candidate. With an archive present, the
        archive's best() is authoritative; here we just rank what we were
        handed."""
        scored = [
            c for c in candidates
            if c.measurement is not None and c.measurement.score is not None
        ]
        if not scored:
            return candidates[0].artifact
        return max(scored, key=lambda c: c.measurement.score).artifact
