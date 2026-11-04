"""Compatibility adapter for memory learning through the Search protocol.

MemoryLearner owns the fixed value update. This adapter retains the existing
MemRLSearch API, archive tags, and propose/select events for older callers.
The runtime memory application mechanism remains in MemoryHooks and the tier.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from helix.artifact import Artifact, ArtifactKind
from helix.memory.archive_bridge import entry_utility, latest_artifact
from helix.memory.base import MemoryEntry
from helix.memory.learning import Editor, MemoryLearner, SEARCH_METHOD, ieu
from helix.observability.events import MutationProposed, SearchCompleted, SearchStarted
from helix.search.base import SearchBudget, SearchCostModel, SearchKind, Variant
from helix.signal import Signal

if TYPE_CHECKING:
    from helix.archive import Archive


class MemRLSearch(MemoryLearner):
    """Expose the memory learner through the existing Search interface.

    select() chooses an archived variant for Search callers. Agent recall
    still uses EpisodicMemory.read(), with its configured ranking formula.
    """

    # ---------------- Search protocol ----------------

    @property
    def kind(self) -> SearchKind:
        return SearchKind.MEMORY_Q_LEARNING

    @property
    def cost_model(self) -> SearchCostModel:
        # One update per run, one utility attribution per update, no proposer
        # model unless an editor is wired in.
        return SearchCostModel(proposes_per_round=1, rounds_typical=1, signal_calls_per_round=1)

    async def propose(
        self,
        seed: Artifact,
        signal: Signal,
        archive: Archive,
        budget: SearchBudget,
        context: dict[str, Any] | None = None,
    ) -> AsyncIterator[Variant]:
        """Yield the operator mutations MemRL produced.

        With `context={"trajectory": ..., "reward": ...}` the search runs an
        update first and yields the versions it minted; without a context it
        replays the last update's variants. `seed` is advisory: when it names a
        memory artifact, only that entry's variants are yielded.
        """
        await self.bus.publish(
            SearchStarted(
                search_kind=self.kind.value,
                seed_artifact_id=seed.id,
                seed_version=seed.version,
                budget={"max_candidates": budget.max_candidates},
            )
        )
        variants: list[Variant] = []
        try:
            if context and context.get("trajectory") is not None:
                await self.update(
                    context["trajectory"],
                    reward=context.get("reward"),
                    outcome_measurement=context.get("outcome_measurement"),
                )
            for variant in self._last_variants:
                if seed.kind == ArtifactKind.MEMORY_ENTRY and variant.artifact.id != seed.id:
                    continue
                if budget.exhausted():
                    break
                await self.bus.publish(
                    MutationProposed(
                        search_kind=self.kind.value,
                        parent_artifact_id=variant.parent.id,
                        parent_version=variant.parent.version,
                        candidate_version=variant.artifact.version,
                        proposer_model="memrl",
                        rationale=(
                            f"{variant.metadata.get('operator')}: q "
                            f"{variant.metadata.get('q_before', 0.0):.2f} -> "
                            f"{variant.metadata.get('q_after', 0.0):.2f} "
                            f"(credit {variant.metadata.get('credit', 0.0):.2f})"
                        ),
                        candidate_excerpt=str(ieu(variant.artifact).get("experience") or "")[:200],
                    )
                )
                budget.saw_candidate()
                variants.append(variant)
                yield variant
        finally:
            winner = await self._pick(variants) if variants else None
            await self.bus.publish(
                SearchCompleted(
                    search_kind=self.kind.value,
                    winner_artifact_id=winner.id if winner else seed.id,
                    winner_version=winner.version if winner else seed.version,
                    candidates_evaluated=len(variants),
                )
            )

    async def select(
        self,
        candidates: list[Variant],
        signal: Signal,
        archive: Archive,
    ) -> Artifact:
        """The highest-utility variant; ties go to the newest version."""
        if candidates:
            return await self._pick(candidates)
        best = await self.best_entries(k=1)
        if best:
            return best[0].artifact
        raise ValueError("MemRLSearch.select: no candidates and no measured memory entries")

    async def _pick(self, candidates: list[Variant]) -> Artifact:
        def key(v: Variant) -> tuple[float, str, int]:
            return (entry_utility(v.artifact), v.artifact.created_at.isoformat(), v.artifact.version)

        return max(candidates, key=key).artifact

    async def rerank(self, entries: list[MemoryEntry]) -> list[MemoryEntry]:
        """Order supplied entries by utility for callers of the legacy helper.

        Chapter 5's harness does not call this helper. Its fixed application
        path is MemoryHooks.inject() followed by the tier's read() ranking.
        """
        async def utility_of(entry: MemoryEntry) -> float:
            if entry.artifact_ref is not None:
                art = await latest_artifact(self.archive, entry.artifact_ref[0])
                if art is not None:
                    return entry_utility(art)
            return float(entry.score)

        keyed = [(await utility_of(e), i, e) for i, e in enumerate(entries)]
        keyed.sort(key=lambda t: (-t[0], t[1]))
        return [e for _, _, e in keyed]
