"""ComposedSearch: joint search over a composite's constituents. SPEC §18.4.

A composite (SPEC §18.2) binds several coupled artifacts under one identity. A
ComposedSearch improves the bundle by coordinate descent: each round it picks
one constituent, runs that constituent's own child Search to propose a variant,
holds the others at their current version, and yields a new composite version
with that one constituent swapped. The whole bundle is then measured by one
composite Signal, and the winner is promoted atomically as a new composite.

This is the cheap default of §18.4: it reuses the per-kind child searches almost
unchanged, differing only in that measurement is whole-bundle, selection is on
the composite signal, and promotion is atomic over constituents. Strongly-
coupled clusters can swap in a bespoke Search registered against the composite
shape; this class is the generic fallback.

The search keeps the current champion artifact per constituent role in memory so
propose can seed each child search and select can advance the accepted role.
Each proposed constituent is also put in the archive, and the variant names it by
ref, so recording works and an improver can expand the composite into the exact
versions it binds (§18.6).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from helix.artifact import Artifact
from helix.search.base import (
    Search,
    SearchBudget,
    SearchCostModel,
    SearchKind,
    Variant,
)
from helix.signal import Signal


class ComposedSearch:
    """Coordinate-descent Search over a composite's constituents. SPEC §18.4.

    Constructed with a child Search per constituent role and the current
    constituent artifacts. Each `propose` mutates one role (round-robin), and
    `select` advances the accepted role's champion.
    """

    def __init__(
        self,
        child_searches: dict[str, Search],
        constituents: dict[str, Artifact],
        *,
        label: str = "composed",
    ) -> None:
        # Roles present in both maps are improvable; a role without a child
        # search is held fixed (it still participates in measurement).
        self._child_searches = dict(child_searches)
        self._current = dict(constituents)
        self._improvable_roles = [r for r in constituents if r in child_searches]
        self._label = label
        self._next = 0
        # Proposed constituents by ref; variant metadata holds only the ref,
        # because the archive records metadata as JSON.
        self._proposed: dict[tuple[str, int], Artifact] = {}

    @property
    def kind(self) -> SearchKind:
        return SearchKind.COMPOSED

    @property
    def cost_model(self) -> SearchCostModel:
        # One child propose plus one whole-bundle signal call per round.
        return SearchCostModel(proposes_per_round=1, signal_calls_per_round=1)

    def current_constituent(self, role: str) -> Artifact | None:
        """The champion artifact currently bound for a constituent role."""
        return self._current.get(role)

    async def propose(
        self,
        seed: Artifact,
        signal: Signal,
        archive: Any,
        budget: SearchBudget,
    ) -> AsyncIterator[Variant]:
        if not self._improvable_roles:
            return
        # Round-robin pick of the constituent to improve this round.
        role = self._improvable_roles[self._next % len(self._improvable_roles)]
        self._next += 1
        child = self._child_searches[role]
        constituent = self._current[role]

        async for child_variant in child.propose(
            seed=constituent, signal=signal, archive=archive, budget=budget
        ):
            new_constituent = child_variant.artifact
            new_content = self._rebuild_content(seed, role, new_constituent)
            new_composite = seed.mutate(
                new_content, created_by=f"{self._label}:{role}"
            )
            self._proposed[new_constituent.ref] = new_constituent
            if archive is not None and hasattr(archive, "put_artifact"):
                await archive.put_artifact(new_constituent)
            yield Variant(
                artifact=new_composite,
                parent=seed,
                search_method=f"{self.kind.value}:{role}",
                metadata={
                    "changed_role": role,
                    "constituent_ref": list(new_constituent.ref),
                },
            )
            return  # one constituent variant per round (coordinate descent)

    def constituents_for(self, variant: Variant) -> list[Artifact]:
        """The constituents a proposed composite binds that only this search
        holds so far, for expanding it before evaluation (§18.6)."""
        ref = variant.metadata.get("constituent_ref")
        found = self._proposed.get(tuple(ref)) if ref else None
        return [found] if found is not None else []

    async def select(
        self,
        candidates: list[Variant],
        signal: Signal,
        archive: Any,
    ) -> Artifact:
        if not candidates:
            raise ValueError("ComposedSearch.select: no candidates")
        scored = [
            c for c in candidates
            if c.measurement is not None and c.measurement.score is not None
        ]
        winner = (
            max(scored, key=lambda c: c.measurement.score)
            if scored
            else candidates[0]
        )
        # Advance the accepted role's champion so the next round seeds from it.
        role = winner.metadata.get("changed_role")
        accepted = self.constituents_for(winner)
        if role is not None and accepted:
            self._current[role] = accepted[0]
        return winner.artifact

    def _rebuild_content(
        self, composite: Artifact, role: str, new_constituent: Artifact
    ) -> dict[str, Any]:
        """Rebuild a composite's content with one constituent swapped."""
        replacement = {
            "id": new_constituent.id,
            "version": new_constituent.version,
            "kind": new_constituent.kind.value,
            "subtype": new_constituent.subtype.value if new_constituent.subtype else None,
            "role": role,
        }
        items = [
            replacement if c.get("role") == role else dict(c)
            for c in composite.constituents
        ]
        binding = (
            composite.content.get("binding", {})
            if isinstance(composite.content, dict)
            else {}
        )
        return {"constituents": items, "binding": binding}
