"""CandidateSearch: a hill climb over a fixed, ordered set of candidates.

Some searches choose rather than invent: chapter 6 chooses a learning
mechanism from five defined ones, with chapter 5's RL first. The seed is
the first candidate, and `propose` yields every other candidate as a new
version of the seed, so each carries its parent and its measurement. The
caller's `evaluate` turns a candidate into a measurement, which for
chapter 6 means learning a memory and comparing its behavior on validation
cases.

`select` keeps the seed as the incumbent and replaces it only when a later
candidate scores strictly higher (validation-gated retention), so a tie
never displaces what is already there. Promotion stays with a person; the
search only reports what it would keep.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from typing import Any

from helix.artifact import Artifact
from helix.search.base import SearchBudget, SearchCostModel, SearchKind, Variant
from helix.signal import GapMeasurement, Signal

Evaluate = Callable[[Artifact], Awaitable[tuple[GapMeasurement, dict[str, Any]]]]


class CandidateSearch:
    """Propose each candidate in order; select by strict improvement. SPEC §4.1."""

    def __init__(self, candidates: Sequence[Any], evaluate: Evaluate, *, created_by: str = "search") -> None:
        self.candidates = list(candidates)
        self.evaluate = evaluate
        self.created_by = created_by

    @property
    def kind(self) -> SearchKind:
        return SearchKind.HILL_CLIMB

    @property
    def cost_model(self) -> SearchCostModel:
        return SearchCostModel()

    async def propose(self, seed: Artifact, signal: Signal | None, archive: Any, budget: SearchBudget | None
                      ) -> AsyncIterator[Variant]:
        """The seed first, measured as it stands, then each candidate as its child."""
        parent = seed
        for index, content in enumerate(self.candidates):
            if index == 0 and content == seed.content:
                artifact = seed
            else:
                version = await archive.next_version(seed.id) if archive is not None else seed.version + index
                artifact = seed.mutate(content, created_by=self.created_by, version=version)
            measurement, metadata = await self.evaluate(artifact)
            if archive is not None:
                await archive.record(Variant(artifact, parent, "candidate_search", measurement, metadata),
                                     measurement)
            yield Variant(artifact, parent, "candidate_search", measurement, metadata)

    async def select(self, candidates: list[Variant], signal: Signal | None = None, archive: Any = None) -> Artifact:
        """The incumbent after every candidate has had its chance to beat it."""
        if not candidates:
            raise ValueError("no candidates to select from")
        incumbent = candidates[0]
        for variant in candidates[1:]:
            if (variant.measurement.score or 0.0) > (incumbent.measurement.score or 0.0):
                incumbent = variant
        return incumbent.artifact
