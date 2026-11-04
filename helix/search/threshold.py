"""ThresholdSearch: move one named threshold one step at a time.

The thresholds artifact holds a few numbers on [0, 1], and the cheapest way
to tune them is coordinate search: nudge one number up or down by a step and
keep the change if the signal improves. No LLM is involved, so the search
costs nothing beyond its measurements, and every candidate is one readable
change ("deliberate 0.50 -> 0.45").

Offline, a script measures every neighbor the search yields and keeps the
best. Online, the improver takes one neighbor per trigger and shadow-tests
it, so the search remembers where it stopped. When the seed it is handed has
changed, the last step was promoted; when a full cycle passes without a
promotion, the step halves, down to `min_step`.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from helix.artifact import Artifact
from helix.scaffold.config import load_thresholds
from helix.search.base import SearchBudget, SearchCostModel, SearchKind, Variant
from helix.signal import Signal


class ThresholdSearch:
    """Deterministic coordinate steps over a thresholds artifact. SPEC §4.1."""

    def __init__(self, *, step: float = 0.1, min_step: float = 0.025, names: list[str] | None = None,
                 low: float = 0.0, high: float = 1.0) -> None:
        if not 0.0 < min_step <= step:
            raise ValueError("need 0 < min_step <= step")
        self.step = step
        self.min_step = min_step
        self.names = list(names) if names else None
        self.low = low
        self.high = high
        self._next = 0
        self._misses = 0
        self._seed_ref: tuple[str, int] | None = None

    @property
    def kind(self) -> SearchKind:
        return SearchKind.HILL_CLIMB

    @property
    def cost_model(self) -> SearchCostModel:
        return SearchCostModel(proposes_per_round=2 * len(self.names or [None]), signal_calls_per_round=1)

    def _moves(self, values: dict[str, float]) -> list[tuple[str, int]]:
        names = [n for n in (self.names or sorted(values)) if n in values]
        return [(name, direction) for name in names for direction in (1, -1)]

    async def propose(self, seed: Artifact, signal: Signal, archive: Any,
                      budget: SearchBudget) -> AsyncIterator[Variant]:
        values = load_thresholds(seed)
        if self._seed_ref is not None and seed.ref != self._seed_ref:
            self._misses = 0  # the champion moved, so the last step was kept
        self._seed_ref = seed.ref
        moves = self._moves(values)
        for _ in range(len(moves)):
            if budget.exhausted():
                return
            # A full cycle of moves since the last promotion: try smaller steps.
            if self._misses >= len(moves) and self.step > self.min_step:
                self.step = max(self.min_step, self.step / 2)
                self._misses = 0
            name, direction = moves[self._next % len(moves)]
            self._next += 1
            current = values[name]
            proposed = round(min(self.high, max(self.low, current + direction * self.step)), 6)
            if proposed == current:
                continue
            self._misses += 1
            step = self.step
            # Never at or below the seed, even when the archive has not seen it.
            version = seed.version + 1
            if hasattr(archive, "next_version"):
                version = max(version, await archive.next_version(seed.id))
            child = seed.mutate(json.dumps({**values, name: proposed}, indent=2), created_by="threshold_search",
                                metadata={"format": seed.metadata.get("format", "thresholds")}, version=version)
            if hasattr(archive, "put_artifact"):
                await archive.put_artifact(child)  # so the next version number moves on
            budget.saw_candidate()
            yield Variant(artifact=child, parent=seed, search_method=self.kind.value,
                          metadata={"name": name, "from": current, "to": proposed, "step": step})

    async def select(self, candidates: list[Variant], signal: Signal, archive: Any) -> Artifact:
        if not candidates:
            raise ValueError("ThresholdSearch.select: no candidates")
        scored = [c for c in candidates if c.measurement is not None and c.measurement.score is not None]
        if not scored:
            return candidates[0].artifact
        return max(scored, key=lambda c: c.measurement.score).artifact
