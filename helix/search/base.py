"""Search protocol, budget, and shared types.

A Search proposes variants of an artifact and selects among them, guided by a
Signal. The protocol absorbs hill-climbing, pairwise mutation (SPO), genetic-
Pareto (GEPA), Bayesian (MIPROv2), MCTS (AFlow), archive evolution, and
Q-learning over memory. SPEC section 4.

Two cardinal facts encoded here:
1. `propose` is an async iterator because search methods produce candidates
   over time. Hill-climb yields one at a time, GEPA yields a generation,
   evolutionary search yields continuously.
2. `select` resolves to the winning Artifact and is recorded back to the
   Archive with its measurement by the caller.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from helix.artifact import Artifact
from helix.signal import Cost, GapMeasurement, Signal

if TYPE_CHECKING:
    from helix.archive import Archive


class SearchKind(str, Enum):
    HILL_CLIMB = "hill_climb"
    PAIRWISE = "pairwise"  # SPO
    GENETIC_PARETO = "genetic_pareto"  # GEPA
    BAYESIAN = "bayesian"  # MIPROv2
    MCTS = "mcts"  # AFlow
    EVOLUTIONARY_ARCHIVE = "evolutionary_archive"  # DGM, AlphaEvolve, ShinkaEvolve
    MEMORY_Q_LEARNING = "memory_q_learning"  # MemRL
    EDITABLE_META = "editable_meta"  # HyperAgents
    HUMAN = "human"  # hand-authored variants, treated as a Search


@dataclass
class Variant:
    """A candidate produced by a Search, before it is judged or archived.

    Variants carry their parent pointer and the search method that produced
    them so the Archive can record full lineage when a winner is committed.
    """

    artifact: Artifact
    parent: Artifact
    search_method: str
    measurement: GapMeasurement | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchBudget:
    """Caps on a Search run. SPEC section 4.4.

    The caller checks `remaining()` between iterations and stops the search
    when any cap is exceeded. Search implementations should treat this as
    advisory: an in-flight model call should be allowed to finish.
    """

    max_tokens: int | None = None
    max_dollars: float | None = None
    max_wall_clock_sec: float | None = None
    max_candidates: int | None = None

    _spent_tokens: int = 0
    _spent_dollars: float = 0.0
    _started_at: float = field(default_factory=time.monotonic)
    _candidates_seen: int = 0

    def charge(self, cost: Cost) -> None:
        self._spent_tokens += cost.tokens
        self._spent_dollars += cost.dollars

    def saw_candidate(self) -> None:
        self._candidates_seen += 1

    def remaining(self) -> dict[str, float | int | None]:
        wall = time.monotonic() - self._started_at
        return {
            "tokens": None if self.max_tokens is None else max(0, self.max_tokens - self._spent_tokens),
            "dollars": None if self.max_dollars is None else max(0.0, self.max_dollars - self._spent_dollars),
            "wall_clock_sec": None if self.max_wall_clock_sec is None else max(0.0, self.max_wall_clock_sec - wall),
            "candidates": None if self.max_candidates is None else max(0, self.max_candidates - self._candidates_seen),
        }

    def exhausted(self) -> bool:
        r = self.remaining()
        return any(v is not None and v <= 0 for v in r.values())


@dataclass
class SearchCostModel:
    """A Search's declared cost shape. Lets the planner compose budgets."""

    proposes_per_round: int = 1
    rounds_typical: int = 10
    signal_calls_per_round: int = 1

    def estimated_total(self, signal_cost: Cost) -> Cost:
        per_round = signal_cost.tokens * self.signal_calls_per_round * self.proposes_per_round
        return Cost(
            tokens=per_round * self.rounds_typical,
            dollars=signal_cost.dollars * self.signal_calls_per_round * self.proposes_per_round * self.rounds_typical,
            wall_clock_sec=signal_cost.wall_clock_sec * self.signal_calls_per_round * self.proposes_per_round * self.rounds_typical,
        )


@runtime_checkable
class Search(Protocol):
    """The Search protocol. SPEC section 4.1."""

    @property
    def kind(self) -> SearchKind: ...

    @property
    def cost_model(self) -> SearchCostModel: ...

    def propose(
        self,
        seed: Artifact,
        signal: Signal,
        archive: Archive,
        budget: SearchBudget,
    ) -> AsyncIterator[Variant]: ...

    async def select(
        self,
        candidates: list[Variant],
        signal: Signal,
        archive: Archive,
    ) -> Artifact: ...
