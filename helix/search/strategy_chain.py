"""StrategyChain: ordered fallback across multiple Search methods.

A StrategyChain wraps a list of Search instances and runs them in order.
The active strategy keeps proposing until it accumulates more than
`max_failures_per_strategy` consecutive failures (non-promoted rounds), at
which point the chain rotates to the next strategy. Promotions reset the
active strategy's failure count to zero.

When all strategies are exhausted, propose() yields nothing; the caller (the
Improver) sees no candidate and aborts the round. SPEC's search-by-signal
grid: this is one execution policy over a list of grid cells.

Pedagogical motivation: cheap search methods (Hill-Climb, SPO) try first;
expensive ones (GEPA, evolutionary archive) escalate only when the cheap
methods stop producing wins. The user controls how much cheap search to
spend before paying for expensive search.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from helix.artifact import Artifact
from helix.observability.bus import EventBus, get_bus
from helix.observability.events import SearchStrategySwitched
from helix.search.base import (
    Search,
    SearchBudget,
    SearchCostModel,
    SearchKind,
    Variant,
)
from helix.signal import Signal

if TYPE_CHECKING:
    from helix.archive import Archive
    from helix.improvement.round import RoundResult


class StrategyChain:
    """A Search composed of an ordered list of Searches with failure rotation.

    Failure accounting:
      - Each strategy starts with failures = 0.
      - A non-promoted round increments the active strategy's failure count.
      - A promoted round resets the active strategy's failure count to 0.
      - When failures > max_failures_per_strategy, the chain rotates to the
        next strategy and emits SearchStrategySwitched.
      - When all strategies are retired, propose() yields nothing; rounds
        fail gracefully.
    """

    def __init__(
        self,
        strategies: list[Search],
        max_failures_per_strategy: int = 1,
        bus: EventBus | None = None,
    ) -> None:
        if not strategies:
            raise ValueError("StrategyChain requires at least one strategy")
        if max_failures_per_strategy < 1:
            raise ValueError(
                "max_failures_per_strategy must be >= 1; "
                "1 means 'retire after the first failure'"
            )
        self.strategies = strategies
        self.max_failures = max_failures_per_strategy
        self.bus = bus or get_bus()

        self._active_idx = 0
        self._failures: list[int] = [0] * len(strategies)
        self._retired: list[bool] = [False] * len(strategies)

    @property
    def kind(self) -> SearchKind:
        # Report the active strategy's kind so events stay informative.
        return self.active_strategy.kind if self.active_strategy is not None else SearchKind.PAIRWISE

    @property
    def cost_model(self) -> SearchCostModel:
        return self.active_strategy.cost_model if self.active_strategy is not None else SearchCostModel()

    @property
    def active_strategy(self) -> Search | None:
        """The currently active Search, or None if all are retired."""
        if self._active_idx >= len(self.strategies):
            return None
        return self.strategies[self._active_idx]

    @property
    def active_kind(self) -> str:
        s = self.active_strategy
        return s.kind.value if s is not None else "retired"

    @property
    def all_retired(self) -> bool:
        return all(self._retired)

    async def propose(
        self,
        seed: Artifact,
        signal: Signal,
        archive: Archive,
        budget: SearchBudget,
    ) -> AsyncIterator[Variant]:
        """Delegate to the active strategy. Yields nothing when all retired."""
        if self.active_strategy is None:
            return
        async for variant in self.active_strategy.propose(
            seed=seed, signal=signal, archive=archive, budget=budget
        ):
            # Re-wrap rather than mutate: the inner strategy may keep a
            # reference to its variant, and the chain tag is this wrapper's
            # concern. The archive records which strategy produced it.
            yield Variant(
                artifact=variant.artifact,
                parent=variant.parent,
                search_method=variant.search_method,
                measurement=variant.measurement,
                metadata={
                    **(variant.metadata or {}),
                    "strategy_chain_kind": self.active_strategy.kind.value,
                    "strategy_chain_idx": self._active_idx,
                },
            )

    async def select(
        self,
        candidates: list[Variant],
        signal: Signal,
        archive: Archive,
    ) -> Artifact:
        if self.active_strategy is None:
            if candidates:
                return candidates[0].artifact
            raise ValueError("StrategyChain.select: no active strategy and no candidates")
        return await self.active_strategy.select(candidates, signal, archive)

    async def on_round_result(self, result: RoundResult) -> None:
        """Called by the Improver after each round completes.

        Updates failure counts and rotates strategies when needed. This is
        the only stateful method on the StrategyChain; everything else is
        delegation to the active strategy.
        """
        if self.active_strategy is None:
            return

        if result.promoted:
            # Reset failures for the active strategy on a winning round.
            self._failures[self._active_idx] = 0
            return

        # Non-promoted round: increment failures and check whether we've
        # reached the retirement threshold. max_failures=1 means "retire
        # after the first failure"; max_failures=2 means "retire after the
        # second consecutive failure"; etc.
        self._failures[self._active_idx] += 1
        if self._failures[self._active_idx] >= self.max_failures:
            await self._rotate(reason="failures_exhausted")

    async def _rotate(self, reason: str) -> None:
        """Retire the active strategy and advance to the next live one."""
        prev_kind = self.active_kind
        failures_at_switch = self._failures[self._active_idx]
        self._retired[self._active_idx] = True
        next_idx = self._active_idx + 1
        # Skip already-retired strategies (defensive; shouldn't happen with
        # left-to-right rotation but safe in case of manual retirement).
        while next_idx < len(self.strategies) and self._retired[next_idx]:
            next_idx += 1
        self._active_idx = next_idx
        to_kind = self.active_kind if next_idx < len(self.strategies) else "all_retired"
        await self.bus.publish(
            SearchStrategySwitched(
                from_kind=prev_kind,
                to_kind=to_kind,
                reason=reason if next_idx < len(self.strategies) else "all_strategies_retired",
                failures_at_switch=failures_at_switch,
            )
        )

    # ---------------- introspection ----------------

    def status(self) -> dict:
        """Return current state of each strategy in the chain."""
        return {
            "active_idx": self._active_idx,
            "active_kind": self.active_kind,
            "all_retired": self.all_retired,
            "strategies": [
                {
                    "kind": s.kind.value,
                    "failures": self._failures[i],
                    "retired": self._retired[i],
                }
                for i, s in enumerate(self.strategies)
            ],
        }
