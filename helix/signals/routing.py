"""RoutingSignal: regret of the path the router chose.

Each request can run down three paths: fast, deliberate, or a question back
to the user. When every path has been run once (the paired runs of chapter
6's evaluation), hindsight says which path was best, and regret is how much
value the chosen path left behind. A router that deliberates when the fast
path would have succeeded shows regret only when cost is priced, so the
weights decide whether over-thinking counts.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import asdict, dataclass
from typing import Any

from helix._async import resolve
from helix.artifact import Artifact
from helix.scaffold.common import META_KEY
from helix.signal import Cost, GapMeasurement, Preference, SignalKind, derive_signal_id
from helix.signals.cost import CostWeights
from helix.trajectory import Trajectory


@dataclass(frozen=True)
class BranchResult:
    """One path's result for one request: its reward, dollars, and seconds."""

    outcome: float
    dollars: float = 0.0
    seconds: float = 0.0

    def value(self, weights: CostWeights) -> float:
        return self.outcome - weights.penalty(self.dollars, self.seconds)


# The results of every path for this run's request, or None when not known.
BranchLookup = Callable[[Trajectory], "Mapping[str, BranchResult] | None | Awaitable[Any]"]


class RoutingSignal:
    """Score 1 minus the chosen path's regret. SPEC §3.3."""

    def __init__(self, branches: BranchLookup, *, weights: CostWeights | None = None,
                 tolerance: float = 1e-9, version: int = 1) -> None:
        self.branches = branches
        self.weights = weights or CostWeights()
        self.tolerance = tolerance
        self._version = version

    @property
    def kind(self) -> SignalKind:
        return SignalKind.GROUND_TRUTH

    @property
    def cost_estimate(self) -> Cost:
        return Cost(wall_clock_sec=0.001)

    @property
    def signal_id(self) -> str:
        return derive_signal_id("RoutingSignal", {"weights": asdict(self.weights), "tolerance": self.tolerance})

    @property
    def signal_version(self) -> int:
        return self._version

    def _empty(self, reason: str) -> GapMeasurement:
        return GapMeasurement(confidence=0.0, signal_id=self.signal_id, signal_version=self.signal_version,
                              feedback=f"routing: {reason}", metadata={"empty": True, "reason": reason})

    async def measure(self, candidate: Artifact, trajectory: Trajectory | None = None,
                      reference: Artifact | None = None, ground_truth: Any | None = None) -> GapMeasurement:
        if trajectory is None:
            return self._empty("no trajectory")
        route = (trajectory.metadata.get(META_KEY) or {}).get("route")
        table = await resolve(self.branches(trajectory))
        if route is None or not table or route not in table:
            return self._empty(f"no branch result for route {route!r}")
        values = {name: result.value(self.weights) for name, result in table.items()}
        best = max(values, key=lambda name: (values[name], name == route))
        regret = max(0.0, values[best] - values[route])
        triggered = regret > self.tolerance
        return GapMeasurement(
            score=max(0.0, 1.0 - regret),
            preference=Preference.NONE,
            feedback=(f"chose {route}, hindsight best {best}, regret {regret:.3f}" if triggered
                      else f"chose {route}, the best path in hindsight"),
            confidence=1.0,
            signal_id=self.signal_id,
            signal_version=self.signal_version,
            triggered=triggered,
            raw_value=regret,
            cost=self.cost_estimate,
            metadata={"trajectory_id": trajectory.id, "route": route, "best": best, "values": values,
                      "branches": {name: asdict(result) for name, result in table.items()}},
        )
