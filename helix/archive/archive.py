"""Archive protocol and shared types.

The Archive is the durable store of every Artifact mutation, with measurements
and lineage. SPEC section 5.

Hill-climbing barely uses the Archive (single best wins). Pairwise methods use
it as recent history. GEPA, DGM, AlphaEvolve, and ShinkaEvolve are entirely
built on it. The protocol covers all of them with one shape.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol, runtime_checkable

from helix.artifact import Artifact
from helix.search.base import Variant
from helix.signal import GapMeasurement


class SamplingStrategy(str, Enum):
    """Strategies for Archive.sample(). SPEC section 5.3."""
    RANDOM = "random"
    BEST = "best"
    WEIGHTED_BY_SCORE = "weighted_by_score"
    QUALITY_DIVERSITY = "quality_diversity"


@dataclass
class DiversityMetrics:
    """Summary statistics about how diverse the archive's variants are."""
    n_variants: int
    n_unique_content_hashes: int
    mean_pairwise_distance: float
    score_range: tuple[float | None, float | None]


@runtime_checkable
class Archive(Protocol):
    """The Archive protocol. SPEC section 5.2.

    All methods are async because production archives (Postgres-backed,
    distributed) need it. SQLite backends are sync internally but expose async
    wrappers so the protocol is uniform.
    """

    async def record(self, variant: Variant, measurement: GapMeasurement) -> None: ...

    async def best(self, k: int = 1, by: str = "score") -> list[Variant]: ...

    async def pareto_front(self, objectives: list[str]) -> list[Variant]: ...

    async def sample(self, strategy: SamplingStrategy) -> Variant: ...

    async def lineage(self, artifact: Artifact) -> list[Artifact]: ...

    async def descendants(self, artifact: Artifact) -> list[Artifact]: ...

    async def diversity_metrics(self) -> DiversityMetrics: ...

    async def by_id(self, artifact_id: str, version: int | None = None) -> Variant | None: ...

    async def next_version(self, artifact_id: str) -> int: ...

    async def all_artifacts_with_measurements(self) -> list[Variant]: ...

    async def get_measurement_history(self, artifact_id: str, version: int) -> list[GapMeasurement]: ...

    async def lineage_tree(self) -> list[dict]: ...

    async def list_question_verdicts(self) -> list[dict]: ...

    async def all_cached_trajectories(self) -> list[dict]: ...
