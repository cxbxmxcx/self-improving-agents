"""Resolve what deploying an artifact means. SPEC §18.6.

Deploying a composite deploys the exact constituent versions it binds, so a
combination that runs is always one that was measured as a unit. Expanding
a composite into those versions is how an improver's clone, or a deployment,
gets them.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from helix.artifact import Artifact, ArtifactKind


async def expand_composite(archive: Any, artifact: Artifact, known: Iterable[Artifact] = ()) -> dict[str, Artifact]:
    """The overrides that deploy `artifact`: itself and, for a composite, each
    constituent version it binds, taken from `known` or else the archive."""
    overrides = {artifact.id: artifact}
    if artifact.kind is not ArtifactKind.COMPOSITE:
        return overrides
    by_ref = {a.ref: a for a in known}
    for ref in artifact.constituent_refs:
        found = by_ref.get(ref)
        if found is None and archive is not None:
            variant = await archive.by_id(ref[0], ref[1])
            found = variant.artifact if variant is not None else None
        if found is None:
            raise LookupError(f"composite {artifact.id}:v{artifact.version} binds {ref[0]}:v{ref[1]}, "
                              "which the archive does not hold")
        overrides[found.id] = found
    return overrides
