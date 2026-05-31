"""Construction-time guards shared by the improver classes.

These enforce structural rules at the earliest possible point (improver
construction) rather than leaving them to documentation. SPEC §17.3, §18.7.
"""

from __future__ import annotations

from typing import Any


def reject_composite_constituent(
    agent: Any, target_artifact_id: str, improver_id: str
) -> None:
    """Raise if the target is a constituent of a composite the agent declares.

    A composite's constituents are improved jointly by a ComposedSearch over the
    composite and deployed as a unit, so a solo improver mutating one
    constituent would fight that joint search. Declaring the composite opts its
    constituents out of solo improvement; this makes the rule structural rather
    than advisory. SPEC §18.7. Target the composite itself for joint search.
    """
    constituent_ids = getattr(agent, "composite_constituent_ids", None)
    if callable(constituent_ids) and target_artifact_id in constituent_ids():
        raise ValueError(
            f"Improver {improver_id}: cannot solo-target "
            f"'{target_artifact_id}', which is a constituent of a composite. "
            f"Constituents are improved jointly via a ComposedSearch over the "
            f"composite; target the composite instead (SPEC §18.7)."
        )
