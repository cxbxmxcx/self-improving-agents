"""The bridge from a memory tier to the shared archive. SPEC §7.3, v0.3.

A memory entry is an Artifact with kind=memory_entry. The tier table stays
the live index the agent reads at request time; the archive holds every
version of every entry with its parent pointer, which is what lets MemRL and
the ExpeL operators search memory the way earlier chapters searched prompts.

The artifact's content carries the entry payload, its utility, and a
revision counter together. Utility lives in content, not metadata, because
the archive de-duplicates by content hash within an id: an UPVOTE that
changed only metadata would collapse onto its parent and leave no lineage.
The revision counter keeps an UPVOTE followed by an equal DOWNVOTE from
collapsing onto the version it restored, so every vote stays a node.
"""

from __future__ import annotations

from typing import Any

from helix.artifact import Artifact, ArtifactKind, genesis
from helix.memory.base import MemoryEntry, memory_artifact_id


def memory_artifact_content(
    tier: str,
    entry: MemoryEntry,
    utility: float | None = None,
) -> dict[str, Any]:
    """The canonical content dict for a memory-entry artifact."""
    return {
        "tier": tier,
        "scope": entry.scope_key.to_str(),
        "entry": dict(entry.content),
        "utility": float(entry.score if utility is None else utility),
        "revision": 0,
    }


def entry_payload(artifact: Artifact) -> dict[str, Any]:
    """The tier-facing payload stored inside a memory-entry artifact."""
    if isinstance(artifact.content, dict):
        return dict(artifact.content.get("entry", {}))
    return {}


def entry_utility(artifact: Artifact) -> float:
    if isinstance(artifact.content, dict):
        try:
            return float(artifact.content.get("utility", 0.0))
        except (TypeError, ValueError):
            return 0.0
    return 0.0


async def record_genesis(archive: Any, tier: str, entry: MemoryEntry) -> tuple[str, int]:
    """Record a brand-new entry as a genesis artifact (ExpeL ADD).

    Returns the canonical ref the tier row should carry. `created_by` defaults
    to the agent rather than a human because the loop, not a person, wrote it.
    """
    created_by = str(entry.metadata.get("created_by") or "agent")
    art = genesis(
        id=memory_artifact_id(tier, entry.id),
        kind=ArtifactKind.MEMORY_ENTRY,
        content=memory_artifact_content(tier, entry),
        created_by=created_by,
        metadata={
            "tier": tier,
            "scope": entry.scope_key.to_str(),
            "entry_id": entry.id,
            "operator": "add",
            **{k: v for k, v in entry.metadata.items() if k != "created_by"},
        },
    )
    result = await archive.put_artifact(art)
    return tuple(result.canonical_ref)  # type: ignore[return-value]


async def record_version(
    archive: Any,
    parent: Artifact,
    new_content: dict[str, Any],
    created_by: str,
    operator: str,
    extra_metadata: dict[str, Any] | None = None,
) -> tuple[str, int]:
    """Record a child version of an existing memory artifact (EDIT, votes).

    The version number comes from the archive so sibling mutations of the
    same parent never collide, and the parent pointer is the one lineage
    rule the book never bends (CLAUDE.md non-negotiable 1).
    """
    version = await archive.next_version(parent.id)
    if isinstance(new_content, dict):
        parent_rev = parent.content.get("revision", 0) if isinstance(parent.content, dict) else 0
        new_content = {**new_content, "revision": int(parent_rev or 0) + 1}
    child = parent.mutate(
        new_content,
        created_by=created_by,
        version=version,
        metadata={
            **dict(parent.metadata),
            "operator": operator,
            **(extra_metadata or {}),
        },
    )
    result = await archive.put_artifact(child)
    return tuple(result.canonical_ref)  # type: ignore[return-value]


async def latest_artifact(archive: Any, artifact_id: str) -> Artifact | None:
    """The newest version of a memory artifact, or None if unknown."""
    variant = await archive.by_id(artifact_id)
    return variant.artifact if variant is not None else None
