"""The Artifact primitive.

An Artifact is any mutable object under search: prompts, skills, tool
descriptions, memory entries, rubrics, planner code, monitor scaffolds, and
(at the frontier) agent code itself. Spec §1.

Three disciplines are enforced here:

1. Immutability. Mutations produce new versions; existing versions are never
   modified. This is what makes archive search, lineage analysis, and rollback
   tractable.
2. Lineage. Every artifact carries a parent pointer. Genesis artifacts have
   parent_id=None. Human edits use created_by="human".
3. Content addressing. The content hash is part of the storage key. Two
   artifacts with identical content collapse to the same row.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class ArtifactKind(str, Enum):
    PROMPT = "prompt"
    SKILL = "skill"
    TOOL_DESCRIPTION = "tool_description"
    MEMORY_ENTRY = "memory_entry"
    RUBRIC = "rubric"
    PLANNER = "planner"
    MONITOR = "monitor"
    CODE = "code"


ParentRef = tuple[str, int] | None


@dataclass(frozen=True)
class Artifact:
    """An immutable, versioned, content-addressed configuration object.

    Frozen because mutations must go through `mutate()`, which produces a new
    version with a parent pointer. Hand-editing an Artifact in place is a bug
    the type system catches.
    """

    id: str
    version: int
    kind: ArtifactKind
    content: str | dict[str, Any]
    parent_id: ParentRef = None
    created_by: str = "human"
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def content_hash(self) -> str:
        """SHA-256 of the canonical serialization of content.

        Strings hash directly. Dicts hash as sorted-key JSON so logically
        identical structures collapse regardless of key order.
        """
        if isinstance(self.content, str):
            payload = self.content.encode("utf-8")
        else:
            payload = json.dumps(self.content, sort_keys=True).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    @property
    def ref(self) -> tuple[str, int]:
        """The (id, version) tuple used as a parent pointer elsewhere."""
        return (self.id, self.version)

    def mutate(
        self,
        new_content: str | dict[str, Any],
        created_by: str,
        metadata: dict[str, Any] | None = None,
        version: int | None = None,
    ) -> Artifact:
        """Produce a new version of this artifact.

        The new version's parent_id points at the current (id, version).
        `version` may be supplied explicitly to avoid sibling collisions when
        multiple Searches mutate the same parent: in that case the caller
        looks up `archive.next_version(self.id)` first. When omitted, the
        version defaults to self.version + 1, which is correct only when no
        siblings exist. SPEC section 1.1.
        """
        return Artifact(
            id=self.id,
            version=version if version is not None else self.version + 1,
            kind=self.kind,
            content=new_content,
            parent_id=self.ref,
            created_by=created_by,
            metadata=metadata or {},
        )


def genesis(
    id: str,
    kind: ArtifactKind,
    content: str | dict[str, Any],
    created_by: str = "human",
    metadata: dict[str, Any] | None = None,
) -> Artifact:
    """Create the first (genesis) version of an artifact.

    Genesis artifacts have parent_id=None and version=1. Most artifacts in a
    fresh project start here.
    """
    return Artifact(
        id=id,
        version=1,
        kind=kind,
        content=content,
        parent_id=None,
        created_by=created_by,
        metadata=metadata or {},
    )
