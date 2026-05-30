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
    """The search-economics kind of an artifact (SPEC §1.2).

    Kind answers "how is this searched," not "what role does it play." Role is
    carried by an optional Subtype, and deploy risk by a derived layer. The set
    is the v0.2 consolidation of the earlier per-role kinds.
    """

    TEXT = "text"
    MEMORY_ENTRY = "memory_entry"
    CODE = "code"
    COMPOSITE = "composite"


class Subtype(str, Enum):
    """The role of an artifact within its kind (SPEC §1.2, §18.2.1).

    The text subtypes are a closed set; the composite subtypes are open, and a
    composite subtype names a blessed composition shape (metacognition first).
    """

    # text subtypes (closed set)
    PROMPT = "prompt"
    TOOL_DESCRIPTION = "tool_description"
    SKILL = "skill"
    RUBRIC = "rubric"
    PLANNER = "planner"
    MONITOR = "monitor"
    # composite subtypes (open set; metacognition is the first, SPEC §18.2.1)
    METACOGNITION = "metacognition"


# Which base kind a subtype belongs to. Lets genesis/mutate accept a bare
# Subtype and infer its kind, so call sites stay terse (genesis(id, PROMPT, ...)).
_KIND_OF_SUBTYPE: dict[Subtype, ArtifactKind] = {
    Subtype.PROMPT: ArtifactKind.TEXT,
    Subtype.TOOL_DESCRIPTION: ArtifactKind.TEXT,
    Subtype.SKILL: ArtifactKind.TEXT,
    Subtype.RUBRIC: ArtifactKind.TEXT,
    Subtype.PLANNER: ArtifactKind.TEXT,
    Subtype.MONITOR: ArtifactKind.TEXT,
    Subtype.METACOGNITION: ArtifactKind.COMPOSITE,
}

# Composite subtypes that declare an emergent-risk layer floor above their
# constituents (SPEC §18.2.1). Metacognition self-modifies the agent's
# reasoning, so it is L3 even though its parts (L1 planner, L1 monitor, L2
# memory) top out at L2.
_COMPOSITE_LAYER_FLOOR: dict[Subtype | None, int] = {
    Subtype.METACOGNITION: 3,
}

_BASE_LAYER: dict[ArtifactKind, int] = {
    ArtifactKind.TEXT: 1,
    ArtifactKind.MEMORY_ENTRY: 2,
    ArtifactKind.CODE: 4,
}

# Retired v0.1 top-level kinds, mapped to their (kind, subtype) under v0.2.
# The archive uses this to migrate old rows on read (SPEC §1.2, §14).
_RETIRED_KIND_MIGRATION: dict[str, tuple[ArtifactKind, Subtype]] = {
    "prompt": (ArtifactKind.TEXT, Subtype.PROMPT),
    "tool_description": (ArtifactKind.TEXT, Subtype.TOOL_DESCRIPTION),
    "skill": (ArtifactKind.TEXT, Subtype.SKILL),
    "rubric": (ArtifactKind.TEXT, Subtype.RUBRIC),
    "planner": (ArtifactKind.TEXT, Subtype.PLANNER),
    "monitor": (ArtifactKind.TEXT, Subtype.MONITOR),
}


def layer_of(
    kind: ArtifactKind,
    subtype: Subtype | None = None,
    constituent_layers: tuple[int, ...] = (),
) -> int:
    """The improvement layer (1-4) for a (kind, subtype). SPEC §1.2, §18.2.

    text -> 1, memory_entry -> 2, code -> 4. A composite is
    `max(subtype floor, max over constituent layers)`: the floor lets a blessed
    composition such as metacognition (L3) carry a risk higher than its parts,
    while the constituent term keeps a bundle holding code at L4. Online
    improvers refuse any target at layer >= 3 (SPEC §17.3).
    """
    if kind is ArtifactKind.COMPOSITE:
        floor = _COMPOSITE_LAYER_FLOOR.get(subtype, 1)
        return max(floor, max(constituent_layers, default=1))
    return _BASE_LAYER[kind]


def resolve_kind(
    kind: ArtifactKind | Subtype,
    subtype: Subtype | None = None,
) -> tuple[ArtifactKind, Subtype | None]:
    """Normalize a (kind-or-subtype, subtype) pair to (ArtifactKind, Subtype | None).

    Callers may pass a bare Subtype (which implies its base kind) or an
    ArtifactKind plus an optional Subtype. Raises if the subtype does not
    belong to the kind.
    """
    if isinstance(kind, Subtype):
        if subtype is not None and subtype is not kind:
            raise ValueError(
                f"conflicting subtype: kind token {kind} vs subtype {subtype}"
            )
        return _KIND_OF_SUBTYPE[kind], kind
    if subtype is not None and _KIND_OF_SUBTYPE.get(subtype) is not kind:
        raise ValueError(f"subtype {subtype} does not belong to kind {kind}")
    return kind, subtype


def migrate_kind(
    raw_kind: str, raw_subtype: str | None = None
) -> tuple[ArtifactKind, Subtype | None]:
    """Map a persisted (kind, subtype) to v0.2, migrating retired v0.1 kinds.

    A v0.1 row stores a retired top-level kind (e.g. "prompt") and no subtype;
    it migrates to `(text, prompt)`. A v0.2 row stores `("text", "prompt")` and
    passes through. SPEC §1.2, §14.
    """
    if raw_kind in _RETIRED_KIND_MIGRATION:
        return _RETIRED_KIND_MIGRATION[raw_kind]
    return ArtifactKind(raw_kind), (Subtype(raw_subtype) if raw_subtype else None)


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
    content: str | dict[str, Any] | list[Any]
    subtype: Subtype | None = None
    parent_id: ParentRef = None
    created_by: str = "human"
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def layer(self) -> int:
        """This artifact's improvement layer (1-4). SPEC §1.2, §18.2.

        For a composite, the constituent layers are read from content and folded
        into `max(subtype floor, max constituents)`; other kinds derive from
        (kind, subtype) alone.
        """
        if self.kind is ArtifactKind.COMPOSITE:
            return layer_of(self.kind, self.subtype, self._constituent_layers())
        return layer_of(self.kind, self.subtype)

    @property
    def constituents(self) -> list[dict[str, Any]]:
        """For a composite, the ordered constituent descriptors; else []."""
        if self.kind is ArtifactKind.COMPOSITE and isinstance(self.content, dict):
            return list(self.content.get("constituents", []))
        return []

    @property
    def constituent_refs(self) -> list[tuple[str, int]]:
        """The (id, version) refs of a composite's constituents (SPEC §18.2)."""
        return [(c["id"], int(c["version"])) for c in self.constituents]

    def _constituent_layers(self) -> tuple[int, ...]:
        return tuple(
            layer_of(
                ArtifactKind(c["kind"]),
                Subtype(c["subtype"]) if c.get("subtype") else None,
            )
            for c in self.constituents
        )

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
            subtype=self.subtype,
            parent_id=self.ref,
            created_by=created_by,
            metadata=metadata or {},
        )


def genesis(
    id: str,
    kind: ArtifactKind | Subtype,
    content: str | dict[str, Any] | list[Any],
    created_by: str = "human",
    metadata: dict[str, Any] | None = None,
    subtype: Subtype | None = None,
) -> Artifact:
    """Create the first (genesis) version of an artifact.

    Genesis artifacts have parent_id=None and version=1. The kind argument may
    be a bare Subtype (which implies its base kind, e.g. `genesis(id, PROMPT,
    ...)` makes a text artifact) or an ArtifactKind with an optional subtype.
    """
    base_kind, sub = resolve_kind(kind, subtype)
    return Artifact(
        id=id,
        version=1,
        kind=base_kind,
        content=content,
        subtype=sub,
        parent_id=None,
        created_by=created_by,
        metadata=metadata or {},
    )


def composite_content(
    constituents: list[Artifact | tuple[Artifact, str]],
    binding: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the content payload for a composite from its constituents.

    Each entry records the constituent's ref, kind, subtype, and binding role,
    so a composite's layer is computable without an archive lookup (SPEC §18.2).
    A constituent may be an Artifact (role defaults to its subtype or kind) or
    an (artifact, role) tuple.
    """
    items: list[dict[str, Any]] = []
    for c in constituents:
        art, role = c if isinstance(c, tuple) else (c, None)
        items.append(
            {
                "id": art.id,
                "version": art.version,
                "kind": art.kind.value,
                "subtype": art.subtype.value if art.subtype else None,
                "role": role or (art.subtype.value if art.subtype else art.kind.value),
            }
        )
    return {"constituents": items, "binding": binding or {}}


def compose(
    id: str,
    constituents: list[Artifact | tuple[Artifact, str]],
    *,
    subtype: Subtype | None = None,
    binding: dict[str, Any] | None = None,
    created_by: str = "human",
    metadata: dict[str, Any] | None = None,
) -> Artifact:
    """Create a genesis composite artifact binding the given constituents.

    The composite is the unit of joint search and joint deployment for a
    coupled cluster (SPEC §18). Its layer is `max(subtype floor, max constituent
    layers)`, so a metacognition composite (subtype floor L3) over an L1 planner,
    an L1 monitor, and an L2 memory is L3.
    """
    return Artifact(
        id=id,
        version=1,
        kind=ArtifactKind.COMPOSITE,
        content=composite_content(constituents, binding),
        subtype=subtype,
        parent_id=None,
        created_by=created_by,
        metadata=metadata or {},
    )
