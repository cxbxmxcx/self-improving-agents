"""Shared types for the memory subsystem.

The MemoryTier protocol from SPEC section 7.1 in formal form, plus the
common types used across all four tiers. Each concrete tier implements this
protocol; the Agent and Improver hold tiers as a dict keyed by tier name.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Protocol, runtime_checkable


# ---------------------------------------------------------------------------
# Scope: per-user / per-org / global. SPEC section 7.5.
# ---------------------------------------------------------------------------

class Scope(str, Enum):
    SESSION = "session"   # one conversation
    USER = "user"         # one end user, across sessions
    ORG = "org"           # one organization, across users
    GLOBAL = "global"     # everyone, everywhere


@dataclass(frozen=True)
class ScopeKey:
    """The tuple identifying which scope-row a memory belongs to.

    Per-session memory is keyed by (SESSION, session_id).
    Per-user by (USER, user_id).
    Per-org by (ORG, org_id).
    Global by (GLOBAL, "*").
    """

    scope: Scope
    key: str

    def to_str(self) -> str:
        return f"{self.scope.value}:{self.key}"

    @classmethod
    def from_str(cls, s: str) -> ScopeKey:
        scope_str, _, key = s.partition(":")
        return cls(scope=Scope(scope_str), key=key)


# ---------------------------------------------------------------------------
# MemoryEntry: the unit of storage. Every tier stores entries with metadata.
# ---------------------------------------------------------------------------

EntryId = str


@dataclass
class MemoryEntry:
    """A single memory record. The shape that all tiers manipulate.

    Different tiers populate different fields:
      - Working memory: content is a message dict
      - Episodic: content is a {task, trajectory_id, summary, answer} payload
      - Semantic: content is a fact (key, value) pair
      - Procedural: content is a skill payload (name, code, description)
    """

    id: EntryId
    scope_key: ScopeKey
    content: dict[str, Any]
    embedding: list[float] | None = None  # optional vector for retrieval
    score: float = 0.0  # utility / relevance score, updated by usage
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_accessed_at: datetime | None = None
    access_count: int = 0

    def to_row(self) -> dict[str, Any]:
        """Serialize to a flat dict (for SQLite or JSON)."""
        return {
            "id": self.id,
            "scope": self.scope_key.scope.value,
            "scope_key": self.scope_key.key,
            "content": self.content,
            "embedding": self.embedding,
            "score": self.score,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "last_accessed_at": self.last_accessed_at.isoformat() if self.last_accessed_at else None,
            "access_count": self.access_count,
        }


# ---------------------------------------------------------------------------
# EvictionPolicy
# ---------------------------------------------------------------------------

@dataclass
class EvictionPolicy:
    """Knobs for tier eviction passes. SPEC section 7.1."""

    max_entries: int | None = None       # drop oldest beyond this
    max_age_days: float | None = None    # drop entries older than this
    min_score: float | None = None       # drop entries scoring below this
    only_scope: Scope | None = None      # restrict to one scope


# ---------------------------------------------------------------------------
# ConsolidationReport
# ---------------------------------------------------------------------------

@dataclass
class ConsolidationReport:
    """Result of a consolidate() pass."""

    tier: str
    entries_before: int = 0
    entries_after: int = 0
    entries_summarized: int = 0
    entries_dropped: int = 0
    notes: str = ""


# ---------------------------------------------------------------------------
# Query and Context types
# ---------------------------------------------------------------------------

@dataclass
class MemoryQuery:
    """What a caller wants when they read() from a tier.

    text: free-text query, used for vector / FTS similarity
    scope_keys: which (scope, key) pairs to search
    k: top-k results to return
    extra: tier-specific filter hints
    """

    text: str = ""
    scope_keys: list[ScopeKey] = field(default_factory=list)
    k: int = 5
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryContext:
    """Auxiliary context for memory operations.

    session_id, user_id, org_id are convenience fields; the tier translates
    them into ScopeKeys internally as appropriate.
    """

    session_id: str | None = None
    user_id: str | None = None
    org_id: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def scope_keys(self, scopes: list[Scope] | None = None) -> list[ScopeKey]:
        """Translate this context into ScopeKeys for the requested scopes.

        If scopes is None, returns all available levels populated by the
        context.
        """
        out: list[ScopeKey] = []
        wanted = scopes or [Scope.SESSION, Scope.USER, Scope.ORG, Scope.GLOBAL]
        if Scope.SESSION in wanted and self.session_id:
            out.append(ScopeKey(Scope.SESSION, self.session_id))
        if Scope.USER in wanted and self.user_id:
            out.append(ScopeKey(Scope.USER, self.user_id))
        if Scope.ORG in wanted and self.org_id:
            out.append(ScopeKey(Scope.ORG, self.org_id))
        if Scope.GLOBAL in wanted:
            out.append(ScopeKey(Scope.GLOBAL, "*"))
        return out


# ---------------------------------------------------------------------------
# MemoryTier Protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class MemoryTier(Protocol):
    """Contract every tier implements. SPEC section 7.1.

    All methods are async; storage backends can be SQLite (sync internally,
    wrapped) or remote (true async).
    """

    @property
    def tier_name(self) -> str: ...

    async def read(
        self,
        query: MemoryQuery,
        context: MemoryContext,
    ) -> list[MemoryEntry]: ...

    async def write(
        self,
        entry: MemoryEntry,
    ) -> EntryId: ...

    async def score(self, entry: MemoryEntry) -> float: ...

    async def evict(self, policy: EvictionPolicy) -> int: ...

    async def consolidate(self) -> ConsolidationReport: ...
