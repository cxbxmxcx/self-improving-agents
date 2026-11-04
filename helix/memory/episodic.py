"""Episodic memory: events and trajectories indexed for retrieval.

Storage is SQLite with optional vector embeddings over the entry's
searchable text (typically `user_message` + `final_output`). Retrieval
combines vector similarity and recency with stored utility under fixed weights in a ranked
list. SPEC section 7.2.

The unit of storage is a trajectory summary: one entry per agent.run().
At write time the entry contains the user's question, the agent's answer,
metadata (scope, timestamps), and an embedding. At read time the tier
returns the top-k entries most relevant to the current query.

Scope-aware: every entry belongs to exactly one (Scope, key) pair, and
reads can request entries from multiple scope levels at once (e.g. session
+ user + global). Cross-scope leakage is prevented by always filtering on
the requested scope_keys.

The default embedding model is lazy-loaded once per process. The vector format is
a JSON list[float] stored as a column; for production-scale you'd swap to
sqlite-vec or pgvector with no API change.
"""

from __future__ import annotations

from collections.abc import Callable
import json
import math
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from helix.memory.archive_bridge import record_genesis
from helix.memory.base import (
    ConsolidationReport,
    EntryId,
    EvictionPolicy,
    MemoryContext,
    MemoryEntry,
    MemoryQuery,
    Scope,
    ScopeKey,
    ref_from_str,
    ref_to_str,
)


# Reuse the retrieval-corpus embedder; same model, same dimensions.
_EMBED_MODEL_NAME = "BAAI/bge-base-en-v1.5"
_EMBED_DIM = 768

_embedder = None


def _get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer(_EMBED_MODEL_NAME)
    return _embedder


def _embed(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    model = _get_embedder()
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return vectors.tolist()


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    return max(0.0, min(1.0, dot))  # already-normalized vectors


_SCHEMA = """
CREATE TABLE IF NOT EXISTS episodic_entries (
    id TEXT PRIMARY KEY,
    scope TEXT NOT NULL,
    scope_key TEXT NOT NULL,
    searchable_text TEXT NOT NULL,
    content_json TEXT NOT NULL,
    embedding_json TEXT,
    score REAL NOT NULL DEFAULT 0,
    metadata_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    last_accessed_at TEXT,
    access_count INTEGER NOT NULL DEFAULT 0,
    artifact_ref TEXT
);
CREATE INDEX IF NOT EXISTS idx_episodic_scope ON episodic_entries(scope, scope_key);
CREATE INDEX IF NOT EXISTS idx_episodic_created ON episodic_entries(created_at);
"""


class EpisodicMemory:
    """SQLite-backed episodic memory tier with vector retrieval.

    A supplied embedding_fn must return consistently sized, normalized vectors.
    Reopen a persisted store with the same encoder used to write its entries.
    Omitting the callable preserves the default sentence-transformer model.

    Typical usage:
        em = EpisodicMemory(path="data/episodic.sqlite")
        await em.write(MemoryEntry(
            id=str(uuid4()),
            scope_key=ScopeKey(Scope.SESSION, session_id),
            content={"user_message": q, "final_output": a, "trajectory_id": tid},
        ))
        hits = await em.read(
            MemoryQuery(text="similar question?", scope_keys=[...], k=3),
            MemoryContext(session_id=..., user_id=...),
        )
    """

    tier_name: str = "episodic"

    def __init__(
        self,
        path: str | Path = ":memory:",
        recency_weight: float = 0.2,
        check_same_thread: bool = True,
        archive: Any | None = None,
        utility_weight: float = 0.0,
        *,
        embedding_fn: Callable[[list[str]], list[list[float]]] | None = None,
    ) -> None:
        self.path = str(path)
        self.recency_weight = recency_weight
        # A caller may supply a fixed local encoder for a small offline corpus.
        # Keep that encoder consistent when reopening the store; learning never
        # selects or modifies it. None retains the existing default embedder.
        self.embedding_fn = embedding_fn
        # Fixed harness configuration: learned values enter this ranking,
        # but learning does not tune its weight. Zero is the relevance-only
        # control; Chapter 5 configures 0.35 before running the learner.
        self.utility_weight = max(0.0, min(1.0, utility_weight))
        # The shared archive, when given, makes every entry a MEMORY_ENTRY
        # artifact with lineage (SPEC §7.3); the tier table stays the live
        # index. Without it the tier behaves as it did in Chapter 2.
        self.archive = archive
        self._conn = sqlite3.connect(self.path, check_same_thread=check_same_thread)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._migrate_artifact_ref_column()
        self._conn.commit()

    def _migrate_artifact_ref_column(self) -> None:
        """Stores created before v0.3 lack the artifact_ref column."""
        cols = {r["name"] for r in self._conn.execute("PRAGMA table_info(episodic_entries)")}
        if "artifact_ref" not in cols:
            self._conn.execute("ALTER TABLE episodic_entries ADD COLUMN artifact_ref TEXT")

    # ---------------- internals ----------------

    @staticmethod
    def _searchable(content: dict[str, Any]) -> str:
        """Build the text we'll embed/search. user_message + final_output + summary."""
        parts: list[str] = []
        for key in ("user_message", "task", "final_output", "answer", "summary"):
            v = content.get(key)
            if isinstance(v, str) and v.strip():
                parts.append(v.strip())
        return "\n".join(parts) or json.dumps(content, default=str)

    def _row_to_entry(self, row: sqlite3.Row) -> MemoryEntry:
        return MemoryEntry(
            id=row["id"],
            scope_key=ScopeKey(Scope(row["scope"]), row["scope_key"]),
            content=json.loads(row["content_json"]),
            embedding=json.loads(row["embedding_json"]) if row["embedding_json"] else None,
            score=float(row["score"]),
            metadata=json.loads(row["metadata_json"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            last_accessed_at=datetime.fromisoformat(row["last_accessed_at"]) if row["last_accessed_at"] else None,
            access_count=int(row["access_count"]),
            artifact_ref=ref_from_str(row["artifact_ref"]) if "artifact_ref" in row.keys() else None,
        )

    # ---------------- MemoryTier contract ----------------

    async def write(self, entry: MemoryEntry) -> EntryId:
        """Persist an entry. Computes the embedding if not already set.

        The caller's entry is updated in place with the id and artifact ref
        the write assigned, so the hook that wrote it can report them.
        """
        if entry.embedding is None:
            text = self._searchable(entry.content)
            if text:
                vectors = (self.embedding_fn or _embed)([text])
                if vectors:
                    entry.embedding = vectors[0]

        eid = entry.id or str(uuid.uuid4())
        if not entry.id:
            entry.id = eid
        # First write of an entry under an archive is ExpeL's ADD: a genesis
        # artifact. Later writes arrive with a ref already set by an operator
        # (EDIT, UPVOTE, DOWNVOTE minted the child version) and only refresh
        # the index row.
        if self.archive is not None and entry.artifact_ref is None:
            entry.artifact_ref = await record_genesis(self.archive, self.tier_name, entry)
        self._conn.execute(
            """
            INSERT OR REPLACE INTO episodic_entries
            (id, scope, scope_key, searchable_text, content_json, embedding_json,
             score, metadata_json, created_at, last_accessed_at, access_count,
             artifact_ref)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                eid,
                entry.scope_key.scope.value,
                entry.scope_key.key,
                self._searchable(entry.content),
                json.dumps(entry.content, default=str),
                json.dumps(entry.embedding) if entry.embedding else None,
                entry.score,
                json.dumps(entry.metadata),
                entry.created_at.isoformat(),
                entry.last_accessed_at.isoformat() if entry.last_accessed_at else None,
                entry.access_count,
                ref_to_str(entry.artifact_ref),
            ),
        )
        self._conn.commit()
        return eid

    async def get(self, entry_id: EntryId) -> MemoryEntry | None:
        """Fetch one entry by id without touching its access stats."""
        row = self._conn.execute(
            "SELECT * FROM episodic_entries WHERE id = ?", (entry_id,)
        ).fetchone()
        return self._row_to_entry(row) if row is not None else None

    async def by_artifact_id(self, artifact_id: str) -> MemoryEntry | None:
        """Fetch the entry whose live index row points at this artifact id."""
        row = self._conn.execute(
            "SELECT * FROM episodic_entries WHERE artifact_ref LIKE ?",
            (f"{artifact_id}:%",),
        ).fetchone()
        return self._row_to_entry(row) if row is not None else None

    async def read(
        self,
        query: MemoryQuery,
        context: MemoryContext,
    ) -> list[MemoryEntry]:
        """Return top-k entries under the configured relevance/utility blend.

        The set of (scope, key) pairs to search comes from query.scope_keys
        if provided, otherwise from context.scope_keys() across all levels.
        Cross-scope leakage is prevented by strict filtering on these pairs.
        """
        scope_keys = query.scope_keys or context.scope_keys()
        if not scope_keys:
            return []

        # Build the IN clause for (scope, scope_key) pairs
        placeholders = ",".join(["(?,?)"] * len(scope_keys))
        params: list[Any] = []
        for sk in scope_keys:
            params.extend([sk.scope.value, sk.key])

        rows = self._conn.execute(
            f"""
            SELECT * FROM episodic_entries
            WHERE (scope, scope_key) IN ({placeholders})
            ORDER BY created_at DESC
            """,
            params,
        ).fetchall()

        if not rows:
            return []

        candidates = [self._row_to_entry(r) for r in rows]

        # Score by similarity + recency.
        scored: list[tuple[float, MemoryEntry]] = []
        query_vec: list[float] | None = None
        if query.text:
            vectors = (self.embedding_fn or _embed)([query.text])
            query_vec = vectors[0] if vectors else None

        now = datetime.now(timezone.utc)
        for entry in candidates:
            sim = 0.0
            if query_vec is not None and entry.embedding:
                sim = _cosine(query_vec, entry.embedding)

            # Recency: 1.0 at creation, decays with log of age in days
            age_days = max(0.001, (now - entry.created_at).total_seconds() / 86400.0)
            recency = 1.0 / (1.0 + math.log10(age_days + 1.0))

            relevance = (1.0 - self.recency_weight) * sim + self.recency_weight * recency
            # Scope already determined eligibility. The formula and weight
            # stay fixed while learned entry values can change the order.
            combined = (1.0 - self.utility_weight) * relevance + self.utility_weight * entry.score
            scored.append((combined, entry))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = [e for _, e in scored[: query.k]]

        # Update access stats
        for entry in top:
            self._conn.execute(
                "UPDATE episodic_entries SET access_count = access_count + 1, last_accessed_at = ? WHERE id = ?",
                (now.isoformat(), entry.id),
            )
        self._conn.commit()
        return top

    async def score(self, entry: MemoryEntry) -> float:
        """Use access_count + recency as a simple utility proxy."""
        row = self._conn.execute(
            "SELECT access_count, created_at FROM episodic_entries WHERE id = ?",
            (entry.id,),
        ).fetchone()
        if row is None:
            return 0.0
        age_days = max(0.001, (datetime.now(timezone.utc) - datetime.fromisoformat(row["created_at"])).total_seconds() / 86400.0)
        recency = 1.0 / (1.0 + math.log10(age_days + 1.0))
        return 0.7 * recency + 0.3 * min(1.0, row["access_count"] / 10.0)

    async def evict(self, policy: EvictionPolicy) -> int:
        """Drop entries per policy. Returns count dropped."""
        clauses: list[str] = []
        params: list[Any] = []
        if policy.only_scope is not None:
            clauses.append("scope = ?")
            params.append(policy.only_scope.value)
        if policy.max_age_days is not None:
            from datetime import timedelta
            cutoff = (datetime.now(timezone.utc) - timedelta(days=policy.max_age_days)).isoformat()
            clauses.append("created_at < ?")
            params.append(cutoff)
        if policy.min_score is not None:
            clauses.append("score < ?")
            params.append(policy.min_score)

        # Count the whole table before and after so the return reflects every
        # eviction, whether it came from the clause delete or the max_entries
        # cap (a filtered before-count miscounts both paths).
        before = self._conn.execute("SELECT COUNT(*) AS n FROM episodic_entries").fetchone()["n"]

        if clauses:
            self._conn.execute(
                f"DELETE FROM episodic_entries WHERE {' AND '.join(clauses)}",
                params,
            )

        if policy.max_entries is not None:
            # Keep newest max_entries; drop the rest
            self._conn.execute(
                """
                DELETE FROM episodic_entries
                WHERE id NOT IN (
                    SELECT id FROM episodic_entries
                    ORDER BY created_at DESC LIMIT ?
                )
                """,
                (policy.max_entries,),
            )

        self._conn.commit()
        after = self._conn.execute("SELECT COUNT(*) AS n FROM episodic_entries").fetchone()["n"]
        return before - after

    async def consolidate(self) -> ConsolidationReport:
        """Run the 3 AM pass with default settings. SPEC §7.4.

        The policy knobs, the merge and promote steps, and the scheduler live
        in helix.memory.consolidation; this method is the MemoryTier contract
        entry point that callers with no opinions use.
        """
        from helix.memory.consolidation import consolidate_episodic

        return await consolidate_episodic(self)

    # ---------------- introspection helpers (used by dashboard) ----------------

    async def count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) AS n FROM episodic_entries").fetchone()["n"]

    async def list_all(self, scope_key: ScopeKey | None = None) -> list[MemoryEntry]:
        """Every live row, optionally within one scope, oldest first."""
        if scope_key is None:
            rows = self._conn.execute(
                "SELECT * FROM episodic_entries ORDER BY created_at ASC"
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM episodic_entries WHERE scope = ? AND scope_key = ? ORDER BY created_at ASC",
                (scope_key.scope.value, scope_key.key),
            ).fetchall()
        return [self._row_to_entry(r) for r in rows]

    async def remove(self, entry_id: EntryId) -> bool:
        """Forget one index row; the archive keeps its artifact chain."""
        cur = self._conn.execute("DELETE FROM episodic_entries WHERE id = ?", (entry_id,))
        self._conn.commit()
        return cur.rowcount > 0

    async def list_recent(self, n: int = 20) -> list[MemoryEntry]:
        rows = self._conn.execute(
            "SELECT * FROM episodic_entries ORDER BY created_at DESC LIMIT ?",
            (n,),
        ).fetchall()
        return [self._row_to_entry(r) for r in rows]

    def close(self) -> None:
        self._conn.close()
