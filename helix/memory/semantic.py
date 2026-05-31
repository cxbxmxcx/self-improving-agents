"""Semantic memory: facts learned about the world.

Stores key-value facts scoped per-user / per-org / globally. Examples:
  - per-user: "this user prefers metric units", "this user is on Linux"
  - per-org:  "use the company style guide", "don't recommend competitor X"
  - global:   "the current date is X", "the corpus was last rebuilt on Y"

SQLite backend with optional vector indexing so callers can fuzzy-retrieve
facts by description ("what do we know about this user's environment?")
rather than exact key. SPEC section 7.2.

The key-value model maps cleanly to the MemoryTier protocol:
  - write(MemoryEntry(content={"key": "preferred_units", "value": "metric"}))
  - read(MemoryQuery(text="units")) returns the entry with key "preferred_units"

For exact-key lookup, use read_fact(key, scope_key) directly (faster, no
embedding cost).
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from helix.memory.base import (
    ConsolidationReport,
    EntryId,
    EvictionPolicy,
    MemoryContext,
    MemoryEntry,
    MemoryQuery,
    Scope,
    ScopeKey,
)
from helix.memory.episodic import _cosine, _embed


_SCHEMA = """
CREATE TABLE IF NOT EXISTS semantic_facts (
    id TEXT PRIMARY KEY,
    scope TEXT NOT NULL,
    scope_key TEXT NOT NULL,
    fact_key TEXT NOT NULL,
    fact_value TEXT NOT NULL,
    description TEXT,
    embedding_json TEXT,
    score REAL NOT NULL DEFAULT 0,
    metadata_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    last_accessed_at TEXT,
    access_count INTEGER NOT NULL DEFAULT 0,
    UNIQUE(scope, scope_key, fact_key)
);
CREATE INDEX IF NOT EXISTS idx_semantic_scope ON semantic_facts(scope, scope_key);
CREATE INDEX IF NOT EXISTS idx_semantic_key ON semantic_facts(fact_key);
"""


class SemanticMemory:
    """SQLite-backed semantic memory tier.

    Two access patterns:
      1. Exact key (fast, no embedding): read_fact("preferred_units", scope_key)
      2. Fuzzy text (slower, retrieves by description): read(MemoryQuery(text=...))
    """

    tier_name: str = "semantic"

    def __init__(self, path: str | Path = ":memory:", check_same_thread: bool = True) -> None:
        self.path = str(path)
        self._conn = sqlite3.connect(self.path, check_same_thread=check_same_thread)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # ---------------- direct fact API ----------------

    async def write_fact(
        self,
        key: str,
        value: str,
        scope_key: ScopeKey,
        description: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> EntryId:
        """Upsert (scope_key, fact_key) -> value. Convenience wrapper around write()."""
        embedding = None
        if description:
            vectors = _embed([description])
            embedding = vectors[0] if vectors else None

        entry_id = str(uuid.uuid4())
        self._conn.execute(
            """
            INSERT INTO semantic_facts
            (id, scope, scope_key, fact_key, fact_value, description,
             embedding_json, score, metadata_json, created_at,
             last_accessed_at, access_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(scope, scope_key, fact_key) DO UPDATE SET
                fact_value = excluded.fact_value,
                description = excluded.description,
                embedding_json = excluded.embedding_json,
                metadata_json = excluded.metadata_json,
                last_accessed_at = excluded.created_at
            """,
            (
                entry_id,
                scope_key.scope.value,
                scope_key.key,
                key,
                value,
                description,
                json.dumps(embedding) if embedding else None,
                0.0,
                json.dumps(metadata or {}),
                datetime.now(timezone.utc).isoformat(),
                None,
                0,
            ),
        )
        self._conn.commit()
        return entry_id

    async def read_fact(self, key: str, scope_key: ScopeKey) -> str | None:
        """Exact-key lookup. Returns the value or None."""
        row = self._conn.execute(
            "SELECT fact_value FROM semantic_facts WHERE scope = ? AND scope_key = ? AND fact_key = ?",
            (scope_key.scope.value, scope_key.key, key),
        ).fetchone()
        if row is None:
            return None
        self._conn.execute(
            "UPDATE semantic_facts SET access_count = access_count + 1, last_accessed_at = ? WHERE scope = ? AND scope_key = ? AND fact_key = ?",
            (datetime.now(timezone.utc).isoformat(), scope_key.scope.value, scope_key.key, key),
        )
        self._conn.commit()
        return row["fact_value"]

    async def list_facts(self, scope_key: ScopeKey) -> list[tuple[str, str]]:
        """Return (key, value) pairs for every fact in this scope."""
        rows = self._conn.execute(
            "SELECT fact_key, fact_value FROM semantic_facts WHERE scope = ? AND scope_key = ? ORDER BY fact_key",
            (scope_key.scope.value, scope_key.key),
        ).fetchall()
        return [(r["fact_key"], r["fact_value"]) for r in rows]

    # ---------------- internals ----------------

    def _row_to_entry(self, row: sqlite3.Row) -> MemoryEntry:
        return MemoryEntry(
            id=row["id"],
            scope_key=ScopeKey(Scope(row["scope"]), row["scope_key"]),
            content={
                "key": row["fact_key"],
                "value": row["fact_value"],
                "description": row["description"],
            },
            embedding=json.loads(row["embedding_json"]) if row["embedding_json"] else None,
            score=float(row["score"]),
            metadata=json.loads(row["metadata_json"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            last_accessed_at=datetime.fromisoformat(row["last_accessed_at"]) if row["last_accessed_at"] else None,
            access_count=int(row["access_count"]),
        )

    # ---------------- MemoryTier contract ----------------

    async def write(self, entry: MemoryEntry) -> EntryId:
        """Generic write path. Caller's content must include 'key' and 'value'."""
        key = entry.content.get("key")
        value = entry.content.get("value")
        if not key or value is None:
            raise ValueError("SemanticMemory.write requires content with 'key' and 'value'")
        return await self.write_fact(
            key=str(key),
            value=str(value),
            scope_key=entry.scope_key,
            description=entry.content.get("description"),
            metadata=entry.metadata,
        )

    async def read(
        self,
        query: MemoryQuery,
        context: MemoryContext,
    ) -> list[MemoryEntry]:
        """Fuzzy text retrieval across scope_keys. Ranks by cosine similarity."""
        scope_keys = query.scope_keys or context.scope_keys()
        if not scope_keys:
            return []

        placeholders = ",".join(["(?,?)"] * len(scope_keys))
        params: list[Any] = []
        for sk in scope_keys:
            params.extend([sk.scope.value, sk.key])

        rows = self._conn.execute(
            f"SELECT * FROM semantic_facts WHERE (scope, scope_key) IN ({placeholders})",
            params,
        ).fetchall()
        candidates = [self._row_to_entry(r) for r in rows]

        if not query.text:
            return candidates[: query.k]

        vectors = _embed([query.text])
        if not vectors:
            return candidates[: query.k]
        qvec = vectors[0]

        scored = []
        for entry in candidates:
            sim = _cosine(qvec, entry.embedding) if entry.embedding else 0.0
            scored.append((sim, entry))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [e for _, e in scored[: query.k]]

    async def score(self, entry: MemoryEntry) -> float:
        row = self._conn.execute(
            "SELECT access_count FROM semantic_facts WHERE id = ?",
            (entry.id,),
        ).fetchone()
        if row is None:
            return 0.0
        return min(1.0, row["access_count"] / 10.0)

    async def evict(self, policy: EvictionPolicy) -> int:
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

        before = self._conn.execute("SELECT COUNT(*) AS n FROM semantic_facts").fetchone()["n"]
        if clauses:
            self._conn.execute(
                f"DELETE FROM semantic_facts WHERE {' AND '.join(clauses)}",
                params,
            )
        if policy.max_entries is not None:
            # Keep the newest max_entries; drop the rest (uniform memory
            # contract, SPEC §7.1).
            self._conn.execute(
                """
                DELETE FROM semantic_facts
                WHERE id NOT IN (
                    SELECT id FROM semantic_facts
                    ORDER BY created_at DESC LIMIT ?
                )
                """,
                (policy.max_entries,),
            )
        self._conn.commit()
        after = self._conn.execute("SELECT COUNT(*) AS n FROM semantic_facts").fetchone()["n"]
        return before - after

    async def consolidate(self) -> ConsolidationReport:
        n = self._conn.execute("SELECT COUNT(*) AS n FROM semantic_facts").fetchone()["n"]
        return ConsolidationReport(
            tier=self.tier_name,
            entries_before=n,
            entries_after=n,
            notes="semantic consolidation (deduplication, contradiction-resolution) is a stub",
        )

    async def count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) AS n FROM semantic_facts").fetchone()["n"]

    def close(self) -> None:
        self._conn.close()
