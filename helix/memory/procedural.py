"""Procedural memory: skills with provenance and reuse.

Stores named, reusable procedures (skills) the agent has accumulated. Each
skill has a description, optional code, provenance (who/what created it,
when), and a usage count that grows over time. SPEC section 7.2.

Chapter 6 puts this tier to work: text skills reach the model as context
through the memory hooks, and code skills (body_kind "python") become tools
through helix/memory/code_skills.py. Chapter 7 builds on both when the
harness itself comes under search.

Skills are also Artifacts (kind=Subtype.SKILL); this tier handles
their storage and retrieval. The Archive handles their lineage and
measurements. The two systems cooperate: when SPO/GEPA produces a new skill
candidate, the Archive records the version chain and the Improver writes
the winner here for the Agent to consume.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from collections.abc import Callable
from pathlib import Path
from typing import Any

from helix.memory.archive_bridge import (
    latest_artifact,
    memory_artifact_content,
    record_genesis,
    record_version,
)
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
from helix.memory.episodic import _cosine, _embed


_SCHEMA = """
CREATE TABLE IF NOT EXISTS procedural_skills (
    id TEXT PRIMARY KEY,
    scope TEXT NOT NULL,
    scope_key TEXT NOT NULL,
    skill_name TEXT NOT NULL,
    description TEXT NOT NULL,
    body TEXT,
    body_kind TEXT,
    artifact_ref TEXT,
    embedding_json TEXT,
    usage_count INTEGER NOT NULL DEFAULT 0,
    last_used_at TEXT,
    score REAL NOT NULL DEFAULT 0,
    metadata_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(scope, scope_key, skill_name)
);
CREATE INDEX IF NOT EXISTS idx_procedural_scope ON procedural_skills(scope, scope_key);
CREATE INDEX IF NOT EXISTS idx_procedural_name ON procedural_skills(skill_name);
"""


class ProceduralMemory:
    """SQLite-backed procedural memory.

    Direct API (for skill management):
        save_skill(name, description, body, ...)
        get_skill(name, scope_key)
        list_skills(scope_key)
        record_use(name, scope_key)  # increments usage_count

    Generic MemoryTier API (for uniform inspection):
        read(MemoryQuery(text="describe a skill"))
        write(MemoryEntry(content={"skill_name": ..., "description": ..., "body": ...}))
    """

    tier_name: str = "procedural"

    def __init__(
        self,
        path: str | Path = ":memory:",
        check_same_thread: bool = True,
        archive: Any | None = None,
        embedding_fn: Callable[[list[str]], list[list[float]]] | None = None,
    ) -> None:
        self.path = str(path)
        # As in the episodic tier: a chapter may fix its own encoder so
        # recall needs no model download. None keeps the default embedder.
        self.embedding_fn = embedding_fn
        # With an archive, a skill row with no explicit artifact_ref is
        # recorded as a MEMORY_ENTRY artifact and re-saves become child
        # versions. Chapter 7 passes an explicit ref to a text:skill artifact
        # instead, and that ref wins.
        self.archive = archive
        self._conn = sqlite3.connect(self.path, check_same_thread=check_same_thread)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # ---------------- direct skill API ----------------

    async def save_skill(
        self,
        name: str,
        description: str,
        body: str = "",
        body_kind: str = "instructions",  # "instructions" | "python" | "json"
        scope_key: ScopeKey | None = None,
        artifact_ref: tuple[str, int] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> EntryId:
        """Upsert a skill. Description is embedded for fuzzy retrieval."""
        scope_key = scope_key or ScopeKey(Scope.GLOBAL, "*")
        vectors = (self.embedding_fn or _embed)([description]) if description else []
        embedding = vectors[0] if vectors else None
        existing = self._conn.execute(
            "SELECT id, artifact_ref FROM procedural_skills WHERE scope = ? AND scope_key = ? AND skill_name = ?",
            (scope_key.scope.value, scope_key.key, name),
        ).fetchone()
        entry_id = existing["id"] if existing is not None else str(uuid.uuid4())
        if artifact_ref is None and self.archive is not None:
            prior = ref_from_str(existing["artifact_ref"]) if existing is not None else None
            entry = MemoryEntry(
                id=entry_id,
                scope_key=scope_key,
                content={"name": name, "description": description, "body": body, "body_kind": body_kind},
                metadata=dict(metadata or {}),
            )
            parent = await latest_artifact(self.archive, prior[0]) if prior else None
            if parent is None or not parent.id.startswith("memory."):
                artifact_ref = await record_genesis(self.archive, self.tier_name, entry)
            else:
                artifact_ref = await record_version(
                    self.archive,
                    parent,
                    memory_artifact_content(self.tier_name, entry),
                    created_by=str((metadata or {}).get("created_by") or "agent"),
                    operator="edit",
                )
        artifact_ref_str = ref_to_str(artifact_ref)
        self._conn.execute(
            """
            INSERT INTO procedural_skills
            (id, scope, scope_key, skill_name, description, body, body_kind,
             artifact_ref, embedding_json, usage_count, last_used_at, score,
             metadata_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(scope, scope_key, skill_name) DO UPDATE SET
                description = excluded.description,
                body = excluded.body,
                body_kind = excluded.body_kind,
                artifact_ref = excluded.artifact_ref,
                embedding_json = excluded.embedding_json,
                metadata_json = excluded.metadata_json
            """,
            (
                entry_id,
                scope_key.scope.value,
                scope_key.key,
                name,
                description,
                body,
                body_kind,
                artifact_ref_str,
                json.dumps(embedding) if embedding else None,
                0,
                None,
                0.0,
                json.dumps(metadata or {}),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        self._conn.commit()
        return entry_id

    async def get_skill(self, name: str, scope_key: ScopeKey) -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM procedural_skills WHERE scope = ? AND scope_key = ? AND skill_name = ?",
            (scope_key.scope.value, scope_key.key, name),
        ).fetchone()
        if row is None:
            return None
        return self._row_to_skill(row)

    async def list_skills(self, scope_key: ScopeKey) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM procedural_skills WHERE scope = ? AND scope_key = ? ORDER BY usage_count DESC, skill_name",
            (scope_key.scope.value, scope_key.key),
        ).fetchall()
        return [self._row_to_skill(r) for r in rows]

    async def record_use(self, name: str, scope_key: ScopeKey) -> None:
        self._conn.execute(
            "UPDATE procedural_skills SET usage_count = usage_count + 1, last_used_at = ? WHERE scope = ? AND scope_key = ? AND skill_name = ?",
            (datetime.now(timezone.utc).isoformat(), scope_key.scope.value, scope_key.key, name),
        )
        self._conn.commit()

    # ---------------- internals ----------------

    def _row_to_skill(self, row: sqlite3.Row) -> dict:
        return {
            "id": row["id"],
            "scope_key": ScopeKey(Scope(row["scope"]), row["scope_key"]),
            "name": row["skill_name"],
            "description": row["description"],
            "body": row["body"],
            "body_kind": row["body_kind"],
            "artifact_ref": row["artifact_ref"],
            "usage_count": row["usage_count"],
            "last_used_at": row["last_used_at"],
            "metadata": json.loads(row["metadata_json"]),
            "created_at": row["created_at"],
        }

    def _row_to_entry(self, row: sqlite3.Row) -> MemoryEntry:
        return MemoryEntry(
            id=row["id"],
            scope_key=ScopeKey(Scope(row["scope"]), row["scope_key"]),
            content={
                "skill_name": row["skill_name"],
                "description": row["description"],
                "body": row["body"],
                "body_kind": row["body_kind"],
                "artifact_ref": row["artifact_ref"],
            },
            embedding=json.loads(row["embedding_json"]) if row["embedding_json"] else None,
            score=float(row["score"]),
            metadata=json.loads(row["metadata_json"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            last_accessed_at=datetime.fromisoformat(row["last_used_at"]) if row["last_used_at"] else None,
            access_count=int(row["usage_count"]),
            artifact_ref=ref_from_str(row["artifact_ref"]),
        )

    # ---------------- MemoryTier contract ----------------

    async def write(self, entry: MemoryEntry) -> EntryId:
        name = entry.content.get("skill_name") or entry.content.get("name")
        description = entry.content.get("description", "")
        if not name:
            raise ValueError("ProceduralMemory.write requires content with 'skill_name' (or 'name')")
        return await self.save_skill(
            name=str(name),
            description=str(description),
            body=str(entry.content.get("body", "")),
            body_kind=str(entry.content.get("body_kind", "instructions")),
            scope_key=entry.scope_key,
            artifact_ref=entry.content.get("artifact_ref"),
            metadata=entry.metadata,
        )

    async def read(
        self,
        query: MemoryQuery,
        context: MemoryContext,
    ) -> list[MemoryEntry]:
        scope_keys = query.scope_keys or context.scope_keys()
        if not scope_keys:
            return []
        placeholders = ",".join(["(?,?)"] * len(scope_keys))
        params: list[Any] = []
        for sk in scope_keys:
            params.extend([sk.scope.value, sk.key])
        rows = self._conn.execute(
            f"SELECT * FROM procedural_skills WHERE (scope, scope_key) IN ({placeholders})",
            params,
        ).fetchall()
        candidates = [self._row_to_entry(r) for r in rows]
        if not query.text:
            # Sort by usage_count desc as a sensible default
            candidates.sort(key=lambda e: e.access_count, reverse=True)
            return candidates[: query.k]
        vectors = (self.embedding_fn or _embed)([query.text])
        if not vectors:
            return candidates[: query.k]
        qvec = vectors[0]
        scored = []
        for entry in candidates:
            sim = _cosine(qvec, entry.embedding) if entry.embedding else 0.0
            # Kept on the returned copy only, so a harness can report how
            # closely the recalled skill matched; the stored row is unchanged.
            entry.metadata["retrieval_score"] = round(float(sim), 4)
            scored.append((sim, entry))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [e for _, e in scored[: query.k]]

    async def score(self, entry: MemoryEntry) -> float:
        row = self._conn.execute(
            "SELECT usage_count FROM procedural_skills WHERE id = ?",
            (entry.id,),
        ).fetchone()
        if row is None:
            return 0.0
        return min(1.0, row["usage_count"] / 20.0)

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
            clauses.append("usage_count < ?")
            params.append(int(policy.min_score * 20))  # rough inverse of score
        before = self._conn.execute("SELECT COUNT(*) AS n FROM procedural_skills").fetchone()["n"]
        if clauses:
            self._conn.execute(
                f"DELETE FROM procedural_skills WHERE {' AND '.join(clauses)}",
                params,
            )
        if policy.max_entries is not None:
            # Keep the newest max_entries; drop the rest (uniform memory
            # contract, SPEC §7.1).
            self._conn.execute(
                """
                DELETE FROM procedural_skills
                WHERE id NOT IN (
                    SELECT id FROM procedural_skills
                    ORDER BY created_at DESC LIMIT ?
                )
                """,
                (policy.max_entries,),
            )
        self._conn.commit()
        after = self._conn.execute("SELECT COUNT(*) AS n FROM procedural_skills").fetchone()["n"]
        return before - after

    async def consolidate(self) -> ConsolidationReport:
        n = self._conn.execute("SELECT COUNT(*) AS n FROM procedural_skills").fetchone()["n"]
        return ConsolidationReport(
            tier=self.tier_name,
            entries_before=n,
            entries_after=n,
            notes="procedural consolidation (promote frequently-used skills, prune unused) is a stub",
        )

    async def count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) AS n FROM procedural_skills").fetchone()["n"]

    def close(self) -> None:
        self._conn.close()
