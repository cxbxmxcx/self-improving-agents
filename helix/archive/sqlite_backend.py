"""SQLite-backed Archive implementation.

Three tables:
  - artifacts: one row per (artifact_id, version). Stores content, kind, parent
    pointer, created_by, content_hash.
  - measurements: many rows per artifact. Each row is one Signal measurement
    of that artifact, with score / preference / feedback / confidence.
  - variants: bookkeeping for Search.record() — which Search produced which
    artifact, with optional metadata.

In-memory mode (path=":memory:") is the default for tests. File-backed
archives persist across process restarts. SPEC section 5.4.
"""

from __future__ import annotations

import json
import random
import sqlite3
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from helix.artifact import Artifact, ArtifactKind
from helix.archive.archive import DiversityMetrics, SamplingStrategy
from helix.observability.bus import EventBus, get_bus
from helix.observability.events import ArtifactCreated, ArtifactMeasured
from helix.search.base import Variant
from helix.signal import GapMeasurement

if TYPE_CHECKING:
    from helix.trajectory import Trajectory


_SCHEMA = """
CREATE TABLE IF NOT EXISTS artifacts (
    artifact_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    kind TEXT NOT NULL,
    content TEXT NOT NULL,
    content_is_json INTEGER NOT NULL,
    parent_id TEXT,
    parent_version INTEGER,
    created_by TEXT NOT NULL,
    created_at TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    artifact_metadata TEXT NOT NULL,
    PRIMARY KEY (artifact_id, version)
);
CREATE INDEX IF NOT EXISTS idx_artifacts_hash ON artifacts(content_hash);
CREATE INDEX IF NOT EXISTS idx_artifacts_kind ON artifacts(kind);

CREATE TABLE IF NOT EXISTS measurements (
    rowid INTEGER PRIMARY KEY AUTOINCREMENT,
    artifact_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    score REAL,
    preference TEXT NOT NULL,
    feedback TEXT,
    confidence REAL NOT NULL,
    rubric_id TEXT,
    rubric_version INTEGER,
    cost_tokens INTEGER NOT NULL DEFAULT 0,
    cost_dollars REAL NOT NULL DEFAULT 0,
    cost_wall_sec REAL NOT NULL DEFAULT 0,
    measurement_metadata TEXT NOT NULL,
    recorded_at TEXT NOT NULL,
    FOREIGN KEY (artifact_id, version) REFERENCES artifacts(artifact_id, version)
);
CREATE INDEX IF NOT EXISTS idx_meas_artifact ON measurements(artifact_id, version);

CREATE TABLE IF NOT EXISTS variants (
    artifact_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    search_method TEXT NOT NULL,
    variant_metadata TEXT NOT NULL,
    recorded_at TEXT NOT NULL,
    PRIMARY KEY (artifact_id, version)
);

CREATE TABLE IF NOT EXISTS trajectories (
    artifact_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    question_id TEXT NOT NULL,
    trajectory_json TEXT NOT NULL,
    answer TEXT,
    recorded_at TEXT NOT NULL,
    PRIMARY KEY (artifact_id, version, question_id)
);
CREATE INDEX IF NOT EXISTS idx_traj_artifact ON trajectories(artifact_id, version);
"""


class SQLiteArchive:
    """SQLite-backed Archive. Conforms to the Archive Protocol."""

    def __init__(self, path: str | Path = ":memory:", bus: EventBus | None = None) -> None:
        self.path = str(path)
        self._conn = sqlite3.connect(self.path)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        self._bus = bus or get_bus()

    # ---------------- internals ----------------

    def _serialize_content(self, content: str | dict[str, Any]) -> tuple[str, int]:
        if isinstance(content, str):
            return content, 0
        return json.dumps(content, sort_keys=True), 1

    def _row_to_artifact(self, row: sqlite3.Row) -> Artifact:
        content: str | dict[str, Any] = json.loads(row["content"]) if row["content_is_json"] else row["content"]
        parent_id = None
        if row["parent_id"] is not None:
            parent_id = (row["parent_id"], int(row["parent_version"]))
        return Artifact(
            id=row["artifact_id"],
            version=int(row["version"]),
            kind=ArtifactKind(row["kind"]),
            content=content,
            parent_id=parent_id,
            created_by=row["created_by"],
            created_at=datetime.fromisoformat(row["created_at"]),
            metadata=json.loads(row["artifact_metadata"]),
        )

    def _store_artifact(self, art: Artifact) -> bool:
        """Insert an artifact if not already present. Returns True if new."""
        existing = self._conn.execute(
            "SELECT 1 FROM artifacts WHERE artifact_id = ? AND version = ?",
            (art.id, art.version),
        ).fetchone()
        if existing:
            return False
        content_str, is_json = self._serialize_content(art.content)
        parent_id = art.parent_id[0] if art.parent_id else None
        parent_version = art.parent_id[1] if art.parent_id else None
        self._conn.execute(
            """
            INSERT INTO artifacts
            (artifact_id, version, kind, content, content_is_json, parent_id, parent_version,
             created_by, created_at, content_hash, artifact_metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                art.id,
                art.version,
                art.kind.value,
                content_str,
                is_json,
                parent_id,
                parent_version,
                art.created_by,
                art.created_at.isoformat(),
                art.content_hash,
                json.dumps(art.metadata),
            ),
        )
        return True

    def _latest_measurement(self, artifact_id: str, version: int) -> GapMeasurement | None:
        row = self._conn.execute(
            """
            SELECT * FROM measurements
            WHERE artifact_id = ? AND version = ?
            ORDER BY rowid DESC LIMIT 1
            """,
            (artifact_id, version),
        ).fetchone()
        if row is None:
            return None
        from helix.signal import Cost, Preference
        rubric_id = None
        if row["rubric_id"] is not None:
            rubric_id = (row["rubric_id"], int(row["rubric_version"]))
        return GapMeasurement(
            score=row["score"],
            preference=Preference(row["preference"]),
            feedback=row["feedback"],
            confidence=float(row["confidence"]),
            rubric_id=rubric_id,
            cost=Cost(
                tokens=int(row["cost_tokens"]),
                dollars=float(row["cost_dollars"]),
                wall_clock_sec=float(row["cost_wall_sec"]),
            ),
            metadata=json.loads(row["measurement_metadata"]),
        )

    # ---------------- protocol methods ----------------

    async def record(self, variant: Variant, measurement: GapMeasurement) -> None:
        # Track which artifacts were freshly created so we know which to announce.
        parent_was_new = self._store_artifact(variant.parent)
        artifact_was_new = self._store_artifact(variant.artifact)

        rid = measurement.rubric_id[0] if measurement.rubric_id else None
        rv = measurement.rubric_id[1] if measurement.rubric_id else None
        self._conn.execute(
            """
            INSERT INTO measurements
            (artifact_id, version, score, preference, feedback, confidence,
             rubric_id, rubric_version, cost_tokens, cost_dollars, cost_wall_sec,
             measurement_metadata, recorded_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                variant.artifact.id,
                variant.artifact.version,
                measurement.score,
                measurement.preference.value,
                measurement.feedback,
                measurement.confidence,
                rid,
                rv,
                measurement.cost.tokens,
                measurement.cost.dollars,
                measurement.cost.wall_clock_sec,
                json.dumps(measurement.metadata),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        self._conn.execute(
            """
            INSERT OR REPLACE INTO variants
            (artifact_id, version, search_method, variant_metadata, recorded_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                variant.artifact.id,
                variant.artifact.version,
                variant.search_method,
                json.dumps(variant.metadata),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        self._conn.commit()

        # Emit events on the bus so observers (console renderer, drift
        # detection, cost dashboards) see archive activity in real time.
        if artifact_was_new:
            await self._bus.publish(
                ArtifactCreated(
                    artifact_id=variant.artifact.id,
                    version=variant.artifact.version,
                    kind=variant.artifact.kind.value,
                    parent_id=variant.artifact.parent_id[0] if variant.artifact.parent_id else None,
                    parent_version=variant.artifact.parent_id[1] if variant.artifact.parent_id else None,
                    created_by=variant.artifact.created_by,
                )
            )
        await self._bus.publish(
            ArtifactMeasured(
                artifact_id=variant.artifact.id,
                version=variant.artifact.version,
                signal_kind=str(measurement.metadata.get("role", "unknown")),
                score=measurement.score,
                preference=measurement.preference.value,
                confidence=measurement.confidence,
                cost_tokens=measurement.cost.tokens,
                cost_dollars=measurement.cost.dollars,
            )
        )

    async def best(self, k: int = 1, by: str = "score") -> list[Variant]:
        """Return the top-k variants by latest measurement score.

        Ties are broken by latest measurement timestamp (newer wins).
        """
        if by != "score":
            raise NotImplementedError(f"Sorting by '{by}' not yet supported")
        rows = self._conn.execute(
            """
            SELECT a.*, m.score AS latest_score, m.preference AS latest_pref,
                   v.search_method, v.variant_metadata
            FROM artifacts a
            INNER JOIN (
                SELECT artifact_id, version, score, preference,
                       MAX(rowid) AS latest_row
                FROM measurements
                GROUP BY artifact_id, version
            ) m ON a.artifact_id = m.artifact_id AND a.version = m.version
            LEFT JOIN variants v ON a.artifact_id = v.artifact_id AND a.version = v.version
            WHERE m.score IS NOT NULL
            ORDER BY m.score DESC, m.latest_row DESC
            LIMIT ?
            """,
            (k,),
        ).fetchall()

        return [self._row_to_variant(row) for row in rows]

    def _row_to_variant(self, row: sqlite3.Row) -> Variant:
        art = self._row_to_artifact(row)
        parent = art  # placeholder; could fetch parent artifact too
        if art.parent_id:
            parent_row = self._conn.execute(
                "SELECT * FROM artifacts WHERE artifact_id = ? AND version = ?",
                (art.parent_id[0], art.parent_id[1]),
            ).fetchone()
            if parent_row:
                parent = self._row_to_artifact(parent_row)
        measurement = self._latest_measurement(art.id, art.version)
        return Variant(
            artifact=art,
            parent=parent,
            search_method=row["search_method"] if "search_method" in row.keys() and row["search_method"] else "unknown",
            measurement=measurement,
            metadata=json.loads(row["variant_metadata"]) if "variant_metadata" in row.keys() and row["variant_metadata"] else {},
        )

    async def pareto_front(self, objectives: list[str]) -> list[Variant]:
        """Return the Pareto-non-dominated set across the requested objectives.

        Objective names map to metadata keys on the measurement or to "score".
        v1 supports "score" plus any numeric key in measurement.metadata. A
        variant dominates another if it is no worse on all objectives and
        strictly better on at least one.
        """
        all_variants = await self._all_variants_with_measurements()
        if not all_variants:
            return []

        def value(v: Variant, obj: str) -> float | None:
            if v.measurement is None:
                return None
            if obj == "score":
                return v.measurement.score
            return v.measurement.metadata.get(obj)

        front: list[Variant] = []
        for cand in all_variants:
            dominated = False
            for other in all_variants:
                if other is cand:
                    continue
                strictly_better = False
                no_worse = True
                for obj in objectives:
                    c_val = value(cand, obj)
                    o_val = value(other, obj)
                    if c_val is None or o_val is None:
                        no_worse = False
                        break
                    if o_val < c_val:
                        no_worse = False
                        break
                    if o_val > c_val:
                        strictly_better = True
                if no_worse and strictly_better:
                    dominated = True
                    break
            if not dominated:
                front.append(cand)
        return front

    async def _all_variants_with_measurements(self) -> list[Variant]:
        rows = self._conn.execute(
            """
            SELECT a.*, v.search_method, v.variant_metadata
            FROM artifacts a
            LEFT JOIN variants v ON a.artifact_id = v.artifact_id AND a.version = v.version
            """
        ).fetchall()
        return [self._row_to_variant(row) for row in rows]

    async def sample(self, strategy: SamplingStrategy) -> Variant:
        variants = await self._all_variants_with_measurements()
        if not variants:
            raise ValueError("Archive is empty; cannot sample")

        if strategy == SamplingStrategy.RANDOM:
            return random.choice(variants)
        if strategy == SamplingStrategy.BEST:
            best = await self.best(k=1)
            return best[0]
        if strategy == SamplingStrategy.WEIGHTED_BY_SCORE:
            scored = [(v, v.measurement.score) for v in variants if v.measurement and v.measurement.score is not None]
            if not scored:
                return random.choice(variants)
            total = sum(s for _, s in scored)
            if total <= 0:
                return random.choice([v for v, _ in scored])
            return random.choices([v for v, _ in scored], weights=[s for _, s in scored])[0]
        if strategy == SamplingStrategy.QUALITY_DIVERSITY:
            # Toy QD: pick a high-score variant, then prefer one whose content
            # hash differs from the most recent N. Production QD uses behavior
            # characterization functions per artifact kind (SPEC section 5.3).
            top = sorted(
                [v for v in variants if v.measurement and v.measurement.score is not None],
                key=lambda v: v.measurement.score,
                reverse=True,
            )
            seen_hashes: set[str] = set()
            for v in top:
                if v.artifact.content_hash not in seen_hashes:
                    return v
                seen_hashes.add(v.artifact.content_hash)
            return top[0] if top else random.choice(variants)
        raise NotImplementedError(f"Sampling strategy '{strategy}' not implemented")

    async def lineage(self, artifact: Artifact) -> list[Artifact]:
        """Return the ancestor chain from genesis to the given artifact."""
        chain: list[Artifact] = []
        current = artifact
        chain.append(current)
        while current.parent_id is not None:
            row = self._conn.execute(
                "SELECT * FROM artifacts WHERE artifact_id = ? AND version = ?",
                (current.parent_id[0], current.parent_id[1]),
            ).fetchone()
            if row is None:
                break
            current = self._row_to_artifact(row)
            chain.append(current)
        return list(reversed(chain))

    async def descendants(self, artifact: Artifact) -> list[Artifact]:
        rows = self._conn.execute(
            "SELECT * FROM artifacts WHERE parent_id = ? AND parent_version = ?",
            (artifact.id, artifact.version),
        ).fetchall()
        return [self._row_to_artifact(r) for r in rows]

    async def diversity_metrics(self) -> DiversityMetrics:
        rows = self._conn.execute("SELECT content_hash FROM artifacts").fetchall()
        n = len(rows)
        n_unique = len({r["content_hash"] for r in rows})
        scores_row = self._conn.execute(
            "SELECT MIN(score) AS mn, MAX(score) AS mx FROM measurements WHERE score IS NOT NULL"
        ).fetchone()
        score_range = (scores_row["mn"], scores_row["mx"]) if scores_row else (None, None)
        # mean_pairwise_distance is expensive on large archives; v1 returns
        # the simple ratio of unique hashes as a proxy.
        proxy = (n_unique / n) if n else 0.0
        return DiversityMetrics(
            n_variants=n,
            n_unique_content_hashes=n_unique,
            mean_pairwise_distance=proxy,
            score_range=score_range,
        )

    async def by_id(self, artifact_id: str, version: int | None = None) -> Variant | None:
        if version is None:
            row = self._conn.execute(
                """
                SELECT * FROM artifacts
                WHERE artifact_id = ?
                ORDER BY version DESC LIMIT 1
                """,
                (artifact_id,),
            ).fetchone()
        else:
            row = self._conn.execute(
                "SELECT * FROM artifacts WHERE artifact_id = ? AND version = ?",
                (artifact_id, version),
            ).fetchone()
        if row is None:
            return None
        # join variant metadata
        var_row = self._conn.execute(
            "SELECT search_method, variant_metadata FROM variants WHERE artifact_id = ? AND version = ?",
            (row["artifact_id"], row["version"]),
        ).fetchone()
        search_method = var_row["search_method"] if var_row else "unknown"
        variant_metadata = json.loads(var_row["variant_metadata"]) if var_row else {}

        art = self._row_to_artifact(row)
        parent = art
        if art.parent_id:
            p_row = self._conn.execute(
                "SELECT * FROM artifacts WHERE artifact_id = ? AND version = ?",
                (art.parent_id[0], art.parent_id[1]),
            ).fetchone()
            if p_row:
                parent = self._row_to_artifact(p_row)
        measurement = self._latest_measurement(art.id, art.version)
        return Variant(
            artifact=art,
            parent=parent,
            search_method=search_method,
            measurement=measurement,
            metadata=variant_metadata,
        )

    # ---------------- version assignment ----------------

    async def next_version(self, artifact_id: str) -> int:
        """Return the next unused version number for this artifact id.

        Searches use this to avoid (id, version) collisions when producing
        siblings of the same parent. SPEC section 1.1: versions are
        monotonically increasing PER ID across the entire lineage, not just
        per parent. With two Searches both mutating v1, the second mutation
        would collide on v2 without this helper.
        """
        row = self._conn.execute(
            "SELECT MAX(version) AS mv FROM artifacts WHERE artifact_id = ?",
            (artifact_id,),
        ).fetchone()
        max_v = row["mv"] if row and row["mv"] is not None else 0
        return max_v + 1

    # ---------------- dashboard / introspection helpers ----------------

    async def all_artifacts_with_measurements(self) -> list[Variant]:
        """Return every artifact with its latest measurement attached.

        Used by the dashboard to populate tree views, lineage diagrams, and
        comparison dropdowns without N round-trips. Artifacts that have never
        been measured come back with measurement=None.
        """
        rows = self._conn.execute(
            """
            SELECT a.*, v.search_method, v.variant_metadata
            FROM artifacts a
            LEFT JOIN variants v ON a.artifact_id = v.artifact_id AND a.version = v.version
            ORDER BY a.artifact_id, a.version
            """
        ).fetchall()
        return [self._row_to_variant(row) for row in rows]

    async def get_measurement_history(
        self,
        artifact_id: str,
        version: int,
    ) -> list[GapMeasurement]:
        """All measurements ever recorded for one artifact, oldest first.

        Each round of pairwise judging records a fresh measurement against
        whoever was the current best at the time, so an artifact accumulates
        many measurements over its lifetime. The dashboard plots these over
        time to show whether an artifact's perceived quality is rising or
        falling.
        """
        rows = self._conn.execute(
            """
            SELECT * FROM measurements
            WHERE artifact_id = ? AND version = ?
            ORDER BY rowid ASC
            """,
            (artifact_id, version),
        ).fetchall()
        out: list[GapMeasurement] = []
        from helix.signal import Cost, Preference
        for r in rows:
            rubric_id = None
            if r["rubric_id"] is not None:
                rubric_id = (r["rubric_id"], int(r["rubric_version"]))
            out.append(
                GapMeasurement(
                    score=r["score"],
                    preference=Preference(r["preference"]),
                    feedback=r["feedback"],
                    confidence=float(r["confidence"]),
                    rubric_id=rubric_id,
                    cost=Cost(
                        tokens=int(r["cost_tokens"]),
                        dollars=float(r["cost_dollars"]),
                        wall_clock_sec=float(r["cost_wall_sec"]),
                    ),
                    metadata=json.loads(r["measurement_metadata"]),
                )
            )
        return out

    async def lineage_tree(self) -> list[dict]:
        """Return a forest of artifacts as nested {artifact, children, measurement} dicts.

        Roots are artifacts with parent_id IS NULL (genesis artifacts).
        Each node carries its latest measurement so the dashboard can color
        and label nodes without further queries.
        """
        variants = await self.all_artifacts_with_measurements()
        by_ref: dict[tuple[str, int], Variant] = {v.artifact.ref: v for v in variants}
        by_parent: dict[tuple[str, int] | None, list[Variant]] = {}
        for v in variants:
            parent_key = v.artifact.parent_id
            by_parent.setdefault(parent_key, []).append(v)

        def build_node(variant: Variant) -> dict:
            children_variants = by_parent.get(variant.artifact.ref, [])
            return {
                "artifact": variant.artifact,
                "measurement": variant.measurement,
                "search_method": variant.search_method,
                "children": [build_node(c) for c in children_variants],
            }

        roots = by_parent.get(None, [])
        return [build_node(r) for r in roots]

    async def list_question_verdicts(self) -> list[dict]:
        """Flatten per-question verdicts across every measurement in the archive.

        Returns one row per (artifact, question_id) pair with the latest
        preference and confidence. The dashboard pivots / filters this for
        the verdicts table.
        """
        rows = self._conn.execute(
            """
            SELECT a.artifact_id, a.version, a.created_by,
                   m.score, m.preference, m.feedback, m.confidence,
                   m.measurement_metadata, m.recorded_at
            FROM artifacts a
            INNER JOIN measurements m ON a.artifact_id = m.artifact_id AND a.version = m.version
            ORDER BY m.rowid ASC
            """
        ).fetchall()
        out: list[dict] = []
        for r in rows:
            meta = json.loads(r["measurement_metadata"]) or {}
            per_q = meta.get("per_question", [])
            for q in per_q:
                out.append({
                    "artifact_id": r["artifact_id"],
                    "version": r["version"],
                    "created_by": r["created_by"],
                    "recorded_at": r["recorded_at"],
                    "role": meta.get("role", "unknown"),
                    "question_id": q.get("question_id", ""),
                    "band": q.get("band", 0),
                    "preference": q.get("preference", ""),
                    "confidence": q.get("confidence", 0.0),
                    "feedback": q.get("feedback", ""),
                })
        return out

    async def all_cached_trajectories(self) -> list[dict]:
        """List every cached trajectory in the archive for dashboard browsing.

        Returns (artifact_id, version, question_id, answer, recorded_at).
        The trajectory body itself is fetched on demand via get_trajectory().
        """
        rows = self._conn.execute(
            """
            SELECT artifact_id, version, question_id, answer, recorded_at
            FROM trajectories
            ORDER BY recorded_at DESC
            """
        ).fetchall()
        return [
            {
                "artifact_id": r["artifact_id"],
                "version": r["version"],
                "question_id": r["question_id"],
                "answer": r["answer"],
                "recorded_at": r["recorded_at"],
            }
            for r in rows
        ]

    # ---------------- trajectory cache ----------------

    async def store_trajectory(
        self,
        artifact: Artifact,
        question_id: str,
        trajectory: "Trajectory",
        answer: str | None = None,
    ) -> None:
        """Cache a trajectory keyed by (artifact_id, version, question_id).

        Used by the improvement round to avoid re-running an unchanged
        reference artifact against the same question on every round.
        Artifacts are immutable, so the cache key never goes stale.
        """
        traj_dict = trajectory.to_dict() if hasattr(trajectory, "to_dict") else {}
        self._conn.execute(
            """
            INSERT OR REPLACE INTO trajectories
            (artifact_id, version, question_id, trajectory_json, answer, recorded_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                artifact.id,
                artifact.version,
                question_id,
                json.dumps(traj_dict, default=str),
                answer,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        self._conn.commit()

    async def get_trajectory(
        self,
        artifact: Artifact,
        question_id: str,
    ) -> tuple[str | None, "Trajectory"] | None:
        """Return (answer, trajectory) if cached, else None."""
        row = self._conn.execute(
            "SELECT answer, trajectory_json FROM trajectories WHERE artifact_id = ? AND version = ? AND question_id = ?",
            (artifact.id, artifact.version, question_id),
        ).fetchone()
        if row is None:
            return None
        from helix.trajectory import Trajectory
        traj_data = json.loads(row["trajectory_json"])
        traj = Trajectory.from_dict(traj_data) if traj_data else Trajectory(task="")
        return row["answer"], traj

    async def get_trajectory_for_dashboard(
        self,
        artifact_id: str,
        version: int,
        question_id: str,
    ) -> dict | None:
        """Dashboard-friendly: lookup by id/version (no Artifact object needed),
        return a plain dict for Streamlit caching."""
        row = self._conn.execute(
            "SELECT answer, trajectory_json, recorded_at FROM trajectories WHERE artifact_id = ? AND version = ? AND question_id = ?",
            (artifact_id, version, question_id),
        ).fetchone()
        if row is None:
            return None
        return {
            "answer": row["answer"],
            "trajectory": json.loads(row["trajectory_json"]),
            "recorded_at": row["recorded_at"],
        }

    def close(self) -> None:
        self._conn.close()
