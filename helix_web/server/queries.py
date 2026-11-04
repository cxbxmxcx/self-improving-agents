"""Every SELECT the Observatory runs, as pure functions over a connection.

This module never imports SQLiteArchive; the SQL is ported from
helix/archive/sqlite_backend.py so results match what the framework records.
A test greps for that import ban, because the archive class mutates files on
open and a dashboard read must never do that.
"""

from __future__ import annotations

import sqlite3
from typing import Any

from helix_web.server.db import table_columns

TABLES_WITH_ROWIDS = ("artifacts", "measurements", "trajectories", "promotions")


def fetch_variant_rows(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """Artifacts joined to their variant rows, the dashboard's base relation."""
    return conn.execute(
        """
        SELECT a.*, v.search_method, v.variant_metadata
        FROM artifacts a
        LEFT JOIN variants v
          ON a.artifact_id = v.artifact_id AND a.version = v.version
        ORDER BY a.artifact_id, a.version
        """
    ).fetchall()


def fetch_latest_measurements(
    conn: sqlite3.Connection,
) -> dict[tuple[str, int], sqlite3.Row]:
    """Latest measurement per (artifact_id, version), one query for the lot.

    The Streamlit app ran one lookup per artifact; grouping by MAX(rowid)
    keeps the same highest-rowid-wins semantics in a single pass.
    """
    rows = conn.execute(
        """
        SELECT m.* FROM measurements m
        WHERE m.rowid IN (
            SELECT MAX(rowid) FROM measurements GROUP BY artifact_id, version
        )
        """
    ).fetchall()
    return {(r["artifact_id"], int(r["version"])): r for r in rows}


def fetch_measurement_counts(
    conn: sqlite3.Connection,
) -> dict[tuple[str, int], int]:
    """How many times each version was measured; >1 means the search came back.

    Content-hash dedup re-attaches a re-proposed candidate's measurement to the
    canonical version, so this count is the only visible trace of re-proposals.
    """
    rows = conn.execute(
        "SELECT artifact_id, version, COUNT(*) AS n FROM measurements "
        "GROUP BY artifact_id, version"
    ).fetchall()
    return {(r["artifact_id"], int(r["version"])): int(r["n"]) for r in rows}


def fetch_measurement_history(
    conn: sqlite3.Connection, artifact_id: str, version: int
) -> list[sqlite3.Row]:
    """All measurements for one artifact version, oldest first."""
    return conn.execute(
        """
        SELECT * FROM measurements
        WHERE artifact_id = ? AND version = ?
        ORDER BY rowid ASC
        """,
        (artifact_id, version),
    ).fetchall()


def fetch_artifact_row(
    conn: sqlite3.Connection, artifact_id: str, version: int
) -> sqlite3.Row | None:
    """One artifact row with its variant columns, or None."""
    return conn.execute(
        """
        SELECT a.*, v.search_method, v.variant_metadata
        FROM artifacts a
        LEFT JOIN variants v
          ON a.artifact_id = v.artifact_id AND a.version = v.version
        WHERE a.artifact_id = ? AND a.version = ?
        """,
        (artifact_id, version),
    ).fetchone()


def fetch_verdict_rows(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """Measurement rows joined to artifacts, oldest first, for verdict flattening."""
    return conn.execute(
        """
        SELECT a.artifact_id, a.version, a.created_by,
               m.score, m.preference, m.feedback, m.confidence,
               m.measurement_metadata, m.recorded_at
        FROM artifacts a
        INNER JOIN measurements m
          ON a.artifact_id = m.artifact_id AND a.version = m.version
        ORDER BY m.rowid ASC
        """
    ).fetchall()


def fetch_trajectory_rows(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """Every cached trajectory row, including the JSON payload."""
    return conn.execute(
        """
        SELECT artifact_id, version, question_id, trajectory_json, answer, recorded_at
        FROM trajectories
        ORDER BY artifact_id, version, question_id
        """
    ).fetchall()


def fetch_trajectory_row(
    conn: sqlite3.Connection, artifact_id: str, version: int, question_id: str
) -> sqlite3.Row | None:
    """One cached trajectory, or None."""
    return conn.execute(
        """
        SELECT artifact_id, version, question_id, trajectory_json, answer, recorded_at
        FROM trajectories
        WHERE artifact_id = ? AND version = ? AND question_id = ?
        """,
        (artifact_id, version, question_id),
    ).fetchone()


def fetch_promotions(
    conn: sqlite3.Connection, artifact_id: str | None = None
) -> list[sqlite3.Row]:
    """Promotion log rows, oldest first, optionally scoped to one artifact."""
    if artifact_id is None:
        return conn.execute(
            "SELECT rowid, * FROM promotions ORDER BY rowid ASC"
        ).fetchall()
    return conn.execute(
        "SELECT rowid, * FROM promotions WHERE artifact_id = ? ORDER BY rowid ASC",
        (artifact_id,),
    ).fetchall()


def fetch_live_versions(conn: sqlite3.Connection) -> dict[str, int]:
    """Currently-promoted version per artifact_id (latest promotion wins)."""
    rows = conn.execute(
        """
        SELECT p.artifact_id, p.version FROM promotions p
        WHERE p.rowid IN (
            SELECT MAX(rowid) FROM promotions GROUP BY artifact_id
        )
        """
    ).fetchall()
    return {r["artifact_id"]: int(r["version"]) for r in rows}


def count_summary(conn: sqlite3.Connection) -> dict[str, int]:
    """Cheap whole-archive counts for listings and tiles."""
    n_artifacts = conn.execute("SELECT COUNT(*) FROM artifacts").fetchone()[0]
    n_measured = conn.execute(
        "SELECT COUNT(DISTINCT artifact_id || '|' || version) FROM measurements"
    ).fetchone()[0]
    n_traj = conn.execute("SELECT COUNT(*) FROM trajectories").fetchone()[0]
    n_promos = conn.execute("SELECT COUNT(*) FROM promotions").fetchone()[0]
    return {
        "artifacts": int(n_artifacts),
        "measured": int(n_measured),
        "trajectories": int(n_traj),
        "promotions": int(n_promos),
    }


def watermarks(conn: sqlite3.Connection) -> dict[str, tuple[int, int]]:
    """(max_rowid, count) per table, the raw material for the change cursor.

    Counts ride along because INSERT OR REPLACE and purge make MAX(rowid)
    non-monotonic on its own; a count decrease means a resync, not a diff.
    """
    out: dict[str, tuple[int, int]] = {}
    for table in TABLES_WITH_ROWIDS:
        row = conn.execute(
            f"SELECT COALESCE(MAX(rowid), 0), COUNT(*) FROM {table}"
        ).fetchone()
        out[table] = (int(row[0]), int(row[1]))
    return out


def fetch_rows_after(
    conn: sqlite3.Connection, table: str, after_rowid: int, limit: int
) -> list[sqlite3.Row]:
    """Rows past a watermark, for SSE backfill and poll ticks."""
    if table not in TABLES_WITH_ROWIDS:
        raise ValueError(f"unknown table: {table}")
    return conn.execute(
        f"SELECT rowid AS _rowid, * FROM {table} WHERE rowid > ? ORDER BY rowid ASC LIMIT ?",
        (after_rowid, limit),
    ).fetchall()


def is_helix_archive(conn: sqlite3.Connection) -> bool:
    """True when the file carries the archive's core tables."""
    cols = table_columns(conn, "artifacts")
    return {"artifact_id", "version", "content"}.issubset(cols)


def archive_probe(conn: sqlite3.Connection) -> dict[str, Any]:
    """Counts plus a validity flag, tolerating non-archive sqlite files."""
    if not is_helix_archive(conn):
        return {"valid": False}
    info: dict[str, Any] = {"valid": True}
    info.update(count_summary(conn))
    top = conn.execute(
        """
        SELECT MAX(score) FROM measurements m
        WHERE m.rowid IN (
            SELECT MAX(rowid) FROM measurements GROUP BY artifact_id, version
        )
        """
    ).fetchone()[0]
    info["top_score"] = float(top) if top is not None else None
    return info


def fetch_memory_rows(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """Every version of every memory-entry artifact, ordered by id then version.

    The memory panel folds these into one row per id at the latest version and
    keeps the rest as the version chain, so one query serves both endpoints.
    """
    return conn.execute(
        """
        SELECT * FROM artifacts
        WHERE kind = 'memory_entry'
        ORDER BY artifact_id, version
        """
    ).fetchall()


def fetch_memory_measurements(
    conn: sqlite3.Connection, artifact_id: str | None = None
) -> list[sqlite3.Row]:
    """Measurements on memory-entry artifacts, oldest first.

    Utility updates land here under signal_id 'utility'; the full history is
    what the entry drawer plots, the latest row is what the table shows.
    """
    if artifact_id is None:
        return conn.execute(
            """
            SELECT m.* FROM measurements m
            INNER JOIN artifacts a
              ON a.artifact_id = m.artifact_id AND a.version = m.version
            WHERE a.kind = 'memory_entry'
            ORDER BY m.rowid ASC
            """
        ).fetchall()
    return conn.execute(
        "SELECT * FROM measurements WHERE artifact_id = ? ORDER BY rowid ASC",
        (artifact_id,),
    ).fetchall()
