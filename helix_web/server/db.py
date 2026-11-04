"""Read-only SQLite access for the Observatory.

The framework's SQLiteArchive runs schema migrations on every open, which a
dashboard must never do to someone's archive. Reads therefore go through raw
sqlite3 connections opened read-only via URI, one per request, in a thread.
"""

from __future__ import annotations

import asyncio
import sqlite3
import urllib.parse
from pathlib import Path
from typing import Any, Callable, TypeVar

T = TypeVar("T")


def sqlite_uri(path: Path, mode: str = "ro") -> str:
    """Build a file: URI for sqlite3, safe for Windows paths and spaces."""
    quoted = urllib.parse.quote(path.resolve().as_posix())
    return f"file:{quoted}?mode={mode}"


def ro_connect(path: Path, busy_timeout_ms: int = 2000) -> sqlite3.Connection:
    """Open an archive strictly read-only; the caller owns closing it."""
    conn = sqlite3.connect(sqlite_uri(path), uri=True, timeout=busy_timeout_ms / 1000)
    conn.row_factory = sqlite3.Row
    conn.execute(f"PRAGMA busy_timeout = {int(busy_timeout_ms)}")
    return conn


async def run_ro(
    path: Path, fn: Callable[[sqlite3.Connection], T], busy_timeout_ms: int = 2000
) -> T:
    """Run a read against the archive on a worker thread.

    The connection is opened and closed inside the same thread, which keeps
    sqlite3's threading rules satisfied without check_same_thread tricks.
    Callers get whatever the callback returns.
    """

    def _call() -> T:
        conn = ro_connect(path, busy_timeout_ms)
        try:
            return fn(conn)
        finally:
            conn.close()

    return await asyncio.to_thread(_call)


def ensure_wal(path: Path) -> bool:
    """Flip the archive to WAL journal mode so live reads tolerate a writer.

    WAL is a persistent property of the database file, so one brief writable
    connection here covers every later run of the loops. Returns True when the
    file reports WAL afterwards; failures are swallowed and reported as False.
    """
    try:
        conn = sqlite3.connect(str(path), timeout=2.0)
        try:
            mode = conn.execute("PRAGMA journal_mode=WAL").fetchone()[0]
            return str(mode).lower() == "wal"
        finally:
            conn.close()
    except sqlite3.Error:
        return False


def is_locked_error(exc: BaseException) -> bool:
    """True when an OperationalError is sqlite's database-is-locked signal."""
    return isinstance(exc, sqlite3.OperationalError) and "locked" in str(exc).lower()


def table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    """Column names for a table; empty set when the table is absent."""
    try:
        rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    except sqlite3.Error:
        return set()
    return {r["name"] for r in rows}


def row_get(row: sqlite3.Row, key: str, default: Any = None) -> Any:
    """Tolerant column access, mirroring the archive's own _row_get.

    Old archives may predate a column; reading them read-only means the
    migration never runs, so absent columns simply fall back to the default.
    """
    try:
        return row[key]
    except (IndexError, KeyError):
        return default
