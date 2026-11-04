"""The change cursor: one string that names an archive's current state.

Format: a{max}:{count}|m{max}:{count}|t{max}:{count}|p{max}:{count}|r{bytes}.
It doubles as the SSE event id and the ETag validator, so a promote, a purge,
or a running loop all invalidate client caches through the same mechanism.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from helix_web.server.queries import watermarks

_PREFIXES = {"artifacts": "a", "measurements": "m", "trajectories": "t", "promotions": "p"}


def rounds_file_for(archive_path: Path) -> Path:
    """The rounds side-channel lives next to the archive file.

    Preferred name is per-archive (revenue_dgm_rounds.jsonl beside
    revenue_dgm.sqlite) so archives sharing a directory keep separate logs;
    the legacy shared spo_rounds.jsonl remains as the fallback.
    """
    named = archive_path.parent / f"{archive_path.stem}_rounds.jsonl"
    if named.exists():
        return named
    return archive_path.parent / "spo_rounds.jsonl"


def rounds_offset(archive_path: Path) -> int:
    path = rounds_file_for(archive_path)
    try:
        return path.stat().st_size
    except OSError:
        return 0


def encode(marks: dict[str, tuple[int, int]], rounds_bytes: int) -> str:
    parts = [
        f"{_PREFIXES[table]}{marks[table][0]}:{marks[table][1]}"
        for table in _PREFIXES
    ]
    parts.append(f"r{rounds_bytes}")
    return "|".join(parts)


def decode(cursor: str) -> tuple[dict[str, tuple[int, int]], int] | None:
    """Parse a cursor; None on anything malformed (treated as no cursor)."""
    marks: dict[str, tuple[int, int]] = {}
    rounds_bytes = 0
    try:
        for part in cursor.split("|"):
            prefix, rest = part[0], part[1:]
            if prefix == "r":
                rounds_bytes = int(rest)
                continue
            table = next(t for t, p in _PREFIXES.items() if p == prefix)
            max_str, count_str = rest.split(":")
            marks[table] = (int(max_str), int(count_str))
        if set(marks) != set(_PREFIXES):
            return None
    except (ValueError, IndexError, StopIteration):
        return None
    return marks, rounds_bytes


def current(conn: sqlite3.Connection, archive_path: Path) -> str:
    return encode(watermarks(conn), rounds_offset(archive_path))
