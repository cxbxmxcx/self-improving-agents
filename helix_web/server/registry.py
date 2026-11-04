"""Archive discovery and identity for the Observatory.

Archives are found by scanning the chapter runs directories, and the user can
register any path by hand. Keys are stable slugs of the repo-relative path so
deep links survive restarts; manual entries outside the repo get a hashed key.
"""

from __future__ import annotations

import hashlib
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from helix_web.server.config import Settings
from helix_web.server.db import ro_connect
from helix_web.server.queries import archive_probe


@dataclass
class ArchiveEntry:
    """One known archive: identity and where it came from."""

    key: str
    path: Path
    rel_path: str | None
    label: str
    chapter: str | None
    source: str  # "scan" | "manual"


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", text).strip("-.")


def key_for(path: Path, repo_root: Path) -> str:
    """Stable key: dotted repo-relative path, or stem+hash outside the repo."""
    resolved = path.resolve()
    try:
        rel = resolved.relative_to(repo_root.resolve())
        return _slug(rel.as_posix().removesuffix(".sqlite").replace("/", "."))
    except ValueError:
        digest = hashlib.sha256(str(resolved).lower().encode()).hexdigest()[:8]
        return f"{_slug(resolved.stem)}-{digest}"


class ArchiveRegistry:
    """Known archives, scanned plus manually added, keyed for the API."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._entries: dict[str, ArchiveEntry] = {}

    def discover(self) -> None:
        """Rescan the globs; manual entries survive, vanished scans drop."""
        manual = {k: e for k, e in self._entries.items() if e.source == "manual"}
        self._entries = manual
        root = self._settings.repo_root
        for pattern in self._settings.scan_globs:
            for path in sorted(root.glob(pattern)):
                self._add_entry(path, source="scan")
        for path in self._settings.extra_archives:
            self._add_entry(Path(path), source="manual")

    def _add_entry(self, path: Path, source: str) -> ArchiveEntry:
        resolved = path.resolve()
        key = key_for(resolved, self._settings.repo_root)
        try:
            rel = resolved.relative_to(self._settings.repo_root.resolve()).as_posix()
        except ValueError:
            rel = None
        chapter_match = re.search(r"chapters/(ch\d+)/", rel or "")
        entry = ArchiveEntry(
            key=key,
            path=resolved,
            rel_path=rel,
            label=resolved.stem,
            chapter=chapter_match.group(1) if chapter_match else None,
            source=source,
        )
        self._entries[key] = entry
        return entry

    def add(self, path: str) -> ArchiveEntry:
        """Register a path by hand; raises ValueError when it is unusable."""
        p = Path(path)
        if not p.exists():
            raise ValueError(f"no file at {path}")
        probe = self.describe(self._add_entry(p, source="manual"))
        if not probe.get("valid", False):
            self._entries.pop(probe["key"], None)
            raise ValueError(f"{path} is not a Helix archive")
        return self._entries[probe["key"]]

    def remove(self, key: str) -> None:
        """Drop a manual entry; scanned entries only vanish with their files."""
        entry = self.get(key)
        if entry.source != "manual":
            raise ValueError("only manually added archives can be removed")
        del self._entries[key]

    def get(self, key: str) -> ArchiveEntry:
        return self._entries[key]

    def keys(self) -> list[str]:
        return sorted(self._entries)

    def describe(self, entry: ArchiveEntry) -> dict[str, Any]:
        """Listing row: identity plus best-effort counts from a read probe."""
        info: dict[str, Any] = {
            "key": entry.key,
            "path": str(entry.path),
            "rel_path": entry.rel_path,
            "label": entry.label,
            "chapter": entry.chapter,
            "source": entry.source,
            "exists": entry.path.exists(),
            "size_bytes": None,
            "mtime": None,
            "valid": False,
            "n_artifacts": None,
            "n_measured": None,
            "top_score": None,
        }
        if not info["exists"]:
            return info
        stat = entry.path.stat()
        info["size_bytes"] = stat.st_size
        info["mtime"] = stat.st_mtime
        try:
            conn = ro_connect(entry.path, self._settings.busy_timeout_ms)
            try:
                probe = archive_probe(conn)
            finally:
                conn.close()
        except sqlite3.Error:
            return info
        info["valid"] = probe.get("valid", False)
        if info["valid"]:
            info["n_artifacts"] = probe["artifacts"]
            info["n_measured"] = probe["measured"]
            info["top_score"] = probe["top_score"]
        return info

    def list(self) -> list[dict[str, Any]]:
        return [self.describe(self._entries[k]) for k in self.keys()]
