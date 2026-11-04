"""Settings for the Helix Observatory server.

One plain dataclass holds every tunable so tests can build an app against a
temp directory without touching module globals. The repo root is derived from
this file's location because the repo is deliberately not a Python package.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


@dataclass
class Settings:
    """Runtime configuration; defaults match the repo's chapter layout."""

    repo_root: Path = REPO_ROOT
    scan_globs: tuple[str, ...] = ("chapters/*/runs/*.sqlite",)
    extra_archives: list[Path] = field(default_factory=list)
    poll_ms: int = 500
    busy_timeout_ms: int = 2000
    max_content_bytes: int = 200_000
    max_diff_lines: int = 4000
    max_nodes: int = 2000
    max_backfill_rows: int = 500
    web_dir: Path = REPO_ROOT / "helix_web" / "web"
