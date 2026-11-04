"""The rounds side-channel: spo_rounds.jsonl next to the archive file.

The improvement loops can append one JSON object per round here; the file is
optional and no current chapter script writes it, so the archive-derived
fallback in the rounds router is what most archives will show.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from helix_web.server.cursor import rounds_file_for


def read_rounds(archive_path: Path, offset: int = 0) -> tuple[list[dict[str, Any]], int]:
    """Round rows from the jsonl, skipping bad lines, plus the new offset.

    The offset parameter lets the SSE watcher tail the file incrementally;
    callers that want the whole file pass 0. Returns ([], offset) when the
    file does not exist.
    """
    path = rounds_file_for(archive_path)
    if not path.exists():
        return [], 0
    rows: list[dict[str, Any]] = []
    with path.open("rb") as fh:
        fh.seek(offset)
        data = fh.read()
    for line in data.decode("utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            rows.append(parsed)
    return rows, offset + len(data)
