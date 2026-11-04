"""The only module that touches SQLiteArchive: the two write operations.

Promote and purge go through the framework class on purpose, so its events,
validation, and migrations apply exactly as they would in the loops. A lock
serializes writes because the archive connection is not thread-safe.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

_write_lock = asyncio.Lock()


def _promote_sync(
    path: Path, artifact_id: str, version: int, approver: str, reason: str
) -> dict[str, Any]:
    from helix.archive.sqlite_backend import SQLiteArchive

    async def run() -> dict[str, Any]:
        archive = SQLiteArchive(str(path))
        record = await archive.promote(artifact_id, version, approver, reason)
        return {
            "artifact_id": record.artifact_id,
            "version": record.version,
            "approver": record.approver,
            "reason": record.reason,
            "promoted_at": record.promoted_at,
        }

    return asyncio.run(run())


def _purge_sync(path: Path) -> int:
    from helix.archive.sqlite_backend import SQLiteArchive

    async def run() -> int:
        archive = SQLiteArchive(str(path))
        return await archive.purge_failed_trajectories()

    return asyncio.run(run())


async def promote(
    path: Path, artifact_id: str, version: int, approver: str, reason: str
) -> dict[str, Any]:
    async with _write_lock:
        return await asyncio.to_thread(
            _promote_sync, path, artifact_id, version, approver, reason
        )


async def purge_failed(path: Path) -> int:
    async with _write_lock:
        return await asyncio.to_thread(_purge_sync, path)
