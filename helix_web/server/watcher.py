"""Per-archive change watcher: poll watermarks, fan events out to queues.

The event bus is in-process to the loops, so a separate dashboard watches the
SQLite file itself. Each tick diffs (max_rowid, count) per table against the
last tick; growth becomes typed events and anything non-monotonic becomes a
resync, which tells clients to refetch rather than trust an incremental diff.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from helix_web.server import cursor as cursor_mod
from helix_web.server import queries
from helix_web.server.db import ensure_wal, run_ro
from helix_web.server.normalize import champion_refs, classify_trajectory, variant_out
from helix_web.server.rounds import read_rounds
from helix_web.server.views import load_variants

QUEUE_MAX = 500


@dataclass
class _Tick:
    marks: dict[str, tuple[int, int]]
    rounds_offset: int
    champions: set[str] = field(default_factory=set)


def _snapshot(conn: sqlite3.Connection) -> dict[str, Any]:
    variants = load_variants(conn)
    return {
        "marks": queries.watermarks(conn),
        "champions": champion_refs(variants),
    }


def _rows_between(
    conn: sqlite3.Connection, marks: dict[str, tuple[int, int]], limit: int
) -> dict[str, list[sqlite3.Row]]:
    return {
        table: queries.fetch_rows_after(conn, table, marks[table][0], limit)
        for table in queries.TABLES_WITH_ROWIDS
    }


def build_events(
    conn: sqlite3.Connection,
    since: dict[str, tuple[int, int]],
    limit: int,
) -> tuple[list[dict[str, Any]], bool]:
    """Typed events for every row past the since-watermarks.

    Returns (events, overflowed); on overflow the caller should resync
    instead of pretending the capped list is complete.
    """
    rows = _rows_between(conn, since, limit + 1)
    overflowed = any(len(v) > limit for v in rows.values())
    if overflowed:
        return [], True

    latest = queries.fetch_latest_measurements(conn)
    events: list[dict[str, Any]] = []
    for r in rows["artifacts"]:
        v = variant_out(r, latest.get((r["artifact_id"], int(r["version"]))))
        events.append({"event": "artifact", "data": {"node": v}})
    for r in rows["measurements"]:
        events.append(
            {
                "event": "measurement",
                "data": {
                    "artifact_id": r["artifact_id"],
                    "version": int(r["version"]),
                    "score": r["score"],
                    "preference": r["preference"],
                    "feedback": r["feedback"],
                    "recorded_at": r["recorded_at"],
                },
            }
        )
    for r in rows["trajectories"]:
        try:
            trajectory = json.loads(r["trajectory_json"])
        except (TypeError, ValueError):
            trajectory = {}
        info = classify_trajectory(trajectory, r["answer"])
        events.append(
            {
                "event": "trajectory",
                "data": {
                    "artifact_id": r["artifact_id"],
                    "version": int(r["version"]),
                    "question_id": r["question_id"],
                    "recorded_at": r["recorded_at"],
                    "status": info["status"],
                    "error_type": info["error_type"],
                },
            }
        )
    for r in rows["promotions"]:
        events.append(
            {
                "event": "promotion",
                "data": {
                    "artifact_id": r["artifact_id"],
                    "version": int(r["version"]),
                    "approver": r["approver"],
                    "reason": r["reason"],
                    "promoted_at": r["promoted_at"],
                },
            }
        )
    return events, False


class ArchiveWatcher:
    """One poll loop per archive, shared by every SSE subscriber."""

    def __init__(self, path: Path, poll_ms: int, busy_timeout_ms: int) -> None:
        self.path = path
        self.poll_ms = poll_ms
        self.busy_timeout_ms = busy_timeout_ms
        self.subscribers: set[asyncio.Queue] = set()
        self.wal = False
        self._task: asyncio.Task | None = None
        self._last: _Tick | None = None
        self._stop_grace: asyncio.Task | None = None

    async def subscribe(self) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue(maxsize=QUEUE_MAX)
        self.subscribers.add(queue)
        if self._stop_grace is not None:
            self._stop_grace.cancel()
            self._stop_grace = None
        if self._task is None:
            self.wal = await asyncio.to_thread(ensure_wal, self.path)
            self._task = asyncio.create_task(self._run())
        return queue

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        self.subscribers.discard(queue)
        if not self.subscribers and self._task is not None:
            # Linger a few seconds so a panel switch does not thrash the loop.
            self._stop_grace = asyncio.create_task(self._linger())

    async def _linger(self) -> None:
        try:
            await asyncio.sleep(5)
        except asyncio.CancelledError:
            return
        if not self.subscribers and self._task is not None:
            self._task.cancel()
            self._task = None
            self._last = None

    def _publish(self, event: dict[str, Any]) -> None:
        for queue in list(self.subscribers):
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                # A slow consumer loses the incremental stream, not the truth:
                # drain it and leave a single resync marker.
                while not queue.empty():
                    try:
                        queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                queue.put_nowait({"event": "resync", "data": {"reason": "queue_overflow"}})

    async def _run(self) -> None:
        while True:
            try:
                await self._tick()
            except asyncio.CancelledError:
                raise
            except sqlite3.Error:
                pass  # transient lock or mid-write state; next tick retries
            await asyncio.sleep(self.poll_ms / 1000)

    async def _tick(self) -> None:
        def read(conn: sqlite3.Connection) -> dict[str, Any]:
            snap = _snapshot(conn)
            events: list[dict[str, Any]] = []
            resync = False
            if self._last is not None:
                shrunk = any(
                    snap["marks"][t][1] < self._last.marks[t][1]
                    or snap["marks"][t][0] < self._last.marks[t][0]
                    for t in queries.TABLES_WITH_ROWIDS
                )
                if shrunk:
                    resync = True
                elif snap["marks"] != self._last.marks:
                    events, overflow = build_events(conn, self._last.marks, 200)
                    resync = overflow
            return {"snap": snap, "events": events, "resync": resync}

        result = await run_ro(self.path, read, self.busy_timeout_ms)
        snap = result["snap"]
        rounds_offset = cursor_mod.rounds_offset(self.path)
        cursor = cursor_mod.encode(snap["marks"], rounds_offset)

        first = self._last is None
        if not first:
            if result["resync"]:
                self._publish({"event": "resync", "data": {"reason": "rows_changed", "cursor": cursor}, "id": cursor})
            else:
                for ev in result["events"]:
                    ev["id"] = cursor
                    self._publish(ev)
                if snap["champions"] != self._last.champions and not result["resync"]:
                    self._publish(
                        {
                            "event": "champions",
                            "data": {"champions": sorted(snap["champions"])},
                            "id": cursor,
                        }
                    )
                new_rounds, new_offset = read_rounds(self.path, self._last.rounds_offset)
                for row in new_rounds:
                    self._publish({"event": "round", "data": {"row": row}, "id": cursor})
                rounds_offset = max(rounds_offset, new_offset)

        self._last = _Tick(
            marks=snap["marks"],
            rounds_offset=rounds_offset,
            champions=snap["champions"],
        )


class WatcherPool:
    """Registry of watchers, one per archive key."""

    def __init__(self, poll_ms: int, busy_timeout_ms: int) -> None:
        self.poll_ms = poll_ms
        self.busy_timeout_ms = busy_timeout_ms
        self._watchers: dict[str, ArchiveWatcher] = {}

    def get(self, key: str, path: Path) -> ArchiveWatcher:
        if key not in self._watchers:
            self._watchers[key] = ArchiveWatcher(
                path, self.poll_ms, self.busy_timeout_ms
            )
        return self._watchers[key]
