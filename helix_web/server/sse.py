"""SSE framing and the event-stream generator.

Plain StreamingResponse framing; the whole protocol is a dozen lines, so a
dependency would cost more than it saves. The cursor doubles as the event id,
which is what makes Last-Event-ID reconnects line up with backfill.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from pathlib import Path
from typing import Any, AsyncIterator

from helix_web.server import cursor as cursor_mod
from helix_web.server.db import run_ro
from helix_web.server.watcher import ArchiveWatcher, build_events

HEARTBEAT_SEC = 15


def frame(event: dict[str, Any]) -> str:
    lines = []
    if event.get("id"):
        lines.append(f"id: {event['id']}")
    lines.append(f"event: {event['event']}")
    lines.append(f"data: {json.dumps(event.get('data', {}), default=str)}")
    return "\n".join(lines) + "\n\n"


async def backfill_events(
    path: Path, since: str, limit: int, busy_timeout_ms: int
) -> tuple[list[dict[str, Any]], str, bool]:
    """Events between a client's cursor and now, or a resync signal.

    Returns (events, current_cursor, needs_resync). Malformed cursors and
    shrunken tables both degrade to resync rather than a wrong diff.
    """
    decoded = cursor_mod.decode(since)

    def read(conn: sqlite3.Connection) -> tuple[list[dict[str, Any]], dict, bool]:
        from helix_web.server.queries import TABLES_WITH_ROWIDS, watermarks

        marks = watermarks(conn)
        if decoded is None:
            return [], marks, True
        old_marks, _ = decoded
        shrunk = any(
            marks[t][1] < old_marks[t][1] or marks[t][0] < old_marks[t][0]
            for t in TABLES_WITH_ROWIDS
        )
        if shrunk:
            return [], marks, True
        events, overflow = build_events(conn, old_marks, limit)
        return events, marks, overflow

    events, marks, needs_resync = await run_ro(path, read, busy_timeout_ms)
    cursor = cursor_mod.encode(marks, cursor_mod.rounds_offset(path))
    return events, cursor, needs_resync


async def event_stream(
    watcher: ArchiveWatcher,
    archive_info: dict[str, Any],
    since: str | None,
    backfill_limit: int,
    max_events: int | None = None,
) -> AsyncIterator[str]:
    """The SSE body: hello, optional backfill, then live events + heartbeats.

    max_events exists for tests, which need the stream to end; production
    clients keep it unset and the stream runs until the client disconnects.
    """
    queue = await watcher.subscribe()
    sent = 0
    try:
        yield "retry: 3000\n\n"
        cursor = None
        if since:
            events, cursor, needs_resync = await backfill_events(
                watcher.path, since, backfill_limit, watcher.busy_timeout_ms
            )
            hello = {
                "event": "hello",
                "data": {**archive_info, "cursor": cursor, "wal": watcher.wal},
                "id": cursor,
            }
            yield frame(hello)
            if needs_resync:
                yield frame(
                    {"event": "resync", "data": {"reason": "cursor_gap", "cursor": cursor}, "id": cursor}
                )
            else:
                for ev in events:
                    ev["id"] = cursor
                    yield frame(ev)
                    sent += 1
        else:
            def read_cursor(conn: sqlite3.Connection) -> str:
                return cursor_mod.current(conn, watcher.path)

            cursor = await run_ro(watcher.path, read_cursor, watcher.busy_timeout_ms)
            yield frame(
                {
                    "event": "hello",
                    "data": {**archive_info, "cursor": cursor, "wal": watcher.wal},
                    "id": cursor,
                }
            )

        while max_events is None or sent < max_events:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=HEARTBEAT_SEC)
            except asyncio.TimeoutError:
                yield ": ping\n\n"
                continue
            yield frame(event)
            sent += 1
    finally:
        watcher.unsubscribe(queue)
