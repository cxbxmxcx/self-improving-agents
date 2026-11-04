"""The live event stream for one archive."""

from __future__ import annotations

from fastapi import APIRouter, Header, Request
from fastapi.responses import StreamingResponse

from helix_web.server.deps import resolve_archive
from helix_web.server.sse import event_stream
from helix_web.server.watcher import WatcherPool

router = APIRouter(tags=["events"])


def _pool(request: Request) -> WatcherPool:
    if not hasattr(request.app.state, "watcher_pool"):
        settings = request.app.state.settings
        request.app.state.watcher_pool = WatcherPool(
            settings.poll_ms, settings.busy_timeout_ms
        )
    return request.app.state.watcher_pool


@router.get("/archives/{key}/events")
async def events(
    request: Request,
    key: str,
    since: str | None = None,
    max_events: int | None = None,
    last_event_id: str | None = Header(default=None, alias="Last-Event-ID"),
) -> StreamingResponse:
    entry = resolve_archive(request, key)
    settings = request.app.state.settings
    watcher = _pool(request).get(key, entry.path)
    info = {
        "archive": {"key": entry.key, "label": entry.label, "path": str(entry.path)},
        "poll_ms": settings.poll_ms,
    }
    stream = event_stream(
        watcher,
        info,
        since or last_event_id,
        settings.max_backfill_rows,
        max_events=max_events,
    )
    return StreamingResponse(
        stream,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
        },
    )
