"""Shared router dependencies: archive resolution and ETag plumbing."""

from __future__ import annotations

from pathlib import Path

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse, Response

from helix_web.server.registry import ArchiveEntry


def resolve_archive(request: Request, key: str) -> ArchiveEntry:
    """Turn a path key into an ArchiveEntry or the right 404."""
    registry = request.app.state.registry
    try:
        entry = registry.get(key)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"unknown archive: {key}") from exc
    if not Path(entry.path).exists():
        raise HTTPException(
            status_code=404, detail=f"archive file missing: {entry.path}"
        )
    return entry


def etag_response(request: Request, payload: dict, cursor: str) -> Response:
    """JSON with a weak ETag; a matching If-None-Match short-circuits to 304."""
    etag = f'W/"{cursor}"'
    if request.headers.get("if-none-match") == etag:
        return Response(status_code=304, headers={"ETag": etag})
    return JSONResponse(content=payload, headers={"ETag": etag})
