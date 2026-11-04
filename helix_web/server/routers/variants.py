"""Variant listing, detail, and measurement history."""

from __future__ import annotations

import sqlite3
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response

from helix_web.server import cursor as cursor_mod
from helix_web.server import queries
from helix_web.server.db import run_ro
from helix_web.server.deps import etag_response, resolve_archive
from helix_web.server.normalize import measurement_out, ref
from helix_web.server.views import annotate, load_variants

router = APIRouter(tags=["variants"])


@router.get("/archives/{key}/variants")
async def list_variants(
    request: Request,
    key: str,
    focus_id: str | None = None,
    focus_kind: str | None = None,
    include_content: bool = False,
    limit: int = 500,
    offset: int = 0,
) -> Response:
    entry = resolve_archive(request, key)
    settings = request.app.state.settings

    def build(conn: sqlite3.Connection) -> tuple[dict[str, Any], str]:
        variants = load_variants(
            conn,
            focus_id=focus_id,
            focus_kind=focus_kind,
            include_content=include_content,
            max_content_bytes=settings.max_content_bytes,
        )
        annotate(conn, variants)
        page = variants[offset : offset + limit]
        return {"items": page, "total": len(variants)}, cursor_mod.current(
            conn, entry.path
        )

    payload, cur = await run_ro(entry.path, build)
    return etag_response(request, payload, cur)


@router.get("/archives/{key}/variants/{artifact_id}/{version}")
async def variant_detail(
    request: Request,
    key: str,
    artifact_id: str,
    version: int,
) -> Response:
    entry = resolve_archive(request, key)
    settings = request.app.state.settings

    def build(conn: sqlite3.Connection) -> tuple[dict[str, Any] | None, str]:
        cur = cursor_mod.current(conn, entry.path)
        variants = load_variants(
            conn, focus_id=artifact_id, include_content=False
        )
        annotate(conn, variants)
        target = next(
            (v for v in variants if v["version"] == version), None
        )
        if target is None:
            return None, cur
        row = queries.fetch_artifact_row(conn, artifact_id, version)
        detail = dict(target)
        content = row["content"]
        encoded = content.encode("utf-8") if isinstance(content, str) else b""
        truncated = len(encoded) > settings.max_content_bytes
        if truncated:
            content = encoded[: settings.max_content_bytes].decode(
                "utf-8", errors="ignore"
            )
        detail["content"] = content
        detail["content_is_json"] = bool(row["content_is_json"])
        detail["content_truncated"] = truncated
        detail["measurement_history"] = [
            measurement_out(m)
            for m in queries.fetch_measurement_history(conn, artifact_id, version)
        ]
        best = max(
            (
                v
                for v in variants
                if v["measurement"] and v["measurement"]["score"] is not None
            ),
            key=lambda v: v["measurement"]["score"],
            default=None,
        )
        detail["is_best"] = best is not None and best["ref"] == ref(
            artifact_id, version
        )
        return detail, cur

    payload, cur = await run_ro(entry.path, build)
    if payload is None:
        raise HTTPException(
            status_code=404, detail=f"no artifact {artifact_id} v{version}"
        )
    return etag_response(request, payload, cur)


@router.get("/archives/{key}/variants/{artifact_id}/{version}/measurements")
async def measurement_history(
    request: Request, key: str, artifact_id: str, version: int
) -> Response:
    entry = resolve_archive(request, key)

    def build(conn: sqlite3.Connection) -> tuple[list[dict[str, Any]], str]:
        rows = queries.fetch_measurement_history(conn, artifact_id, version)
        return [measurement_out(r) for r in rows], cursor_mod.current(
            conn, entry.path
        )

    items, cur = await run_ro(entry.path, build)
    return etag_response(request, {"items": items}, cur)
