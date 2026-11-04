"""Per-question judge verdicts: filtered table, facets, and the heatmap."""

from __future__ import annotations

import json
import sqlite3
from typing import Any

from fastapi import APIRouter, Query, Request
from fastapi.responses import Response

from helix_web.server import cursor as cursor_mod
from helix_web.server import queries
from helix_web.server.db import run_ro
from helix_web.server.deps import etag_response, resolve_archive
from helix_web.server.views import classified_trajectories

router = APIRouter(tags=["verdicts"])


def _flatten(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """One row per (measurement, question), oldest measurement first."""
    out: list[dict[str, Any]] = []
    for r in queries.fetch_verdict_rows(conn):
        try:
            meta = json.loads(r["measurement_metadata"]) or {}
        except (TypeError, ValueError):
            meta = {}
        for q in meta.get("per_question", []):
            out.append(
                {
                    "artifact_id": r["artifact_id"],
                    "version": int(r["version"]),
                    "created_by": r["created_by"],
                    "recorded_at": r["recorded_at"],
                    "role": meta.get("role", "unknown"),
                    "question_id": q.get("question_id", ""),
                    "band": q.get("band", 0),
                    "preference": q.get("preference", ""),
                    "confidence": q.get("confidence", 0.0),
                    "feedback": q.get("feedback", ""),
                }
            )
    return out


@router.get("/archives/{key}/verdicts")
async def verdicts(
    request: Request,
    key: str,
    focus_id: str | None = None,
    version: list[int] = Query(default=[]),
    band: list[int] = Query(default=[]),
    preference: list[str] = Query(default=[]),
    role: list[str] = Query(default=[]),
    limit: int = 200,
    offset: int = 0,
) -> Response:
    entry = resolve_archive(request, key)

    def build(conn: sqlite3.Connection) -> tuple[dict[str, Any], str]:
        rows = _flatten(conn)
        if focus_id:
            rows = [r for r in rows if r["artifact_id"] == focus_id]

        failed_keys = {
            (t["artifact_id"], t["version"], t["question_id"])
            for t in classified_trajectories(conn)
            if t["status"] == "failed"
        }
        for r in rows:
            r["candidate_failed"] = (
                (r["artifact_id"], r["version"], r["question_id"]) in failed_keys
            )

        facets = {
            "versions": sorted({r["version"] for r in rows}),
            "bands": sorted({r["band"] for r in rows}),
            "preferences": sorted({r["preference"] for r in rows}),
            "roles": sorted({r["role"] for r in rows}),
        }
        filtered = [
            r
            for r in rows
            if (not version or r["version"] in version)
            and (not band or r["band"] in band)
            and (not preference or r["preference"] in preference)
            and (not role or r["role"] in role)
        ]
        counts: dict[str, int] = {}
        for r in filtered:
            counts[r["preference"]] = counts.get(r["preference"], 0) + 1
        payload = {
            "items": filtered[offset : offset + limit],
            "total": len(filtered),
            "grand_total": len(rows),
            "facets": facets,
            "counts_by_preference": counts,
            "n_from_failed": sum(1 for r in rows if r["candidate_failed"]),
        }
        return payload, cursor_mod.current(conn, entry.path)

    payload, cur = await run_ro(entry.path, build)
    return etag_response(request, payload, cur)


@router.get("/archives/{key}/verdicts/heatmap")
async def heatmap(
    request: Request, key: str, focus_id: str | None = None
) -> Response:
    entry = resolve_archive(request, key)

    def build(conn: sqlite3.Connection) -> tuple[dict[str, Any], str]:
        rows = _flatten(conn)
        if focus_id:
            rows = [r for r in rows if r["artifact_id"] == focus_id]
        cells: dict[str, str] = {}
        for r in rows:  # oldest first, so the latest verdict wins each cell
            cells[f"{r['question_id']}|{r['version']}"] = r["preference"]
        payload = {
            "questions": sorted({r["question_id"] for r in rows}),
            "versions": sorted({r["version"] for r in rows}),
            "cells": cells,
        }
        return payload, cursor_mod.current(conn, entry.path)

    payload, cur = await run_ro(entry.path, build)
    return etag_response(request, payload, cur)
