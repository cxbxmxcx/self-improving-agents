"""Cached trajectories: index, single replay payload, and the failed view."""

from __future__ import annotations

import json
import sqlite3
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response

from helix_web.server import cursor as cursor_mod
from helix_web.server import queries
from helix_web.server.db import run_ro
from helix_web.server.deps import etag_response, resolve_archive
from helix_web.server.normalize import (
    LIKELY_CAUSE_FALLBACK,
    LIKELY_CAUSES,
    classify_trajectory,
)
from helix_web.server.views import classified_trajectories, failed_by_type

router = APIRouter(tags=["trajectories"])


@router.get("/archives/{key}/trajectories")
async def list_trajectories(
    request: Request,
    key: str,
    artifact_id: str | None = None,
    version: int | None = None,
    question_id: str | None = None,
    status: str | None = None,
    limit: int = 500,
    offset: int = 0,
) -> Response:
    entry = resolve_archive(request, key)

    def build(conn: sqlite3.Connection) -> tuple[dict[str, Any], str]:
        rows = classified_trajectories(conn)
        facets = {
            "artifact_ids": sorted({r["artifact_id"] for r in rows}),
            "versions": sorted({r["version"] for r in rows}),
            "question_ids": sorted({r["question_id"] for r in rows}),
        }
        filtered = [
            r
            for r in rows
            if (artifact_id is None or r["artifact_id"] == artifact_id)
            and (version is None or r["version"] == version)
            and (question_id is None or r["question_id"] == question_id)
            and (status is None or r["status"] == status)
        ]
        payload = {
            "items": filtered[offset : offset + limit],
            "total": len(filtered),
            "facets": facets,
            "n_failed": sum(1 for r in rows if r["status"] == "failed"),
            "n_purgeable": sum(1 for r in rows if r["purgeable"]),
        }
        return payload, cursor_mod.current(conn, entry.path)

    payload, cur = await run_ro(entry.path, build)
    return etag_response(request, payload, cur)


@router.get("/archives/{key}/trajectories/failed")
async def failed(request: Request, key: str) -> Response:
    entry = resolve_archive(request, key)

    def build(conn: sqlite3.Connection) -> tuple[dict[str, Any], str]:
        rows = classified_trajectories(conn)
        failed_rows = [r for r in rows if r["status"] == "failed"]
        payload = {
            "items": failed_rows,
            "by_type": failed_by_type(rows),
            "total": len(failed_rows),
            "n_purgeable": sum(1 for r in rows if r["purgeable"]),
            "cause_fallback": LIKELY_CAUSE_FALLBACK,
            "known_causes": LIKELY_CAUSES,
        }
        return payload, cursor_mod.current(conn, entry.path)

    payload, cur = await run_ro(entry.path, build)
    return etag_response(request, payload, cur)


@router.get("/archives/{key}/trajectories/{artifact_id}/{version}/{question_id}")
async def trajectory_detail(
    request: Request,
    key: str,
    artifact_id: str,
    version: int,
    question_id: str,
    step_offset: int = 0,
    step_limit: int = 200,
) -> Response:
    entry = resolve_archive(request, key)
    settings = request.app.state.settings

    def build(conn: sqlite3.Connection) -> tuple[dict[str, Any] | None, str]:
        cur = cursor_mod.current(conn, entry.path)
        row = queries.fetch_trajectory_row(conn, artifact_id, version, question_id)
        if row is None:
            return None, cur
        try:
            trajectory = json.loads(row["trajectory_json"])
        except (TypeError, ValueError):
            trajectory = {}
        steps = trajectory.get("steps") or []
        page = steps[step_offset : step_offset + step_limit]
        trimmed = []
        truncated_payloads = False
        for s in page:
            text = json.dumps(s, default=str)
            if len(text) > settings.max_content_bytes:
                truncated_payloads = True
                trimmed.append(
                    {
                        "kind": s.get("kind", "unknown"),
                        "payload": {"truncated": True},
                        "timestamp": s.get("timestamp"),
                    }
                )
            else:
                trimmed.append(s)
        info = classify_trajectory(trajectory, row["answer"])
        payload = {
            "artifact_id": artifact_id,
            "version": version,
            "question_id": question_id,
            "recorded_at": row["recorded_at"],
            "answer": row["answer"],
            "status": info["status"],
            "error_type": info["error_type"],
            "error_message": info["error_message"],
            "purgeable": info["purgeable"],
            "task": trajectory.get("task"),
            "outcome": trajectory.get("outcome"),
            "final_output": trajectory.get("final_output"),
            "metadata": trajectory.get("metadata") or {},
            "n_steps": len(steps),
            "step_offset": step_offset,
            "steps": trimmed,
            "steps_truncated": truncated_payloads,
        }
        return payload, cur

    payload, cur = await run_ro(entry.path, build)
    if payload is None:
        raise HTTPException(status_code=404, detail="trajectory not found")
    return etag_response(request, payload, cur)
