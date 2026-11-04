"""Diff endpoints: artifact content and trajectory answers."""

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
from helix_web.server.diffing import diff_payload

router = APIRouter(tags=["diff"])


def _content_str(row: sqlite3.Row) -> str:
    content = row["content"]
    if row["content_is_json"]:
        try:
            return json.dumps(json.loads(content), indent=2, sort_keys=True)
        except (TypeError, ValueError):
            return str(content)
    return str(content)


@router.get("/archives/{key}/diff")
async def content_diff(
    request: Request,
    key: str,
    a_id: str,
    a_version: int,
    b_id: str,
    b_version: int,
) -> Response:
    entry = resolve_archive(request, key)
    settings = request.app.state.settings

    def build(conn: sqlite3.Connection) -> tuple[dict[str, Any] | None, str]:
        cur = cursor_mod.current(conn, entry.path)
        a = queries.fetch_artifact_row(conn, a_id, a_version)
        b = queries.fetch_artifact_row(conn, b_id, b_version)
        if a is None or b is None:
            return None, cur
        payload = diff_payload(
            _content_str(a),
            _content_str(b),
            f"{a_id} v{a_version}",
            f"{b_id} v{b_version}",
            settings.max_diff_lines,
        )
        return payload, cur

    payload, cur = await run_ro(entry.path, build)
    if payload is None:
        raise HTTPException(status_code=404, detail="one or both artifacts not found")
    return etag_response(request, payload, cur)


@router.get("/archives/{key}/diff/trajectory")
async def trajectory_diff(
    request: Request,
    key: str,
    artifact_id: str,
    a_version: int,
    b_version: int,
    question_id: str,
) -> Response:
    entry = resolve_archive(request, key)
    settings = request.app.state.settings

    def final_output(conn: sqlite3.Connection, version: int) -> str | None:
        row = queries.fetch_trajectory_row(conn, artifact_id, version, question_id)
        if row is None:
            return None
        try:
            trajectory = json.loads(row["trajectory_json"])
        except (TypeError, ValueError):
            trajectory = {}
        return str(trajectory.get("final_output") or row["answer"] or "")

    def build(conn: sqlite3.Connection) -> tuple[dict[str, Any] | None, str]:
        cur = cursor_mod.current(conn, entry.path)
        a = final_output(conn, a_version)
        b = final_output(conn, b_version)
        if a is None or b is None:
            return None, cur
        return (
            diff_payload(
                a,
                b,
                f"v{a_version} on {question_id}",
                f"v{b_version} on {question_id}",
                settings.max_diff_lines,
            ),
            cur,
        )

    payload, cur = await run_ro(entry.path, build)
    if payload is None:
        raise HTTPException(status_code=404, detail="trajectory not found")
    return etag_response(request, payload, cur)
