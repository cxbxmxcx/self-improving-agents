"""Round history: the jsonl side-channel, else a series derived from the archive."""

from __future__ import annotations

import sqlite3
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import Response

from helix_web.server import cursor as cursor_mod
from helix_web.server.cursor import rounds_file_for
from helix_web.server.db import run_ro
from helix_web.server.deps import etag_response, resolve_archive
from helix_web.server.rounds import read_rounds
from helix_web.server.views import annotate, load_variants

router = APIRouter(tags=["rounds"])


@router.get("/archives/{key}/rounds")
async def rounds(
    request: Request,
    key: str,
    focus_id: str | None = None,
    focus_kind: str | None = None,
) -> Response:
    entry = resolve_archive(request, key)
    jsonl_rows, _ = read_rounds(entry.path)

    def build(conn: sqlite3.Connection) -> tuple[dict[str, Any], str]:
        cur = cursor_mod.current(conn, entry.path)
        if jsonl_rows:
            scores = [
                r["mean_score"] for r in jsonl_rows if r.get("mean_score") is not None
            ]
            payload = {
                "source": "jsonl",
                "source_file": str(rounds_file_for(entry.path)),
                "rounds": jsonl_rows,
                "summary": {
                    "n_rounds": len(jsonl_rows),
                    "n_promotions": sum(1 for r in jsonl_rows if r.get("promoted")),
                    "best_score": max(scores) if scores else None,
                    "mean_score": (sum(scores) / len(scores)) if scores else None,
                },
                "fallback_series": [],
            }
            return payload, cur

        variants = load_variants(conn, focus_id=focus_id, focus_kind=focus_kind)
        annotate(conn, variants)
        series = []
        for v in variants:
            m = v["measurement"]
            series.append(
                {
                    "version": v["version"],
                    "created_by": v["created_by"],
                    "search_method": v["search_method"],
                    "score": m["score"] if m else None,
                    "round": v["variant_metadata"].get("round")
                    or v["artifact_metadata"].get("round"),
                }
            )
        payload = {
            "source": "archive",
            "source_file": None,
            "rounds": [],
            "summary": {
                "n_rounds": 0,
                "n_promotions": 0,
                "best_score": max(
                    (s["score"] for s in series if s["score"] is not None),
                    default=None,
                ),
                "mean_score": None,
            },
            "fallback_series": series,
        }
        return payload, cur

    payload, cur = await run_ro(entry.path, build)
    return etag_response(request, payload, cur)
