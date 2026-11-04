"""The overview payload: tiles, breakdowns, failure histogram, recent rounds."""

from __future__ import annotations

import sqlite3
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import Response

from helix_web.server import cursor as cursor_mod
from helix_web.server.db import run_ro
from helix_web.server.deps import etag_response, resolve_archive
from helix_web.server.rounds import read_rounds
from helix_web.server.views import (
    annotate,
    classified_trajectories,
    failed_by_type,
    load_variants,
)

router = APIRouter(tags=["summary"])


@router.get("/archives/{key}/summary")
async def summary(
    request: Request,
    key: str,
    focus_id: str | None = None,
    focus_kind: str | None = None,
) -> Response:
    entry = resolve_archive(request, key)
    rounds_rows, _ = read_rounds(entry.path)

    def build(conn: sqlite3.Connection) -> tuple[dict[str, Any], str]:
        all_variants = load_variants(conn)
        annotate(conn, all_variants)
        focused = [
            v
            for v in all_variants
            if (focus_id is None or v["id"] == focus_id)
            and (focus_kind is None or v["kind"] == focus_kind)
        ]
        trajectories = classified_trajectories(conn)
        n_failed = sum(1 for t in trajectories if t["status"] == "failed")
        n_purgeable = sum(1 for t in trajectories if t["purgeable"])
        scores = [
            v["measurement"]["score"]
            for v in all_variants
            if v["measurement"] and v["measurement"]["score"] is not None
        ]

        by_kind: dict[tuple[str, str], int] = {}
        for v in all_variants:
            k = (v["kind"], v["id"])
            by_kind[k] = by_kind.get(k, 0) + 1

        by_creator: dict[str, dict[str, Any]] = {}
        for v in focused:
            slot = by_creator.setdefault(
                v["created_by"], {"count": 0, "scores": []}
            )
            slot["count"] += 1
            m = v["measurement"]
            if m and m["score"] is not None:
                slot["scores"].append(m["score"])

        champion = None
        candidates = [v for v in focused if v.get("is_champion")]
        if candidates:
            champion = max(
                candidates, key=lambda v: v["measurement"]["score"]
            )

        payload = {
            "totals": {
                "artifacts": len(all_variants),
                "measured": sum(1 for v in all_variants if v["measurement"]),
                "top_score": max(scores) if scores else None,
                "rounds": len(rounds_rows),
                "trajectories": len(trajectories),
                "failed": n_failed,
                "purgeable": n_purgeable,
            },
            "by_kind": [
                {"kind": kind, "id": aid, "versions": count}
                for (kind, aid), count in sorted(by_kind.items())
            ],
            "by_creator": [
                {
                    "created_by": creator,
                    "count": slot["count"],
                    "mean_score": (
                        sum(slot["scores"]) / len(slot["scores"])
                        if slot["scores"]
                        else None
                    ),
                    "max_score": max(slot["scores"]) if slot["scores"] else None,
                }
                for creator, slot in sorted(by_creator.items())
            ],
            "failed_by_type": failed_by_type(trajectories),
            "champion": champion,
            "recent_rounds": rounds_rows[-10:],
        }
        return payload, cursor_mod.current(conn, entry.path)

    payload, cur = await run_ro(entry.path, build)
    return etag_response(request, payload, cur)
