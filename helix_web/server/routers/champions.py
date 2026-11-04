"""Champion state per artifact family: best vs live, lineage path, history."""

from __future__ import annotations

import sqlite3
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import Response

from helix_web.server import cursor as cursor_mod
from helix_web.server import queries
from helix_web.server.db import run_ro
from helix_web.server.deps import etag_response, resolve_archive
from helix_web.server.normalize import compute_champions, family_key
from helix_web.server.views import annotate, lineage_path, load_variants

router = APIRouter(tags=["champions"])


def _top_wins(measurement: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Highest-confidence LEFT verdicts from the latest measurement."""
    if not measurement:
        return []
    per_question = measurement.get("metadata", {}).get("per_question", [])
    wins = [q for q in per_question if q.get("preference") == "left"]
    wins.sort(key=lambda q: q.get("confidence", 0.0), reverse=True)
    return [
        {
            "question_id": q.get("question_id", ""),
            "band": q.get("band", 0),
            "confidence": q.get("confidence", 0.0),
            "feedback": q.get("feedback", ""),
        }
        for q in wins[:5]
    ]


@router.get("/archives/{key}/champions")
async def champions(
    request: Request,
    key: str,
    focus_id: str | None = None,
    focus_kind: str | None = None,
) -> Response:
    entry = resolve_archive(request, key)
    settings = request.app.state.settings

    def build(conn: sqlite3.Connection) -> tuple[dict[str, Any], str]:
        variants = load_variants(
            conn,
            focus_id=focus_id,
            focus_kind=focus_kind,
            include_content=True,
            max_content_bytes=settings.max_content_bytes,
        )
        _, live = annotate(conn, variants)
        best_per_family = compute_champions(variants)
        promo_rows = queries.fetch_promotions(conn)

        families = []
        for fkey in sorted(best_per_family):
            best = best_per_family[fkey]
            family = [v for v in variants if family_key(v) == fkey]
            live_version = live.get(best["id"])
            promotions = [
                {
                    "version": int(p["version"]),
                    "approver": p["approver"],
                    "reason": p["reason"],
                    "promoted_at": p["promoted_at"],
                }
                for p in promo_rows
                if p["artifact_id"] == best["id"]
            ]
            families.append(
                {
                    "family": {
                        "id": best["id"],
                        "kind": best["kind"],
                        "subtype": best["subtype"],
                        "label": fkey,
                    },
                    "best": best,
                    "live_version": live_version,
                    "pending": live_version != best["version"],
                    "lineage_path": lineage_path(family, best),
                    "family_size": len(family),
                    "family_tokens": sum(
                        v["measurement"]["cost"]["tokens"]
                        for v in family
                        if v["measurement"]
                    ),
                    "n_measurements": len(
                        queries.fetch_measurement_history(
                            conn, best["id"], best["version"]
                        )
                    ),
                    "confidence": (
                        best["measurement"]["confidence"]
                        if best["measurement"]
                        else None
                    ),
                    "top_wins": _top_wins(best["measurement"]),
                    "promotions": promotions,
                }
            )
        return {"families": families}, cursor_mod.current(conn, entry.path)

    payload, cur = await run_ro(entry.path, build)
    return etag_response(request, payload, cur)
