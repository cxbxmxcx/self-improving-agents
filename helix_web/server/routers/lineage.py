"""The lineage forest: nested nodes, crossover edges, champions, stats.

Also home to the cross-archive compare endpoint, the showdown view that puts
SPO, GEPA, and DGM runs side by side on one score domain.
"""

from __future__ import annotations

import sqlite3
from typing import Any, Iterator

from fastapi import APIRouter, Request
from fastapi.responses import Response

from helix_web.server import cursor as cursor_mod
from helix_web.server.db import run_ro
from helix_web.server.deps import etag_response, resolve_archive
from helix_web.server.views import annotate, build_forest, load_variants

router = APIRouter(tags=["lineage"])


@router.get("/lineage-compare")
async def lineage_compare(
    request: Request,
    keys: str,
    focus_id: str | None = None,
    max_nodes: int | None = None,
) -> Response:
    """The showdown view: one forest per archive, on a shared score domain."""
    settings = request.app.state.settings
    cap = max_nodes or settings.max_nodes
    key_list = [k.strip() for k in keys.split(",") if k.strip()][:6]

    archives = []
    scores: list[float] = []
    cursors = []
    for key in key_list:
        entry = resolve_archive(request, key)

        def build(conn: sqlite3.Connection, entry=entry) -> tuple[dict[str, Any], str]:
            variants = load_variants(conn, focus_id=focus_id)
            champs, _ = annotate(conn, variants)
            forest = build_forest(variants, cap)
            forest["champions"] = sorted(champs)
            return forest, cursor_mod.current(conn, entry.path)

        forest, cur = await run_ro(entry.path, build)
        cursors.append(cur)
        for v in _walk_scores(forest["roots"]):
            scores.append(v)
        archives.append(
            {
                "key": entry.key,
                "label": entry.label,
                "chapter": entry.chapter,
                "forest": forest,
            }
        )
    payload = {
        "archives": archives,
        "score_domain": [min(scores), max(scores)] if scores else None,
    }
    return etag_response(request, payload, "+".join(cursors))


def _walk_scores(roots: list[dict[str, Any]]) -> Iterator[float]:
    stack = list(roots)
    while stack:
        node = stack.pop()
        m = node.get("measurement")
        if m and m.get("score") is not None:
            yield m["score"]
        stack.extend(node.get("children", []))


@router.get("/archives/{key}/lineage")
async def lineage(
    request: Request,
    key: str,
    focus_id: str | None = None,
    focus_kind: str | None = None,
    max_nodes: int | None = None,
) -> Response:
    entry = resolve_archive(request, key)
    settings = request.app.state.settings
    cap = max_nodes or settings.max_nodes

    def build(conn: sqlite3.Connection) -> tuple[dict[str, Any], str]:
        variants = load_variants(conn, focus_id=focus_id, focus_kind=focus_kind)
        champs, _ = annotate(conn, variants)
        forest = build_forest(variants, cap)
        forest["champions"] = sorted(champs)
        return forest, cursor_mod.current(conn, entry.path)

    payload, cur = await run_ro(entry.path, build)
    return etag_response(request, payload, cur)
