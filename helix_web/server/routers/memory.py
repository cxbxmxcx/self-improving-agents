"""Memory entries as artifacts: the ranked store and one entry's lineage.

Chapter 5 turns every memory entry into a MEMORY_ENTRY artifact whose content
carries its utility. This router reads those artifacts back as the tiers see
them: one row per entry at its latest version, ranked by utility, with the
version chain and utility history available per entry.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response

from helix_web.server import cursor as cursor_mod
from helix_web.server import queries
from helix_web.server.db import row_get, run_ro
from helix_web.server.deps import etag_response, resolve_archive
from helix_web.server.normalize import measurement_out

router = APIRouter(tags=["memory"])

PREVIEW_KEYS = ("insight", "user_message", "summary", "task", "description")


def _load(raw: Any) -> Any:
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return raw


def _content(row: sqlite3.Row) -> dict[str, Any]:
    content = _load(row["content"]) if row_get(row, "content_is_json") else row["content"]
    return content if isinstance(content, dict) else {}


def _meta(row: sqlite3.Row) -> dict[str, Any]:
    meta = _load(row_get(row, "artifact_metadata"))
    return meta if isinstance(meta, dict) else {}


def preview_of(entry: dict[str, Any], limit: int = 160) -> str:
    """A one-line summary of an entry payload, tier-agnostic.

    Episodic entries carry an insight or user message, semantic ones a key and
    value, procedural ones a name; the first that exists wins.
    """
    for key in PREVIEW_KEYS:
        value = entry.get(key)
        if value:
            return str(value)[:limit]
    if entry.get("key") is not None:
        return f"{entry['key']}: {entry.get('value', '')}"[:limit]
    if entry.get("name"):
        return str(entry["name"])[:limit]
    return json.dumps(entry, default=str)[:limit] if entry else ""


def _utility(content: dict[str, Any]) -> float | None:
    try:
        return float(content["utility"]) if "utility" in content else None
    except (TypeError, ValueError):
        return None


def _version_out(row: sqlite3.Row) -> dict[str, Any]:
    content = _content(row)
    meta = _meta(row)
    parent = None
    if row_get(row, "parent_id") is not None:
        parent = {"id": row["parent_id"], "version": int(row["parent_version"])}
    return {
        "version": int(row["version"]),
        "operator": meta.get("operator"),
        "utility": _utility(content),
        "created_by": row["created_by"],
        "created_at": row["created_at"],
        "parent": parent,
        "content": content,
        "metadata": meta,
    }


def _chains(conn: sqlite3.Connection) -> dict[str, list[dict[str, Any]]]:
    chains: dict[str, list[dict[str, Any]]] = {}
    for row in queries.fetch_memory_rows(conn):
        chains.setdefault(row["artifact_id"], []).append(_version_out(row))
    return chains


def _entry_row(
    artifact_id: str,
    chain: list[dict[str, Any]],
    latest_measurement: dict[str, Any] | None,
) -> dict[str, Any]:
    latest = chain[-1]
    content = latest["content"]
    meta = latest["metadata"]
    entry = content.get("entry") if isinstance(content.get("entry"), dict) else {}
    return {
        "artifact_id": artifact_id,
        "entry_id": meta.get("entry_id") or artifact_id.rsplit(".", 1)[-1],
        "tier": content.get("tier") or meta.get("tier"),
        "scope": content.get("scope") or meta.get("scope"),
        "version": latest["version"],
        "n_versions": len(chain),
        "utility": latest["utility"],
        "operator": latest["operator"],
        "created_by": latest["created_by"],
        "created_at": latest["created_at"],
        "parent": latest["parent"],
        "latest_measurement": latest_measurement,
        "preview": preview_of(entry),
    }


@router.get("/archives/{key}/memory")
async def list_memory(
    request: Request,
    key: str,
    tier: str | None = None,
    scope: str | None = None,
    limit: int = 500,
    offset: int = 0,
) -> Response:
    entry = resolve_archive(request, key)

    def build(conn: sqlite3.Connection) -> tuple[dict[str, Any], str]:
        chains = _chains(conn)
        latest_meas: dict[str, dict[str, Any]] = {}
        for m in queries.fetch_memory_measurements(conn):
            latest_meas[m["artifact_id"]] = measurement_out(m)
        rows = [
            _entry_row(aid, chain, latest_meas.get(aid))
            for aid, chain in chains.items()
        ]
        tiers: dict[str, int] = {}
        scopes: dict[str, int] = {}
        for r in rows:
            tiers[str(r["tier"])] = tiers.get(str(r["tier"]), 0) + 1
            scopes[str(r["scope"])] = scopes.get(str(r["scope"]), 0) + 1
        filtered = [
            r
            for r in rows
            if (not tier or r["tier"] == tier) and (not scope or r["scope"] == scope)
        ]
        # Utility descending, then newest first so ties read in arrival order.
        filtered.sort(
            key=lambda r: (
                r["utility"] if r["utility"] is not None else float("-inf"),
                r["created_at"] or "",
            ),
            reverse=True,
        )
        payload = {
            "items": filtered[offset : offset + limit],
            "total": len(filtered),
            "grand_total": len(rows),
            "facets": {"tiers": tiers, "scopes": scopes},
        }
        return payload, cursor_mod.current(conn, entry.path)

    payload, cur = await run_ro(entry.path, build)
    return etag_response(request, payload, cur)


@router.get("/archives/{key}/memory/{artifact_id}")
async def memory_detail(request: Request, key: str, artifact_id: str) -> Response:
    entry = resolve_archive(request, key)

    def build(conn: sqlite3.Connection) -> tuple[dict[str, Any] | None, str]:
        cur = cursor_mod.current(conn, entry.path)
        chain = _chains(conn).get(artifact_id)
        if not chain:
            return None, cur
        history = [
            measurement_out(m)
            for m in queries.fetch_memory_measurements(conn, artifact_id)
        ]
        detail = _entry_row(artifact_id, chain, history[-1] if history else None)
        detail["content"] = chain[-1]["content"]
        detail["metadata"] = chain[-1]["metadata"]
        detail["versions"] = [
            {k: v for k, v in version.items() if k not in ("content", "metadata")}
            for version in chain
        ]
        detail["utility_history"] = history
        return detail, cur

    payload, cur = await run_ro(entry.path, build)
    if payload is None:
        raise HTTPException(status_code=404, detail=f"no memory entry {artifact_id}")
    return etag_response(request, payload, cur)
