"""Composition helpers shared by the read routers.

queries.py speaks SQL and normalize.py speaks rows; this module glues them
into the payload fragments several endpoints share, like the variant list
with champions attached or the lineage forest with crossover edges.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any

from helix_web.server import queries
from helix_web.server.normalize import (
    LIKELY_CAUSE_FALLBACK,
    LIKELY_CAUSES,
    champion_refs,
    classify_trajectory,
    variant_out,
)


def load_variants(
    conn: sqlite3.Connection,
    *,
    focus_id: str | None = None,
    focus_kind: str | None = None,
    include_content: bool = False,
    max_content_bytes: int | None = None,
) -> list[dict[str, Any]]:
    """Every artifact as API JSON, latest measurement attached, focus applied."""
    latest = queries.fetch_latest_measurements(conn)
    counts = queries.fetch_measurement_counts(conn)
    variants = []
    for row in queries.fetch_variant_rows(conn):
        key = (row["artifact_id"], int(row["version"]))
        v = variant_out(
            row,
            latest.get(key),
            include_content=include_content,
            max_content_bytes=max_content_bytes,
        )
        v["n_measurements"] = counts.get(key, 0)
        if focus_id is not None and v["id"] != focus_id:
            continue
        if focus_kind is not None and v["kind"] != focus_kind:
            continue
        variants.append(v)
    return variants


def annotate(
    conn: sqlite3.Connection, variants: list[dict[str, Any]]
) -> tuple[set[str], dict[str, int]]:
    """Champion refs and live (promoted) versions, stamped onto the variants."""
    champs = champion_refs(variants)
    live = queries.fetch_live_versions(conn)
    for v in variants:
        v["is_champion"] = v["ref"] in champs
        v["live_version"] = live.get(v["id"])
    return champs, live


def build_forest(
    variants: list[dict[str, Any]], max_nodes: int
) -> dict[str, Any]:
    """Nest variants into roots plus dashed crossover edges and stats.

    Roots are parentless artifacts, children group by the parent pointer,
    exactly like the archive's own lineage_tree. Trees larger than max_nodes
    are cut breadth-first and flagged rather than silently trimmed.
    """
    by_ref = {v["ref"]: v for v in variants}
    children: dict[str | None, list[dict[str, Any]]] = {}
    for v in variants:
        parent = v["parent"]
        parent_ref = f"{parent['id']}__v{parent['version']}" if parent else None
        if parent_ref is not None and parent_ref not in by_ref:
            parent_ref = None  # orphan under focus filtering becomes a root
        children.setdefault(parent_ref, []).append(v)

    truncated = len(variants) > max_nodes
    budget = max_nodes
    max_depth = 0

    def attach(node: dict[str, Any], depth: int) -> dict[str, Any]:
        nonlocal budget, max_depth
        max_depth = max(max_depth, depth)
        out = dict(node)
        kids = children.get(node["ref"], [])
        attached = []
        for kid in kids:
            if budget <= 0:
                break
            budget -= 1
            attached.append(attach(kid, depth + 1))
        out["children"] = attached
        return out

    roots = []
    for root in children.get(None, []):
        if budget <= 0:
            break
        budget -= 1
        roots.append(attach(root, 1))

    extra_edges = [
        {
            "from": f"{v['crossover_parent']['id']}__v{v['crossover_parent']['version']}",
            "to": v["ref"],
            "type": "crossover",
        }
        for v in variants
        if v.get("crossover_parent")
    ]
    return {
        "roots": roots,
        "extra_edges": extra_edges,
        "stats": {
            "n_nodes": len(variants),
            "n_roots": len(children.get(None, [])),
            "max_depth": max_depth,
            "truncated": truncated,
        },
    }


def lineage_path(
    variants: list[dict[str, Any]], leaf: dict[str, Any]
) -> list[dict[str, Any]]:
    """Genesis-to-leaf walk along primary parent pointers."""
    by_ref = {v["ref"]: v for v in variants}
    path = []
    node: dict[str, Any] | None = leaf
    seen: set[str] = set()
    while node is not None and node["ref"] not in seen:
        seen.add(node["ref"])
        m = node.get("measurement")
        path.append(
            {
                "version": node["version"],
                "created_by": node["created_by"],
                "score": m.get("score") if m else None,
            }
        )
        parent = node.get("parent")
        node = (
            by_ref.get(f"{parent['id']}__v{parent['version']}") if parent else None
        )
    path.reverse()
    return path


def classified_trajectories(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """Every cached trajectory with its status/purgeable classification."""
    out = []
    for row in queries.fetch_trajectory_rows(conn):
        try:
            trajectory = json.loads(row["trajectory_json"])
        except (TypeError, ValueError):
            trajectory = {}
        info = classify_trajectory(trajectory, row["answer"])
        info.update(
            {
                "artifact_id": row["artifact_id"],
                "version": int(row["version"]),
                "question_id": row["question_id"],
                "recorded_at": row["recorded_at"],
            }
        )
        out.append(info)
    return out


def failed_by_type(trajectories: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Failure histogram with the ported likely-cause prose attached."""
    counts: dict[str, int] = {}
    for t in trajectories:
        if t["status"] == "failed":
            counts[t["error_type"] or "unknown"] = (
                counts.get(t["error_type"] or "unknown", 0) + 1
            )
    return [
        {
            "error_type": etype,
            "count": count,
            "likely_cause": LIKELY_CAUSES.get(etype, LIKELY_CAUSE_FALLBACK),
        }
        for etype, count in sorted(counts.items(), key=lambda kv: -kv[1])
    ]
