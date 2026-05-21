"""Archive-reading helpers wrapped in Streamlit caching.

Every function here is `@st.cache_data` so a page render doesn't hit the
SQLite archive multiple times. Cache TTL is 30 seconds; the sidebar's
'Refresh' button clears the cache to force a re-read after a new run.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import streamlit as st

from helix.archive import SQLiteArchive
from helix.artifact import Artifact
from helix.search.base import Variant
from helix.signal import GapMeasurement
from helix.trajectory import Trajectory


# ---------------------------------------------------------------------------
# Async-to-sync bridge for Streamlit
# ---------------------------------------------------------------------------

def _run(coro):
    """Run an async coroutine from Streamlit's sync context.

    Streamlit doesn't expose an event loop; each call creates a fresh loop
    via asyncio.run(). Archive helpers are all short-lived so the per-call
    overhead is negligible compared to render time.
    """
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Archive accessors
# ---------------------------------------------------------------------------

@st.cache_resource
def open_archive_cached(path: str) -> SQLiteArchive:
    """Connect to the SQLite archive at `path`. Cached so we don't reconnect.

    Streamlit reruns pages on different threads; pass check_same_thread=False
    so a single archive handle can serve every render. Read-only from the
    dashboard's perspective; concurrent writes would still be unsafe.
    """
    return SQLiteArchive(path, check_same_thread=False)


@st.cache_data(ttl=30)
def list_variants(archive_path: str) -> list[dict]:
    """Every artifact with its latest measurement, as plain dicts for caching."""
    arc = open_archive_cached(archive_path)
    variants = _run(arc.all_artifacts_with_measurements())
    return [_variant_to_dict(v) for v in variants]


@st.cache_data(ttl=30)
def get_lineage_tree(archive_path: str) -> list[dict]:
    arc = open_archive_cached(archive_path)
    tree = _run(arc.lineage_tree())
    return [_tree_node_to_dict(node) for node in tree]


@st.cache_data(ttl=30)
def list_question_verdicts(archive_path: str) -> list[dict]:
    arc = open_archive_cached(archive_path)
    return _run(arc.list_question_verdicts())


@st.cache_data(ttl=30)
def list_cached_trajectories(archive_path: str) -> list[dict]:
    arc = open_archive_cached(archive_path)
    return _run(arc.all_cached_trajectories())


@st.cache_data(ttl=30)
def get_measurement_history(archive_path: str, artifact_id: str, version: int) -> list[dict]:
    arc = open_archive_cached(archive_path)
    history = _run(arc.get_measurement_history(artifact_id, version))
    return [m.to_dict() for m in history]


@st.cache_data(ttl=30)
def get_artifact(archive_path: str, artifact_id: str, version: int) -> dict | None:
    arc = open_archive_cached(archive_path)
    variant = _run(arc.by_id(artifact_id, version))
    if variant is None:
        return None
    return _variant_to_dict(variant)


@st.cache_data(ttl=30)
def get_trajectory(archive_path: str, artifact_id: str, version: int, question_id: str) -> dict | None:
    """Fetch a cached trajectory and return it as a dict."""
    arc = open_archive_cached(archive_path)
    result = _run(arc.get_trajectory_for_dashboard(artifact_id, version, question_id))
    return result


@st.cache_data(ttl=30)
def get_best(archive_path: str, k: int = 1) -> list[dict]:
    arc = open_archive_cached(archive_path)
    top = _run(arc.best(k=k))
    return [_variant_to_dict(v) for v in top]


@st.cache_data(ttl=30)
def get_live_champion(archive_path: str, artifact_id: str) -> dict | None:
    """Return the currently-promoted version of artifact_id, or None if no
    promotion has been recorded.

    Distinct from get_best (which returns the highest-scoring candidate).
    The live champion is what the running agent serves; offline improvers'
    high-scoring candidates appear in get_best but not here until a human
    promotes them. DESIGN_NOTES.md section 10.
    """
    arc = open_archive_cached(archive_path)
    art = _run(arc.live_champion(artifact_id))
    if art is None:
        return None
    return {
        "id": art.id,
        "version": art.version,
        "kind": art.kind.value,
        "content": art.content,
        "created_by": art.created_by,
        "content_hash": art.content_hash,
    }


@st.cache_data(ttl=30)
def get_promotion_history(archive_path: str, artifact_id: str) -> list[dict]:
    """Return all promotion rows for artifact_id, oldest first.

    Each row carries (version, approver, reason, promoted_at). Rollbacks
    appear as ordinary promotion rows pointing at an earlier version.
    """
    arc = open_archive_cached(archive_path)
    recs = _run(arc.promotion_history(artifact_id))
    return [
        {
            "version": r.version,
            "approver": r.approver,
            "reason": r.reason,
            "promoted_at": r.promoted_at,
        }
        for r in recs
    ]


def promote_artifact(
    archive_path: str,
    artifact_id: str,
    version: int,
    approver: str,
    reason: str,
) -> None:
    """Wrapper around archive.promote() that also clears the cache so the
    dashboard refreshes on rerun. Not @st.cache_data because it's a write.
    """
    arc = open_archive_cached(archive_path)
    _run(arc.promote(artifact_id, version, approver, reason))
    # Invalidate caches that depend on promotion state
    get_live_champion.clear()
    get_promotion_history.clear()


@st.cache_data(ttl=30)
def list_round_log(archive_path: str) -> list[dict]:
    """Read the per-round JSONL log if it exists alongside the archive."""
    archive_dir = Path(archive_path).parent
    log_path = archive_dir / "spo_rounds.jsonl"
    if not log_path.exists():
        return []
    rows: list[dict] = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


@st.cache_data(ttl=30)
def get_champions_per_family(archive_path: str) -> dict[str, dict]:
    """Return {focus_key: variant_dict} mapping each (id, kind) family to its
    champion. Champion = artifact with the highest latest score within that
    family. Returns an empty dict for families with no measured artifacts.

    The focus_key matches the format used by the artifact-focus sidebar:
    `<id> (<kind>)`.
    """
    variants = list_variants(archive_path)
    by_family: dict[tuple[str, str], list[dict]] = {}
    for v in variants:
        if v["measurement"] and v["measurement"].get("score") is not None:
            by_family.setdefault((v["id"], v["kind"]), []).append(v)

    out: dict[str, dict] = {}
    for (aid, kind), candidates in by_family.items():
        champ = max(candidates, key=lambda x: x["measurement"]["score"])
        out[f"{aid} ({kind})"] = champ
    return out


def get_champion_refs(archive_path: str) -> set[tuple[str, int]]:
    """Return {(artifact_id, version), ...} of every per-family champion.

    Used by the Lineage panel to give champion nodes special styling.
    """
    champs = get_champions_per_family(archive_path)
    return {(c["id"], c["version"]) for c in champs.values()}


@st.cache_data(ttl=30)
def list_artifact_focus_options(archive_path: str) -> list[str]:
    """Return unique `<id> (<kind>)` strings for the focus selector.

    Helix supports multiple artifact types under improvement in parallel
    (prompt, skill, tool_description, rubric, ...). The dashboard surfaces
    every (artifact_id, kind) pair so a user can scope panels to one
    artifact at a time.
    """
    variants = list_variants(archive_path)
    seen: set[tuple[str, str]] = set()
    for v in variants:
        seen.add((v["id"], v["kind"]))
    return [f"{aid} ({kind})" for aid, kind in sorted(seen)]


def parse_focus(focus: str | None) -> tuple[str, str] | None:
    """Convert a focus selector string back into (artifact_id, kind)."""
    if not focus:
        return None
    if "(" in focus and focus.endswith(")"):
        aid, rest = focus.split("(", 1)
        return aid.strip(), rest[:-1].strip()
    return None


def filter_variants_by_focus(variants: list[dict], focus: str | None) -> list[dict]:
    """Filter a variant list to only artifacts matching the focus."""
    parsed = parse_focus(focus)
    if parsed is None:
        return variants
    aid, kind = parsed
    return [v for v in variants if v["id"] == aid and v["kind"] == kind]


def purge_failed_trajectories(archive_path: str) -> int:
    """Delete zero-step / errored trajectories from the cache. Returns count."""
    arc = open_archive_cached(archive_path)
    n = _run(arc.purge_failed_trajectories())
    clear_caches()
    return n


@st.cache_data(ttl=30)
def get_failed_trajectories(archive_path: str) -> list[dict]:
    """Every cached trajectory that errored, with parsed error type.

    Returns rows with: artifact_id, version, question_id, recorded_at,
    error_type (parsed from metadata.error), error_message.
    """
    import json as _json
    arc = open_archive_cached(archive_path)
    rows = arc._conn.execute(
        "SELECT artifact_id, version, question_id, trajectory_json, recorded_at FROM trajectories"
    ).fetchall()
    out: list[dict] = []
    for r in rows:
        try:
            tj = _json.loads(r["trajectory_json"])
        except Exception:
            continue
        steps = tj.get("steps", [])
        outcome = tj.get("outcome", "")
        meta = tj.get("metadata", {}) or {}
        err = meta.get("error")
        if err or outcome == "failed" or (not steps and not tj.get("final_output")):
            err_type = "unknown"
            if err and ":" in err:
                err_type = err.split(":", 1)[0].strip()
            elif outcome == "failed":
                err_type = "failed_outcome"
            elif not steps:
                err_type = "zero_step"
            out.append({
                "artifact_id": r["artifact_id"],
                "version": int(r["version"]),
                "question_id": r["question_id"],
                "recorded_at": r["recorded_at"],
                "error_type": err_type,
                "error_message": (err or "(no error message; zero-step trajectory)")[:500],
            })
    return out


@st.cache_data(ttl=30)
def count_failed_trajectories(archive_path: str) -> int:
    return len(get_failed_trajectories(archive_path))


def clear_caches() -> None:
    """Drop all Streamlit caches; called by the sidebar Refresh button."""
    list_variants.clear()
    get_lineage_tree.clear()
    list_question_verdicts.clear()
    list_cached_trajectories.clear()
    get_measurement_history.clear()
    get_artifact.clear()
    get_trajectory.clear()
    get_best.clear()
    list_round_log.clear()
    list_artifact_focus_options.clear()
    get_failed_trajectories.clear()
    count_failed_trajectories.clear()
    get_champions_per_family.clear()


# ---------------------------------------------------------------------------
# Plain-dict conversions for Streamlit caching
# ---------------------------------------------------------------------------

def _variant_to_dict(v: Variant) -> dict:
    art = v.artifact
    return {
        "id": art.id,
        "version": art.version,
        "kind": art.kind.value,
        "content": art.content,
        "content_hash": art.content_hash,
        "parent_id": art.parent_id[0] if art.parent_id else None,
        "parent_version": art.parent_id[1] if art.parent_id else None,
        "created_by": art.created_by,
        "created_at": art.created_at.isoformat(),
        "artifact_metadata": art.metadata,
        "search_method": v.search_method,
        "variant_metadata": v.metadata,
        "measurement": _measurement_to_dict(v.measurement) if v.measurement else None,
    }


def _measurement_to_dict(m: GapMeasurement) -> dict:
    return {
        "score": m.score,
        "preference": m.preference.value,
        "feedback": m.feedback,
        "confidence": m.confidence,
        "cost_tokens": m.cost.tokens,
        "cost_dollars": m.cost.dollars,
        "metadata": m.metadata,
    }


def _tree_node_to_dict(node: dict) -> dict:
    return {
        "id": node["artifact"].id,
        "version": node["artifact"].version,
        "kind": node["artifact"].kind.value,
        "content": node["artifact"].content,
        "created_by": node["artifact"].created_by,
        "search_method": node["search_method"],
        "measurement": _measurement_to_dict(node["measurement"]) if node["measurement"] else None,
        "children": [_tree_node_to_dict(c) for c in node["children"]],
    }
