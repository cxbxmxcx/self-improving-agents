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
    """Connect to the SQLite archive at `path`. Cached so we don't reconnect."""
    return SQLiteArchive(path)


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
