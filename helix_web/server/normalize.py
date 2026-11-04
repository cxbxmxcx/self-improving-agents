"""Row-to-response mapping: one measurement shape, champions, classification.

The Streamlit app leaked two different measurement JSON shapes to its UI and
disagreed with the backend about what counts as a failed trajectory. This
module is where both problems are fixed once, for every endpoint.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any

from helix.artifact import migrate_kind

from helix_web.server.db import row_get


def ref(artifact_id: str, version: int) -> str:
    """The canonical selector value used across the whole app."""
    return f"{artifact_id}__v{version}"


def parse_ref(value: str) -> tuple[str, int]:
    """Invert ref(); raises ValueError on malformed input."""
    artifact_id, _, tail = value.rpartition("__v")
    if not artifact_id or not tail.isdigit():
        raise ValueError(f"malformed ref: {value!r}")
    return artifact_id, int(tail)


def _load_json(raw: Any) -> dict[str, Any]:
    if raw is None:
        return {}
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def measurement_out(row: sqlite3.Row) -> dict[str, Any]:
    """The one measurement shape every endpoint returns."""
    rubric = None
    if row_get(row, "rubric_id") is not None:
        rubric = [row["rubric_id"], row_get(row, "rubric_version")]
    return {
        "rowid": row_get(row, "rowid") or row_get(row, "_rowid"),
        "score": row_get(row, "score"),
        "preference": row_get(row, "preference", "none"),
        "feedback": row_get(row, "feedback"),
        "confidence": row_get(row, "confidence", 0.0),
        "rubric_id": rubric,
        "signal_id": row_get(row, "signal_id"),
        "signal_version": row_get(row, "signal_version"),
        "triggered": bool(row_get(row, "triggered", 0)),
        "raw_value": row_get(row, "raw_value"),
        "cost": {
            "tokens": row_get(row, "cost_tokens", 0),
            "dollars": row_get(row, "cost_dollars", 0.0),
            "wall_clock_sec": row_get(row, "cost_wall_sec", 0.0),
        },
        "recorded_at": row_get(row, "recorded_at"),
        "metadata": _load_json(row_get(row, "measurement_metadata")),
    }


def resolved_kind(row: sqlite3.Row) -> tuple[str, str | None]:
    """Apply the framework's kind migration, tolerating unknown values."""
    raw_kind = row["kind"]
    raw_subtype = row_get(row, "subtype")
    try:
        kind, subtype = migrate_kind(raw_kind, raw_subtype)
        return kind.value, (subtype.value if subtype else None)
    except ValueError:
        return str(raw_kind), (str(raw_subtype) if raw_subtype else None)


def crossover_parent(
    artifact_id: str, metadata: dict[str, Any]
) -> dict[str, Any] | None:
    """Second parent of a crossover child, when the search recorded one."""
    other_version = metadata.get("other_parent_version")
    if other_version is None:
        return None
    other_id = metadata.get("other_parent_id", artifact_id)
    return {"id": other_id, "version": int(other_version)}


def variant_out(
    row: sqlite3.Row,
    measurement_row: sqlite3.Row | None,
    *,
    include_content: bool = False,
    max_content_bytes: int | None = None,
) -> dict[str, Any]:
    """One artifact (plus variant columns and latest measurement) as API JSON."""
    artifact_id = row["artifact_id"]
    version = int(row["version"])
    kind, subtype = resolved_kind(row)
    metadata = _load_json(row_get(row, "artifact_metadata"))
    parent = None
    if row_get(row, "parent_id") is not None:
        parent = {"id": row["parent_id"], "version": int(row["parent_version"])}
    out: dict[str, Any] = {
        "ref": ref(artifact_id, version),
        "id": artifact_id,
        "version": version,
        "kind": kind,
        "subtype": subtype,
        "created_by": row["created_by"],
        "created_at": row["created_at"],
        "content_hash": row_get(row, "content_hash"),
        "parent": parent,
        "crossover_parent": crossover_parent(artifact_id, metadata),
        "search_method": row_get(row, "search_method") or "unknown",
        "variant_metadata": _load_json(row_get(row, "variant_metadata")),
        "artifact_metadata": metadata,
        "measurement": measurement_out(measurement_row) if measurement_row else None,
    }
    if include_content:
        content = row["content"]
        truncated = False
        if max_content_bytes is not None and isinstance(content, str):
            encoded = content.encode("utf-8")
            if len(encoded) > max_content_bytes:
                content = encoded[:max_content_bytes].decode("utf-8", errors="ignore")
                truncated = True
        if row_get(row, "content_is_json"):
            try:
                content = json.loads(content)
            except (TypeError, ValueError):
                pass
        out["content"] = content
        out["content_is_json"] = bool(row_get(row, "content_is_json"))
        out["content_truncated"] = truncated
    return out


def family_key(variant: dict[str, Any]) -> str:
    """Champion grouping key, matching the Streamlit '(id, kind)' families."""
    return f"{variant['id']}::{variant['kind']}"


def compute_champions(variants: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Best variant per family, by latest-measurement score.

    Only variants with a non-null score compete, exactly like the Streamlit
    helper; hollow archives simply produce no champions. Ties resolve to the
    earliest version so a rerun cannot silently move the crown.
    """
    best: dict[str, dict[str, Any]] = {}
    for v in variants:
        m = v.get("measurement")
        score = m.get("score") if m else None
        if score is None:
            continue
        key = family_key(v)
        cur = best.get(key)
        if cur is None:
            best[key] = v
            continue
        cur_score = cur["measurement"]["score"]
        if score > cur_score or (score == cur_score and v["version"] < cur["version"]):
            best[key] = v
    return best


def champion_refs(variants: list[dict[str, Any]]) -> set[str]:
    """The refs that get the gold crown."""
    return {v["ref"] for v in compute_champions(variants).values()}


# Ported verbatim from the Streamlit overview panel so the prose survives.
LIKELY_CAUSES: dict[str, str] = {
    "RateLimitError": (
        "Hit provider rate limit. The new token-aware rate budget should "
        "prevent re-occurrence; for older entries, purge them."
    ),
    "Timeout": (
        "LLM call took longer than 120s. May indicate provider slowness or "
        "an unusually long context."
    ),
    "JSONDecodeError": (
        "Judge or proposer returned malformed JSON. Often transient; retry "
        "usually fixes."
    ),
    "zero_step": (
        "Agent run exited before any step was recorded. Most common cause: "
        "rate limit on the first call."
    ),
    "failed_outcome": (
        "Trajectory was marked failed without a specific error message."
    ),
    "litellm.RateLimitError": (
        "Hit provider rate limit. The new token-aware rate budget should "
        "prevent re-occurrence; for older entries, purge them."
    ),
}

LIKELY_CAUSE_FALLBACK = "Unclassified. Click into the Replay panel to see specifics."


def classify_trajectory(
    trajectory: dict[str, Any], answer: str | None
) -> dict[str, Any]:
    """Status per the dashboard's display rule AND the backend's purge rule.

    The two predicates disagree on zero-step trajectories that still carry a
    final output; showing both keeps the mismatch visible instead of silent.
    Returns status, error_type, error_message, and purgeable.
    """
    meta = trajectory.get("metadata") or {}
    steps = trajectory.get("steps") or []
    final_output = trajectory.get("final_output")
    outcome = trajectory.get("outcome")
    error = meta.get("error")

    failed = bool(error) or outcome == "failed" or (not steps and not final_output)
    purgeable = (not steps) or outcome == "failed" or ("error" in meta)

    error_type = None
    error_message = None
    if failed:
        if error:
            text = str(error)
            error_type = text.split(":", 1)[0].strip() or "unknown"
            error_message = text[:500]
        elif outcome == "failed":
            error_type = "failed_outcome"
        elif not steps:
            error_type = "zero_step"
            error_message = "(no error message; zero-step trajectory)"
        else:
            error_type = "unknown"

    return {
        "status": "failed" if failed else "ok",
        "error_type": error_type,
        "error_message": error_message,
        "purgeable": purgeable,
        "answer_preview": (answer or "")[:200] or None,
    }
