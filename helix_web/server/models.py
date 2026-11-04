"""Pydantic response models for the stable public shapes.

Only the shapes clients program against live here; endpoint-local payloads
stay as plain dicts until they prove stable. MeasurementOut is the single
measurement shape for the whole app, replacing Streamlit's two variants.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class CostOut(BaseModel):
    tokens: int = 0
    dollars: float = 0.0
    wall_clock_sec: float = 0.0


class MeasurementOut(BaseModel):
    rowid: int | None = None
    score: float | None = None
    preference: str = "none"
    feedback: str | None = None
    confidence: float = 0.0
    rubric_id: list[Any] | None = None
    signal_id: str | None = None
    signal_version: int | None = None
    triggered: bool = False
    raw_value: float | None = None
    cost: CostOut = CostOut()
    recorded_at: str | None = None
    metadata: dict[str, Any] = {}


class ParentRefOut(BaseModel):
    id: str
    version: int


class VariantOut(BaseModel):
    ref: str
    id: str
    version: int
    kind: str
    subtype: str | None = None
    created_by: str
    created_at: str
    content_hash: str | None = None
    parent: ParentRefOut | None = None
    crossover_parent: ParentRefOut | None = None
    search_method: str = "unknown"
    variant_metadata: dict[str, Any] = {}
    artifact_metadata: dict[str, Any] = {}
    measurement: MeasurementOut | None = None
    content: Any | None = None
    content_is_json: bool | None = None
    content_truncated: bool | None = None


class ArchiveInfo(BaseModel):
    key: str
    path: str
    rel_path: str | None = None
    label: str
    chapter: str | None = None
    source: str
    exists: bool
    valid: bool
    size_bytes: int | None = None
    mtime: float | None = None
    n_artifacts: int | None = None
    n_measured: int | None = None
    top_score: float | None = None


class ArchiveList(BaseModel):
    archives: list[ArchiveInfo]
