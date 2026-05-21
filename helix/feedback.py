"""User feedback store.

SQLite-backed log of user reactions to agent responses. Captures:
  - thumbs up / down per trajectory
  - free-text comments
  - session-level outcomes (resolved, escalated, abandoned)
  - implicit signals (follow-up question shortly after, copy-paste of answer,
    session length, time-to-resolution)

The feedback store is read by helix.signals.implicit_feedback.ImplicitFeedbackSignal
to produce a GapMeasurement for a trajectory. The Improver consumes that
signal as part of a CompositeSignal alongside the live-judge and golden-set
signals.

Scope discipline matches the memory tiers: every feedback record carries
session_id, user_id, and org_id so signals can compute scope-specific
satisfaction metrics ("did user X improve over time? did org Y's satisfaction
shift after the last prompt promotion?").
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any


class FeedbackKind(str, Enum):
    THUMBS = "thumbs"            # explicit +1 / -1
    COMMENT = "comment"          # free-text remark on a specific answer
    FOLLOWUP = "followup"        # user asked a clarifying question after
    OUTCOME = "outcome"          # session-level: resolved / escalated / abandoned
    COPY = "copy"                # user copied the answer (positive signal)
    REGENERATE = "regenerate"    # user asked the agent to try again (negative)
    EDIT = "edit"                # user edited the answer / refined the question


class Outcome(str, Enum):
    """Session-level outcomes (different from Trajectory.Outcome)."""
    RESOLVED = "resolved"
    ESCALATED = "escalated"
    ABANDONED = "abandoned"
    UNKNOWN = "unknown"


@dataclass
class FeedbackRecord:
    """One feedback event, persisted to the store."""

    id: str
    trajectory_id: str | None
    session_id: str | None
    user_id: str | None
    org_id: str | None
    kind: FeedbackKind
    value: float                  # -1.0 to 1.0; thumbs = -1/+1, others derived
    comment: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    recorded_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_row(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "trajectory_id": self.trajectory_id,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "org_id": self.org_id,
            "kind": self.kind.value,
            "value": self.value,
            "comment": self.comment,
            "metadata": json.dumps(self.metadata),
            "recorded_at": self.recorded_at.isoformat(),
        }


_SCHEMA = """
CREATE TABLE IF NOT EXISTS feedback (
    id TEXT PRIMARY KEY,
    trajectory_id TEXT,
    session_id TEXT,
    user_id TEXT,
    org_id TEXT,
    kind TEXT NOT NULL,
    value REAL NOT NULL,
    comment TEXT NOT NULL DEFAULT '',
    metadata_json TEXT NOT NULL DEFAULT '{}',
    recorded_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_feedback_trajectory ON feedback(trajectory_id);
CREATE INDEX IF NOT EXISTS idx_feedback_session ON feedback(session_id);
CREATE INDEX IF NOT EXISTS idx_feedback_user ON feedback(user_id);
CREATE INDEX IF NOT EXISTS idx_feedback_recorded ON feedback(recorded_at);
"""


class FeedbackStore:
    """Persistent feedback log.

    Typical usage:
        store = FeedbackStore("data/feedback.sqlite")
        await store.record_thumbs(trajectory_id="t1", value=1, session_id="s1")
        await store.record_outcome(session_id="s1", outcome=Outcome.RESOLVED)
        score = await store.aggregate_for_trajectory("t1")
    """

    def __init__(self, path: str | Path = ":memory:", check_same_thread: bool = True) -> None:
        self.path = str(path)
        self._conn = sqlite3.connect(self.path, check_same_thread=check_same_thread)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # ---------------- write API ----------------

    async def record(self, record: FeedbackRecord) -> str:
        self._conn.execute(
            """
            INSERT INTO feedback
            (id, trajectory_id, session_id, user_id, org_id, kind, value,
             comment, metadata_json, recorded_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.id,
                record.trajectory_id,
                record.session_id,
                record.user_id,
                record.org_id,
                record.kind.value,
                record.value,
                record.comment,
                json.dumps(record.metadata),
                record.recorded_at.isoformat(),
            ),
        )
        self._conn.commit()
        return record.id

    async def record_thumbs(
        self,
        trajectory_id: str,
        value: int,
        session_id: str | None = None,
        user_id: str | None = None,
        org_id: str | None = None,
        comment: str = "",
    ) -> str:
        """Record an explicit thumbs reaction. value must be -1 or +1."""
        if value not in (-1, 1):
            raise ValueError("thumbs value must be -1 or +1")
        return await self.record(FeedbackRecord(
            id=str(uuid.uuid4()),
            trajectory_id=trajectory_id,
            session_id=session_id,
            user_id=user_id,
            org_id=org_id,
            kind=FeedbackKind.THUMBS,
            value=float(value),
            comment=comment,
        ))

    async def record_outcome(
        self,
        session_id: str,
        outcome: Outcome,
        user_id: str | None = None,
        org_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Record the end-of-session outcome. Mapped to a numeric value:
        resolved=+1, escalated=0, abandoned=-1, unknown=0."""
        value_map = {
            Outcome.RESOLVED: 1.0,
            Outcome.ESCALATED: 0.0,
            Outcome.ABANDONED: -1.0,
            Outcome.UNKNOWN: 0.0,
        }
        return await self.record(FeedbackRecord(
            id=str(uuid.uuid4()),
            trajectory_id=None,
            session_id=session_id,
            user_id=user_id,
            org_id=org_id,
            kind=FeedbackKind.OUTCOME,
            value=value_map[outcome],
            metadata={**(metadata or {}), "outcome": outcome.value},
        ))

    async def record_followup(
        self,
        trajectory_id: str,
        seconds_after: float,
        session_id: str | None = None,
        user_id: str | None = None,
    ) -> str:
        """Record that the user asked a follow-up shortly after this answer.

        Short follow-ups (< 60s) are usually negative (the answer didn't
        resolve the question). Longer ones are usually neutral (new
        question). We map this to a value in [-1, 0]: -1 at 0s, 0 at 120s+.
        """
        gap = max(0.0, min(120.0, seconds_after))
        value = -1.0 + (gap / 120.0)
        return await self.record(FeedbackRecord(
            id=str(uuid.uuid4()),
            trajectory_id=trajectory_id,
            session_id=session_id,
            user_id=user_id,
            org_id=None,
            kind=FeedbackKind.FOLLOWUP,
            value=value,
            metadata={"seconds_after": seconds_after},
        ))

    async def record_copy(
        self,
        trajectory_id: str,
        session_id: str | None = None,
        user_id: str | None = None,
    ) -> str:
        """User copied the answer to clipboard — strong positive."""
        return await self.record(FeedbackRecord(
            id=str(uuid.uuid4()),
            trajectory_id=trajectory_id,
            session_id=session_id,
            user_id=user_id,
            org_id=None,
            kind=FeedbackKind.COPY,
            value=0.8,
        ))

    async def record_regenerate(
        self,
        trajectory_id: str,
        session_id: str | None = None,
        user_id: str | None = None,
    ) -> str:
        """User asked the agent to try again — strong negative."""
        return await self.record(FeedbackRecord(
            id=str(uuid.uuid4()),
            trajectory_id=trajectory_id,
            session_id=session_id,
            user_id=user_id,
            org_id=None,
            kind=FeedbackKind.REGENERATE,
            value=-0.8,
        ))

    # ---------------- read API ----------------

    async def aggregate_for_trajectory(self, trajectory_id: str) -> dict[str, Any]:
        """Return mean signal value + per-kind breakdown for one trajectory."""
        rows = self._conn.execute(
            "SELECT kind, value FROM feedback WHERE trajectory_id = ?",
            (trajectory_id,),
        ).fetchall()
        if not rows:
            return {"n": 0, "mean": 0.0, "by_kind": {}}
        by_kind: dict[str, list[float]] = {}
        for r in rows:
            by_kind.setdefault(r["kind"], []).append(float(r["value"]))
        mean = sum(float(r["value"]) for r in rows) / len(rows)
        return {
            "n": len(rows),
            "mean": mean,
            "by_kind": {k: sum(v) / len(v) for k, v in by_kind.items()},
        }

    async def aggregate_for_session(self, session_id: str) -> dict[str, Any]:
        rows = self._conn.execute(
            "SELECT kind, value FROM feedback WHERE session_id = ?",
            (session_id,),
        ).fetchall()
        if not rows:
            return {"n": 0, "mean": 0.0, "by_kind": {}}
        by_kind: dict[str, list[float]] = {}
        for r in rows:
            by_kind.setdefault(r["kind"], []).append(float(r["value"]))
        mean = sum(float(r["value"]) for r in rows) / len(rows)
        return {
            "n": len(rows),
            "mean": mean,
            "by_kind": {k: sum(v) / len(v) for k, v in by_kind.items()},
        }

    async def recent(self, n: int = 50) -> list[FeedbackRecord]:
        rows = self._conn.execute(
            "SELECT * FROM feedback ORDER BY recorded_at DESC LIMIT ?",
            (n,),
        ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def _row_to_record(self, row: sqlite3.Row) -> FeedbackRecord:
        return FeedbackRecord(
            id=row["id"],
            trajectory_id=row["trajectory_id"],
            session_id=row["session_id"],
            user_id=row["user_id"],
            org_id=row["org_id"],
            kind=FeedbackKind(row["kind"]),
            value=float(row["value"]),
            comment=row["comment"],
            metadata=json.loads(row["metadata_json"]),
            recorded_at=datetime.fromisoformat(row["recorded_at"]),
        )

    async def count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) AS n FROM feedback").fetchone()["n"]

    def close(self) -> None:
        self._conn.close()


# ---------------------------------------------------------------------------
# Process-wide singleton (chat UI and signals share the same store)
# ---------------------------------------------------------------------------

_default_store: FeedbackStore | None = None


def get_feedback_store(path: str | Path | None = None) -> FeedbackStore:
    """Return the process-wide FeedbackStore. Tests can pass an explicit path."""
    global _default_store
    if _default_store is None or path is not None:
        _default_store = FeedbackStore(
            path or ":memory:",
            check_same_thread=False,
        )
    return _default_store


def reset_feedback_store_for_tests() -> None:
    global _default_store
    _default_store = None
