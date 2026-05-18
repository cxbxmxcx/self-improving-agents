"""The Trajectory primitive.

A Trajectory is the structured record of one agent run. It is the unit memory
learns from, eval scores, reflection summarizes, and behavior diffs compare.
Spec §2.

A messages array is a flattened, model-facing projection of a Trajectory. The
framework can produce messages from a trajectory; it cannot produce a trajectory
from messages, because trajectories carry artifact lineage and hook firings
that don't appear in messages.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

from helix.artifact import ParentRef


class StepKind(str, Enum):
    MODEL_CALL = "model_call"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    HOOK_FIRE = "hook_fire"
    HUMAN_INPUT = "human_input"
    ARTIFACT_LOAD = "artifact_load"


class Outcome(str, Enum):
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"
    AWAITING_HUMAN = "awaiting_human"


@dataclass
class Step:
    index: int
    kind: StepKind
    payload: dict[str, Any]
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


@dataclass
class Trajectory:
    """The mutable record of an in-progress or completed agent run.

    Trajectories are the only mutable primitive in the framework. They grow
    during a run as steps are appended. Once outcome is set to a terminal value
    they should not be modified further; downstream systems (memory, eval) read
    them as immutable.
    """

    id: str = field(default_factory=lambda: str(uuid4()))
    task: str = ""
    steps: list[Step] = field(default_factory=list)
    artifacts_used: dict[int, list[ParentRef]] = field(default_factory=dict)
    final_output: Any | None = None
    outcome: Outcome = Outcome.IN_PROGRESS
    started_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    ended_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def append(self, kind: StepKind, payload: dict[str, Any]) -> Step:
        step = Step(index=len(self.steps), kind=kind, payload=payload)
        self.steps.append(step)
        return step

    def record_artifact(self, step_index: int, ref: tuple[str, int]) -> None:
        """Note which artifact was read at this step, for replay and lineage."""
        self.artifacts_used.setdefault(step_index, []).append(ref)

    def complete(self, output: Any, outcome: Outcome = Outcome.COMPLETED) -> None:
        self.final_output = output
        self.outcome = outcome
        self.ended_at = datetime.now(timezone.utc)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict suitable for JSON Lines storage.

        Enums become their string values; datetimes ISO 8601; artifact refs are
        kept as [id, version] lists because JSON has no tuple type.
        """
        return {
            "id": self.id,
            "task": self.task,
            "steps": [
                {
                    "index": s.index,
                    "kind": s.kind.value,
                    "payload": s.payload,
                    "timestamp": s.timestamp.isoformat(),
                }
                for s in self.steps
            ],
            "artifacts_used": {
                str(idx): [list(ref) if ref else None for ref in refs]
                for idx, refs in self.artifacts_used.items()
            },
            "final_output": self.final_output,
            "outcome": self.outcome.value,
            "started_at": self.started_at.isoformat(),
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Trajectory:
        traj = cls(
            id=data["id"],
            task=data["task"],
            final_output=data.get("final_output"),
            outcome=Outcome(data["outcome"]),
            started_at=datetime.fromisoformat(data["started_at"]),
            ended_at=datetime.fromisoformat(data["ended_at"]) if data.get("ended_at") else None,
            metadata=data.get("metadata", {}),
        )
        traj.steps = [
            Step(
                index=s["index"],
                kind=StepKind(s["kind"]),
                payload=s["payload"],
                timestamp=datetime.fromisoformat(s["timestamp"]),
            )
            for s in data.get("steps", [])
        ]
        traj.artifacts_used = {
            int(idx): [tuple(ref) if ref else None for ref in refs]
            for idx, refs in data.get("artifacts_used", {}).items()
        }
        return traj
