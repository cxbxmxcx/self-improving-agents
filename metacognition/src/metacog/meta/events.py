"""Metacognition events on helix's EventBus (SPEC §10: the framework emits; consumers render).

Subscribe with `get_bus().subscribe("metacognition", handler)` to watch a
deployment: every run's setting, procedures, and verdict, and every
quarantine, learning attempt, install, proposal, and approval.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from helix.observability.bus import get_bus
from helix.observability.events import Event


@dataclass
class MetacognitionEvent(Event):
    event_type: str = "metacognition"
    # "run", "quarantined", "installed", "proposed", "refused", "learning_failed", "approved"
    kind: str = ""
    procedure: str = ""
    version: int | None = None
    run_id: str | None = None
    effort: str | None = None
    offered: list[str] = field(default_factory=list)
    detail: str = ""


def emit(**fields: object) -> None:
    """Publish from synchronous code (scheduled as a task when a loop is running)."""
    get_bus().publish_sync(MetacognitionEvent(**fields))  # type: ignore[arg-type]


async def publish(**fields: object) -> None:
    await get_bus().publish(MetacognitionEvent(**fields))  # type: ignore[arg-type]
