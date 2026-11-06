"""Metacognition events on helix's EventBus (SPEC §10: the framework emits; consumers render).

Subscribe with `get_bus().subscribe("metacognition", handler)` to watch a
deployment: every run's setting, procedures, and verdict, and every
quarantine, learning attempt, install, proposal, and approval. `webhook(url)`
posts the ones a person may need to act on to a URL, such as a chat channel's.
Events reach the bus of the process that caused them, so give the webhook to
each process: the serving one (quarantines) and the worker (proposals).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import httpx
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


# The events a person may need to act on: not every run, only the changes in trust and learning.
NOTABLE = ("quarantined", "proposed", "installed", "refused", "learning_failed", "skipped", "approved",
           "learner_chosen", "mechanism_proposed")


def webhook(url: str, kinds: Sequence[str] = NOTABLE, *, client: httpx.AsyncClient | None = None
            ) -> Callable[[Any], Any]:
    """Post metacognition events of `kinds` to `url` as JSON. Returns the handler, for unsubscribing.

    A failed post is logged by the bus and never reaches the run that caused the event.
    """

    async def send(event: Any) -> None:
        if getattr(event, "kind", None) not in kinds:
            return
        payload = {key: value for key, value in asdict(event).items()
                   if key in ("kind", "procedure", "version", "run_id", "effort", "offered", "detail")}
        payload["text"] = f"metacog: {event.kind} {event.procedure}" + (
            f"@{event.version}" if event.version is not None else "") + (f": {event.detail}" if event.detail else "")
        if client is not None:
            (await client.post(url, json=payload)).raise_for_status()
            return
        async with httpx.AsyncClient(timeout=10.0) as http:
            (await http.post(url, json=payload)).raise_for_status()

    get_bus().subscribe("metacognition", send)
    return send
