"""Typed event bus.

Subscribers register handlers per event_type (or "*" for all events). Publish
fan-outs to every matching handler. Handlers run synchronously in the order
they registered; async handlers are awaited individually. Handler exceptions
are caught and logged so one bad subscriber doesn't break the bus.

This is the in-process event bus. The same event types serialize to JSON for
cross-process consumption: an Improver running as a separate service consumes
the same event types off a queue. SPEC section 10.3.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections import defaultdict
from collections.abc import Awaitable, Callable
from typing import Any, Union

from helix.observability.events import Event

_log = logging.getLogger("helix.eventbus")

HandlerFn = Union[
    Callable[[Event], None],
    Callable[[Event], Awaitable[None]],
]


class EventBus:
    """In-process event distribution.

    Use the singleton `get_bus()` unless you need test isolation.
    """

    def __init__(self) -> None:
        self._handlers: dict[str, list[HandlerFn]] = defaultdict(list)

    def subscribe(self, event_type: str, handler: HandlerFn) -> None:
        """Register a handler for a specific event_type or "*" for all events."""
        self._handlers[event_type].append(handler)

    def unsubscribe(self, event_type: str, handler: HandlerFn) -> None:
        if handler in self._handlers[event_type]:
            self._handlers[event_type].remove(handler)

    async def publish(self, event: Event) -> None:
        """Distribute an event to every matching handler.

        Sync handlers run inline; async handlers are awaited. Handler errors
        are caught and logged but never propagate to the publisher.
        """
        matched: list[HandlerFn] = []
        matched.extend(self._handlers.get(event.event_type, []))
        matched.extend(self._handlers.get("*", []))

        for h in matched:
            try:
                result = h(event)
                if inspect.isawaitable(result):
                    await result
            except Exception as exc:  # noqa: BLE001
                _log.exception("event handler %r failed for %s: %s", h, event.event_type, exc)

    def publish_sync(self, event: Event) -> None:
        """Synchronous publish for non-async callers.

        If we're in an event loop, schedule the publish as a task. Otherwise
        run it to completion.
        """
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self.publish(event))
        except RuntimeError:
            asyncio.run(self.publish(event))


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

_default_bus: EventBus | None = None


def get_bus() -> EventBus:
    """Return the default process-wide EventBus.

    Tests that need isolation should instantiate their own EventBus().
    """
    global _default_bus
    if _default_bus is None:
        _default_bus = EventBus()
    return _default_bus


def reset_bus_for_tests() -> None:
    """Drop the default bus; next get_bus() rebuilds a fresh one."""
    global _default_bus
    _default_bus = None
