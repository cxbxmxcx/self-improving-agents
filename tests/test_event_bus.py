"""EventBus publish/subscribe conformance."""

from __future__ import annotations

import asyncio

import pytest

from helix.observability.bus import EventBus
from helix.observability.events import (
    Event,
    TrajectoryCompleted,
    TrajectoryStarted,
)


@pytest.mark.asyncio
async def test_sync_handler_receives_event():
    bus = EventBus()
    received: list[Event] = []
    bus.subscribe("trajectory_started", lambda e: received.append(e))

    await bus.publish(TrajectoryStarted(trajectory_id="t1", task="hello"))
    assert len(received) == 1
    assert received[0].trajectory_id == "t1"


@pytest.mark.asyncio
async def test_async_handler_is_awaited():
    bus = EventBus()
    received: list[Event] = []

    async def handler(event: Event) -> None:
        await asyncio.sleep(0)
        received.append(event)

    bus.subscribe("trajectory_started", handler)
    await bus.publish(TrajectoryStarted(trajectory_id="t1", task="hello"))
    assert len(received) == 1


@pytest.mark.asyncio
async def test_wildcard_handler_receives_every_event_type():
    bus = EventBus()
    received: list[Event] = []
    bus.subscribe("*", lambda e: received.append(e))

    await bus.publish(TrajectoryStarted(trajectory_id="t1", task="t1"))
    await bus.publish(TrajectoryCompleted(trajectory_id="t1", task="t1", outcome="completed", num_steps=2))
    assert len(received) == 2
    assert {e.event_type for e in received} == {"trajectory_started", "trajectory_completed"}


@pytest.mark.asyncio
async def test_handler_exception_does_not_break_bus():
    bus = EventBus()
    successes: list[Event] = []

    def bad_handler(event: Event) -> None:
        raise RuntimeError("intentional")

    bus.subscribe("trajectory_started", bad_handler)
    bus.subscribe("trajectory_started", lambda e: successes.append(e))

    await bus.publish(TrajectoryStarted(trajectory_id="t1", task="t"))
    assert len(successes) == 1  # second handler still ran


@pytest.mark.asyncio
async def test_unsubscribe_stops_delivery():
    bus = EventBus()
    received: list[Event] = []
    handler = lambda e: received.append(e)  # noqa: E731

    bus.subscribe("trajectory_started", handler)
    await bus.publish(TrajectoryStarted(trajectory_id="t1", task="t"))
    bus.unsubscribe("trajectory_started", handler)
    await bus.publish(TrajectoryStarted(trajectory_id="t2", task="t"))

    assert len(received) == 1
    assert received[0].trajectory_id == "t1"
