"""Streaming runs, shaped like the OpenAI Agents SDK's `Runner.run_streamed`.

    result = Runner.run_streamed(agent, "Weather in Tokyo?")
    async for event in result.stream_events():
        if event.type == "run_item_stream_event":
            print(event.name, event.item)
    print(result.final_output)

Events are step-level, read from helix's hooks as the loop runs: an agent
update first, then `tool_called`, `tool_output`, and `message_output_created`
items. helix calls the model without streaming, so there are no token deltas
(no `raw_response_event`). When the run raises, `stream_events` raises it
after the last event.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

_DONE = object()


@dataclass
class RunItemStreamEvent:
    name: str
    item: Any
    type: str = "run_item_stream_event"


@dataclass
class AgentUpdatedStreamEvent:
    new_agent: Any
    type: str = "agent_updated_stream_event"


class RunResultStreaming:
    """A run in progress. Once `is_complete`, every RunResult field is available on it."""

    def __init__(self, agent: Any, input: Any) -> None:
        self.current_agent = agent
        self.input = input
        self.is_complete = False
        self.queue: asyncio.Queue[Any] = asyncio.Queue()
        self.task: asyncio.Task[Any] | None = None
        self.result: Any = None
        self.error: BaseException | None = None

    async def stream_events(self) -> AsyncIterator[Any]:
        yield AgentUpdatedStreamEvent(self.current_agent)
        while True:
            event = await self.queue.get()
            if event is _DONE:
                break
            yield event
        if self.error is not None:
            raise self.error

    def cancel(self) -> None:
        if self.task is not None and not self.task.done():
            self.task.cancel()

    def finish(self, result: Any = None, error: BaseException | None = None) -> None:
        self.result, self.error, self.is_complete = result, error, True
        self.queue.put_nowait(_DONE)

    def __getattr__(self, name: str) -> Any:
        result = self.__dict__.get("result")
        if result is None:
            raise AttributeError(f"{name} is available once the run completes")
        return getattr(result, name)
