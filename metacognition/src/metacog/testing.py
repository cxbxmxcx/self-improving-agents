"""A scripted model for tests: queued replies, no network, every request kept.

    model = ScriptedModel([call("get_weather", {"city": "Tokyo"}), text("Sunny.")])
    with model.active():
        result = await Runner.run(agent, "Weather in Tokyo?")
    model.requests[1]["messages"]   # what the agent sent on its second call
"""

from __future__ import annotations

import copy
import json
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from helix.metacognition import recording


@dataclass
class Reply:
    content: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)  # {"id", "name", "arguments"}
    usage: dict[str, int] = field(default_factory=dict)


def text(content: str, **usage: int) -> Reply:
    return Reply(content=content, usage=dict(usage))


def call(name: str, arguments: dict[str, Any] | None = None, *, id: str | None = None, **usage: int) -> Reply:
    return Reply(tool_calls=[{"id": id or f"call_{name}", "name": name, "arguments": json.dumps(arguments or {})}],
                 usage=dict(usage))


class RespondingModel:
    """A model whose reply is a function of the request, for tests that run calls concurrently."""

    def __init__(self, respond: Any) -> None:
        self.respond = respond
        self.requests: list[dict[str, Any]] = []

    async def __call__(self, **kwargs: Any) -> Any:
        self.requests.append(copy.deepcopy(kwargs))
        reply = self.respond(kwargs)
        return recording._response({"content": reply.content, "tool_calls": reply.tool_calls, "usage": reply.usage})

    @contextmanager
    def active(self) -> Iterator[RespondingModel]:
        recording.install()
        token = recording._transport.set(self)
        try:
            yield self
        finally:
            recording._transport.reset(token)


class ScriptedModel:
    def __init__(self, replies: list[Reply]) -> None:
        self.replies = list(replies)
        self.requests: list[dict[str, Any]] = []

    async def __call__(self, **kwargs: Any) -> Any:
        # helix passes its live message list; keep each request as it was sent.
        self.requests.append(copy.deepcopy(kwargs))
        if not self.replies:
            raise AssertionError("ScriptedModel has no reply left for this call")
        reply = self.replies.pop(0)
        return recording._response({"content": reply.content, "tool_calls": reply.tool_calls,
                                    "usage": reply.usage})

    @contextmanager
    def active(self) -> Iterator[ScriptedModel]:
        recording.install()
        token = recording._transport.set(self)
        try:
            yield self
        finally:
            recording._transport.reset(token)
