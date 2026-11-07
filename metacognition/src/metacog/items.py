"""What a run produced, read back from its helix trajectory.

Items follow the OpenAI Agents SDK's names (message output, tool call, tool
call output); messages are chat-completions dicts, the format helix sends.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from helix.trajectory import Step, StepKind, Trajectory


@dataclass
class ModelResponse:
    """One model call: what came back, what it used, and how it was asked."""

    content: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)  # {"id", "name", "arguments"}
    usage: dict[str, int] = field(default_factory=dict)
    model: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    seconds: float = 0.0


@dataclass
class MessageOutputItem:
    agent: Any
    content: str
    type: str = "message_output_item"


@dataclass
class ToolCallItem:
    agent: Any
    name: str
    arguments: dict[str, Any]
    call_id: str
    type: str = "tool_call_item"


@dataclass
class ToolCallOutputItem:
    agent: Any
    call_id: str | None
    output: Any
    type: str = "tool_call_output_item"


RunItem = MessageOutputItem | ToolCallItem | ToolCallOutputItem


def _arguments(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    try:
        return json.loads(raw or "{}")
    except (TypeError, ValueError):
        return {"_raw": raw}


def model_response(step: Step) -> ModelResponse:
    entry = step.payload.get("response") or {}
    calls = [{"id": tc.get("id"), "name": tc.get("function", {}).get("name"),
              "arguments": tc.get("function", {}).get("arguments")} for tc in entry.get("tool_calls") or []]
    return ModelResponse(content=entry.get("content") or "", tool_calls=calls,
                         usage=dict(step.payload.get("usage") or {}), model=step.payload.get("model") or "",
                         params=dict(step.payload.get("params") or {}),
                         seconds=float(step.payload.get("seconds") or 0.0))


def raw_responses(trajectory: Trajectory) -> list[ModelResponse]:
    return [model_response(s) for s in trajectory.steps if s.kind == StepKind.MODEL_CALL]


def items_from(trajectory: Trajectory, agent: Any) -> list[RunItem]:
    """The run's items in order. Tool results pair with the calls of their turn, in order."""
    items: list[RunItem] = []
    pending: list[str] = []
    for step in trajectory.steps:
        if step.kind == StepKind.MODEL_CALL:
            entry = step.payload.get("response") or {}
            if entry.get("content"):
                items.append(MessageOutputItem(agent, entry["content"]))
            pending = []
            for tc in entry.get("tool_calls") or []:
                fn = tc.get("function", {})
                items.append(ToolCallItem(agent, fn.get("name", ""), _arguments(fn.get("arguments")), tc.get("id", "")))
                pending.append(tc.get("id", ""))
        elif step.kind == StepKind.TOOL_RESULT:
            items.append(ToolCallOutputItem(agent, pending.pop(0) if pending else None, step.payload.get("result")))
    return items


def messages_from(trajectory: Trajectory) -> list[dict[str, Any]]:
    """The chat messages the run added after its input: assistant turns and tool results."""
    messages: list[dict[str, Any]] = []
    pending: list[str] = []
    for step in trajectory.steps:
        if step.kind == StepKind.MODEL_CALL:
            entry = dict(step.payload.get("response") or {})
            messages.append(entry)
            pending = [tc.get("id", "") for tc in entry.get("tool_calls") or []]
        elif step.kind == StepKind.TOOL_RESULT:
            messages.append({"role": "tool", "tool_call_id": pending.pop(0) if pending else "",
                             "content": json.dumps(step.payload.get("result"), default=str)})
    return messages


class ItemHelpers:
    @staticmethod
    def text_message_outputs(items: list[RunItem]) -> str:
        return "".join(i.content for i in items if isinstance(i, MessageOutputItem))
