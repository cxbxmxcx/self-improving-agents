"""Claude models through Anthropic's own SDK, behind the OpenAI-style call helix makes.

helix calls every model with LiteLLM's `acompletion` and reads a LiteLLM-shaped
response. For Claude models metacog sends the call to the Messages API with the
official SDK instead, because the reasoning settings don't survive LiteLLM's
translation on the Claude 5.5 family: LiteLLM sends nothing for `none`, which
leaves thinking on (it is on by default); Sonnet 5.5 rejects disabled thinking
and turns it off with `between_tools`; Opus 5.5 can't turn it off at all.

This module takes the call helix makes and returns what helix reads:

- system messages become `system`; tool results become `tool_result` blocks in
  one user turn; an assistant turn this module produced goes back with its
  original content blocks, thinking included, as the API requires;
- `reasoning_effort` becomes `thinking` and `output_config.effort`; `none`
  turns thinking off where the model allows it, at the lowest effort;
- usage maps to prompt, completion, cached and reasoning (thinking) tokens.

`helix.metacognition.recording` routes every `claude-...` call here, recorded or not.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any

# How each Claude model turns thinking off. Haiku 5.5 takes `disabled` (at effort high or below);
# Sonnet 5.5 rejects `disabled` and turns thinking off with `between_tools`. The models missing
# here (Opus 5.5, Fable 5.1) can't turn thinking off, so their lowest reasoning setting is low.
THINKING_OFF: dict[str, dict[str, str]] = {
    "claude-haiku-5-5": {"type": "disabled"},
    "claude-sonnet-5-5": {"type": "between_tools"},
}
EFFORTS = ("low", "medium", "high", "xhigh", "max")
MAX_TOKENS = 32_000  # room for thinking and the reply; every call streams, so no request timeout
KEPT_TURNS = 20_000  # assistant turns remembered for sending back, before the memory resets

_clients: dict[int, Any] = {}
# Each assistant turn this module returned, as the API sent it (thinking blocks included), keyed by
# its tool calls' ids, or by its text when it called no tool: helix sends the turn back in OpenAI
# form, and the API wants the original blocks.
_turns: dict[str, list[Any]] = {}


def is_claude(model: str | None) -> bool:
    """Whether a call goes to Anthropic: a bare `claude-...` name, or one prefixed `anthropic/`."""
    name = (model or "").lower()
    if name.startswith("anthropic/"):
        name = name.split("/", 1)[1]
    return name.startswith("claude-")


def bare(model: str) -> str:
    return model.split("/", 1)[1] if model.lower().startswith("anthropic/") else model


def can_turn_thinking_off(model: str) -> bool:
    return bare(model) in THINKING_OFF


def thinking_params(model: str, effort: str | None) -> dict[str, Any]:
    """`thinking` and `output_config` for a reasoning setting, or nothing when the call names none."""
    if effort is None:
        return {}
    if effort in ("none", "minimal"):
        off = THINKING_OFF.get(bare(model))
        if off is None:
            raise ValueError(f"{model} can't turn thinking off; its lowest reasoning setting is low")
        return {"thinking": dict(off), "output_config": {"effort": "low"}}
    if effort not in EFFORTS:
        raise ValueError(f"{effort!r} is not a reasoning setting for {model}; use none or one of {EFFORTS}")
    return {"thinking": {"type": "adaptive"}, "output_config": {"effort": effort}}


def _text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)
    return json.dumps(content, default=str)


def _turn_key(tool_ids: list[str], text: str) -> str:
    return "tools:" + ",".join(tool_ids) if tool_ids else "text:" + text


def _arguments(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    try:
        value = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return {"arguments": raw}
    return value if isinstance(value, dict) else {"value": value}


def _only_tool_results(message: dict[str, Any]) -> bool:
    content = message["content"]
    return isinstance(content, list) and all(block.get("type") == "tool_result" for block in content)


def to_messages(messages: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    """helix's OpenAI-style messages as Anthropic's `system` and `messages`."""
    system: list[str] = []
    out: list[dict[str, Any]] = []
    for message in messages:
        role, content = message.get("role"), message.get("content")
        if role == "system":
            if _text(content):
                system.append(_text(content))
        elif role == "tool":
            block = {"type": "tool_result", "tool_use_id": message["tool_call_id"],
                     "content": _text(content) or "(no output)"}
            if out and out[-1]["role"] == "user" and _only_tool_results(out[-1]):
                out[-1]["content"].append(block)  # every result of one turn goes back in one user message
            else:
                out.append({"role": "user", "content": [block]})
        elif role == "assistant":
            calls = message.get("tool_calls") or []
            kept = _turns.get(_turn_key([call["id"] for call in calls], _text(content)))
            if kept is not None:
                out.append({"role": "assistant", "content": kept})
                continue
            blocks: list[dict[str, Any]] = []
            if _text(content):
                blocks.append({"type": "text", "text": _text(content)})
            for call in calls:
                function = call["function"]
                blocks.append({"type": "tool_use", "id": call["id"], "name": function["name"],
                               "input": _arguments(function.get("arguments"))})
            out.append({"role": "assistant", "content": blocks or [{"type": "text", "text": "(no reply)"}]})
        else:
            out.append({"role": "user", "content": _text(content) or "(empty)"})
    return "\n\n".join(system), out


def to_tools(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """OpenAI function tools as Anthropic tools."""
    out = []
    for tool in tools or []:
        function = tool.get("function", tool)
        out.append({"name": function["name"], "description": function.get("description") or "",
                    "input_schema": function.get("parameters") or {"type": "object", "properties": {}}})
    return out


def to_tool_choice(choice: Any) -> dict[str, str] | None:
    if choice is None:
        return None
    if choice == "none":
        return {"type": "none"}
    return {"type": "auto"}  # Sonnet 5.5 and Opus 5.5 reject forced choices; the prompt names the tool


def to_response(message: Any, model: str) -> SimpleNamespace:
    """A Messages API reply shaped like LiteLLM's, the way helix and metacog's recordings read it."""
    text = "".join(block.text for block in message.content if block.type == "text")
    calls = [SimpleNamespace(id=block.id, type="function",
                             function=SimpleNamespace(name=block.name, arguments=json.dumps(block.input)))
             for block in message.content if block.type == "tool_use"]
    if len(_turns) >= KEPT_TURNS:
        _turns.clear()
    _turns[_turn_key([call.id for call in calls], text)] = list(message.content)
    usage = message.usage
    cached = usage.cache_read_input_tokens or 0
    details = getattr(usage, "output_tokens_details", None)
    thinking = (getattr(details, "thinking_tokens", None) or 0) if details is not None else 0
    return SimpleNamespace(
        model=model,
        choices=[SimpleNamespace(finish_reason=message.stop_reason,
                                 message=SimpleNamespace(content=text, tool_calls=calls or None))],
        usage=SimpleNamespace(
            prompt_tokens=usage.input_tokens + cached + (usage.cache_creation_input_tokens or 0),
            completion_tokens=usage.output_tokens,
            completion_tokens_details=SimpleNamespace(reasoning_tokens=thinking),
            prompt_tokens_details=SimpleNamespace(cached_tokens=cached),
        ),
    )


def _client() -> Any:
    """One client per event loop: a client's connections belong to the loop that opened them."""
    import anthropic

    loop = id(asyncio.get_running_loop())
    if loop not in _clients:
        _clients[loop] = anthropic.AsyncAnthropic(max_retries=6, timeout=600.0)
    return _clients[loop]


async def acompletion(*, model: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None,
                      reasoning_effort: str | None = None, tool_choice: Any = None, max_tokens: int | None = None,
                      max_completion_tokens: int | None = None, response_format: Any = None,
                      **ignored: Any) -> SimpleNamespace:
    """Call a Claude model the way helix calls LiteLLM, and answer the way LiteLLM does.

    `ignored` takes LiteLLM's transport settings (retries, timeout) and sampling settings, which the
    Claude 5.5 models reject; the SDK retries rate limits and server errors itself.
    """
    if response_format is not None:
        raise NotImplementedError("structured output on Claude models needs output_config.format; "
                                  "metacog doesn't translate response_format yet")
    name = bare(model)
    system, converted = to_messages(messages)
    params: dict[str, Any] = {"model": name, "max_tokens": max_completion_tokens or max_tokens or MAX_TOKENS,
                              "messages": converted, **thinking_params(name, reasoning_effort)}
    if system:
        params["system"] = system
    if tools:
        params["tools"] = to_tools(tools)
        choice = to_tool_choice(tool_choice)
        if choice:
            params["tool_choice"] = choice
    async with _client().messages.stream(**params) as stream:
        message = await stream.get_final_message()
    return to_response(message, name)
