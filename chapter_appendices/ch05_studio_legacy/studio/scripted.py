"""A scripted responder for a reproducible walkthrough, not an LLM benchmark."""

from __future__ import annotations

from contextlib import contextmanager
import json
from types import SimpleNamespace
from unittest.mock import patch


def response(content: str = "", *, name: str | None = None, arguments: dict | None = None):
    calls = None if name is None else [SimpleNamespace(
        id="studio-call-" + name, type="function",
        function=SimpleNamespace(name=name, arguments=json.dumps(arguments or {})),
    )]
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=calls))],
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2),
    )


def recalled_text(messages: list[dict]) -> str:
    content = messages[0]["content"]
    text = content if isinstance(content, str) else content[0]["text"]
    marker = "## Relevant past interactions"
    return text.split(marker, 1)[1].strip() if marker in text else ""


async def scripted_response(*, messages, **_):
    question = next(m["content"] for m in messages if m["role"] == "user")
    past = recalled_text(messages)
    if question.startswith("History only:"):
        return response("Prior case supplied by the harness:\n" + past if past else "No prior incident is available in this user's context.")
    tool_messages = [m for m in messages if m["role"] == "tool"]
    if not tool_messages:
        # Retrieval sees current symptoms plus the hypothesis memory supplied.
        return response(name="retrieve", arguments={"query": question + " " + past, "k": 3})
    last = json.loads(tool_messages[-1]["content"])
    if isinstance(last, list):
        q = question.lower()
        if "auth17" in q:
            action = "refresh_asset_token"
        elif "net01" in q or ("asset sync" in q and "reconnect_vpn" in past):
            # Deliberately try the remembered fix once before a changed cause
            # becomes visible in the tool result. This is a teaching fixture.
            action = "reconnect_vpn"
        elif "stagelink" in q:
            action = "select_stagelink_input"
        elif "queued" in q:
            action = "check_build_queue"
        elif "perm04" in q:
            action = "request_asset_access"
        else:
            action = "collect_diagnostics"
        return response(name="support_step", arguments={"action": action})
    if "error" in last:
        return response("The simulator did not perform a step: " + last["error"])
    passages = json.loads(tool_messages[0]["content"])
    supporting = next((p for p in passages if last["action"] in p["text"]), None)
    source = supporting["source"] if supporting else passages[0]["source"] if passages else "no matching guide"
    status = "resolved" if last["resolved"] else "unresolved"
    return response(f"Attempted {last['action']}; {status}. {last['observed']} Source consulted: {source}.")


@contextmanager
def model_mode(live: bool):
    if live:
        yield
    else:
        with patch("helix.agent.helix_acompletion", side_effect=scripted_response):
            yield
