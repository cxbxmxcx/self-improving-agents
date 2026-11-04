"""A scripted stand-in for the assistant's model, for walkthroughs without keys.

Each request in requests.json writes down what the stand-in does on each
path: the fast path's documented mistakes, the deliberate path's careful
plan, and the question the ask path puts to the user. The stand-in reads
only what a model would see (the request, the system message, which shows
the path and what memory recalled, and the tool results), never the hidden
truth. Its token counts are illustrative, so its costs show the mechanics,
not a model's measured prices.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from chapter_appendices.ch06_decision_scaffold_legacy.assistant.environment import load_requests, to_minutes
from helix.scaffold import ASK_NOTE, FINISH_NOTE, PLANNER_HEADING, REVISION_PROMPT

FAST_MODEL, DELIBERATE_MODEL = "scripted-fast", "scripted-deliberate"
# Illustrative token counts per call: (prompt, completion).
USAGE = {"fast": (1200, 150), "ask": (1200, 60), "deliberate": (1800, 900)}
SAMS = {"Sam Ortiz": {"event": "E06", "message": "M3"}, "Sam Lee": {"event": "E07", "message": "M6"}}
ASK_HEADING, FINISH_HEADING = ASK_NOTE.splitlines()[0], FINISH_NOTE.splitlines()[0]


def _by_text() -> dict[str, dict[str, Any]]:
    return {r["text"]: r for r in load_requests()}


def _response(content: str = "", *, call: dict[str, Any] | None = None, number: int = 0, path: str = "fast"):
    calls = None if call is None else [SimpleNamespace(
        id=f"scripted-{number}", type="function",
        function=SimpleNamespace(name=call["call"], arguments=json.dumps(call.get("args", {}))))]
    prompt, completion = USAGE[path]
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=calls))],
        usage=SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion,
                              total_tokens=prompt + completion))


def _system(messages: list[dict[str, Any]]) -> str:
    content = messages[0]["content"]
    return content if isinstance(content, str) else content[0]["text"]


def _recalled(system: str) -> str:
    marker = "## Relevant past interactions"
    return system.split(marker, 1)[1] if marker in system else ""


def _lessons(recalled: str) -> str:
    return "\n".join(line for line in recalled.splitlines() if line.strip().startswith("Lesson:"))


def _free(results: list[dict[str, Any]], day: str) -> list[str] | None:
    for result in reversed(results):
        for entry in result.get("days", []) if isinstance(result, dict) else []:
            if entry.get("day") == day:
                return entry.get("free", [])
    return None


def _fits(start: str, minutes: int, free: list[str]) -> bool:
    s = to_minutes(start)
    for gap in free:
        low, high = (to_minutes(t) for t in gap.split("-"))
        if low <= s and s + minutes <= high:
            return True
    return False


def _writing_start(day: str, request: str, recalled: str, results: list[dict[str, Any]]) -> str:
    """The rule chapter 5's stand-in used: a time the request names, else a
    recalled correction, else an earlier accepted time, else 14:00."""
    explicit = re.search(r"\b\d{2}:\d{2}\b", request)
    correction = re.search(r"use (\d{2}:\d{2}) for writing", recalled)
    accepted = re.search(r"at (\d{2}:\d{2})\. Feedback: That works", recalled)
    start = (explicit.group() if explicit else correction.group(1) if correction
             else accepted.group(1) if accepted else "14:00")
    free = _free(results, day)
    if free is not None and not _fits(start, 45, free):
        start = next((t for t in ("09:00", "14:00", "16:00") if _fits(t, 45, free)), start)
    return start


def _resolve(value: Any, context: dict[str, Any]) -> Any:
    if isinstance(value, dict):
        return {k: _resolve(v, context) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve(v, context) for v in value]
    if not isinstance(value, str):
        return value

    def fill(match: re.Match) -> str:
        name, _, argument = match.group(1).partition(":")
        if name == "writing_start":
            return _writing_start(argument, context["request"], context["recalled"], context["results"])
        if name in ("sam_event", "sam_message"):
            meant = re.search(r"means\b[^.]*?(Sam Ortiz|Sam Lee)", _lessons(context["recalled"]))
            return SAMS[meant.group(1)][name.split("_")[1]] if meant else argument
        if name == "agenda":
            return " Agenda: review the proposal, then agree next steps." if re.search(
                r"agenda", _lessons(context["recalled"]), re.I) else ""
        if name == "room":
            return " We can use the Maple room." if re.search(
                r"\broom\b", _lessons(context["recalled"]), re.I) else ""
        if name == "feedback":
            said = [r.get("feedback") for r in context["results"] if isinstance(r, dict) and r.get("feedback")]
            return said[-1] if said else "No feedback yet."
        return match.group(0)

    return re.sub(r"\{([a-z_]+(?::[A-Za-z0-9]+)?)\}", fill, value)


async def scripted_model(*, model: str, messages: list[dict[str, Any]], **_: Any):
    """The agent loop's model call, answered from the request's script."""
    system = _system(messages)
    text = next(m["content"] for m in messages if m["role"] == "user")
    request = _by_text().get(text)
    if request is None:
        return _response("This walkthrough has no script for that request.")
    script = request["script"]
    if ASK_HEADING in system:
        return _response(script.get("ask", "Which one do you mean?"), path="ask")
    path = "deliberate" if PLANNER_HEADING in system else "fast"
    plan = script[path]
    results = [json.loads(m["content"]) for m in messages if m["role"] == "tool"]
    context = {"request": text, "recalled": _recalled(system), "results": results}
    cancelled = next((r for r in results if isinstance(r, dict) and r.get("cancelled")), None)
    if cancelled is not None:
        fallback = f"That step {cancelled.get('reason', 'needs your confirmation')}. Shall I go ahead?"
        return _response(_resolve(plan.get("if_cancelled", fallback), context), path=path)
    failed = next((r for r in results if isinstance(r, dict) and r.get("error")), None)
    if failed is not None:
        return _response(f"I could not finish: {failed['error']}.", path=path)
    made = sum(1 for m in messages if m["role"] == "assistant" and m.get("tool_calls"))
    if made < len(plan["steps"]) and FINISH_HEADING not in system:
        return _response(call=_resolve(plan["steps"][made], context), number=made, path=path)
    return _response(_resolve(plan["say"], context), path=path)


async def scripted_system2(*, model: str, messages: list[dict[str, Any]], **_: Any):
    """The scaffold's System 2 calls: a revision, or a lesson from a rejection."""
    system, user = messages[0]["content"], messages[1]["content"]
    try:
        state = json.loads(user)
    except json.JSONDecodeError:
        state = {}
    request = _by_text().get(state.get("request", ""), {"script": {}})
    script = request["script"]
    if system == REVISION_PROMPT:
        return _response(script.get("revised") or state.get("reply", ""), path="deliberate")
    lesson = script.get("lesson") or (f"For requests like '{state.get('request', '')}', follow this feedback: "
                                      f"{state.get('user_feedback', '')}")
    return _response(lesson, path="deliberate")


@contextmanager
def model_mode(live: bool) -> Iterator[None]:
    """Use real models when live; otherwise answer every model call from the scripts.

    A scripted call takes no time, so the clock that times model calls reads
    zero: wall-clock noise would otherwise decide near-ties in a comparison
    and make a replay differ from its recording.
    """
    if live:
        yield
        return
    still = SimpleNamespace(perf_counter=lambda: 0.0)
    with patch("helix.agent.helix_acompletion", side_effect=scripted_model), \
            patch("helix.scaffold.common.acompletion", side_effect=scripted_system2), \
            patch("helix.agent.time", still), patch("helix.scaffold.common.time", still):
        yield
