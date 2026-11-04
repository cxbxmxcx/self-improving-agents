"""A visible-context responder for the local walkthrough, not a model benchmark."""

from contextlib import contextmanager
import json
import re
from types import SimpleNamespace
from unittest.mock import patch


def response(content: str = "", *, name: str | None = None, arguments: dict | None = None):
    calls = None if name is None else [SimpleNamespace(
        id="calendar-call-" + name, type="function",
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
    question = next(message["content"] for message in messages if message["role"] == "user")
    past = recalled_text(messages)
    if question.startswith("History only:"):
        return response("Past scheduling experience supplied by the harness:\n" + past
                        if past else "No earlier scheduling experience is available for this user.")
    results = [json.loads(message["content"]) for message in messages if message["role"] == "tool"]
    if not results:
        return response(name="read_calendar")
    last = results[-1]
    if "error" in last:
        return response("The calendar reported: " + last["error"])
    if len(results) == 1:
        explicit = re.search(r"\b(?:09|14):00\b", question)
        correction = re.search(r"please use ((?:09|14):00)", past)
        accepted_time = re.search(r"at ((?:09|14):00)\. Booking feedback: accepted", past)
        if explicit:
            start = explicit.group()
        elif correction:
            start = correction.group(1)
        elif accepted_time:
            start = accepted_time.group(1)
        else:
            start = "14:00"
        available = last["available_starts"]
        if start not in available:
            if explicit or not available:
                return response("The requested booking is unavailable; no event was created.")
            start = available[0]
        return response(name="book_time", arguments={"start": start, "duration_minutes": 45})
    if len(results) == 2:
        return response(name="review_booking")
    status = "accepted" if last["accepted"] is True else "rejected" if last["accepted"] is False else "pending"
    feedback = last["feedback"] or "No user feedback yet."
    return response(f"Created a 45-minute writing block on {last['date']} at {last['start']}. "
                    f"Booking feedback: {status}. User feedback: {feedback} "
                    "The event remains at its booked time.")


@contextmanager
def model_mode(live: bool):
    if live:
        yield
    else:
        with patch("helix.agent.helix_acompletion", side_effect=scripted_response):
            yield
