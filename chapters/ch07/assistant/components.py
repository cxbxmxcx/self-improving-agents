"""The learning component written by hand, in the shape a code search has to return.

A component stores memories and returns a router. Each memory has a route that says what serves a request that
retrieves it: a decision program with no language model, or the language model at a reasoning level, read-only when
the request must not be acted on. This one is the reference a searched component is measured against.

Its router and its one decision program are ordinary files in `programs/`, so they can be read and edited as code. The
functions here pack them into the two forms helix takes: a program with its memories, and a component whose `learn`
stores those memories and returns the router.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from helix.memory.mechanism_runtime import pack_memories

PROGRAMS = Path(__file__).parent / "programs"
SEVERAL = ("Call find_shared_starts once for the starts everyone can make, then book with book_time. "
           "When the request asks for two meetings, book them on different days.")
NOT_A_BOOKING = ("This request does not ask for a meeting to be booked now. Answer it from the calendars if you can, "
                 "say what you cannot tell, and change nothing.")


def memories() -> list[dict[str, Any]]:
    """Three kinds of request and what serves each: a program, the model with its tools, and the model read-only."""
    return [
        {"name": "one_meeting", "description": "Requests to book one meeting this week with the people named.",
         "body": (PROGRAMS / "book_meeting.py").read_text(encoding="utf-8"), "route": "program", "read_only": False},
        {"name": "several_meetings", "description": "Requests to book more than one meeting.",
         "body": SEVERAL, "route": "none", "read_only": False},
        {"name": "not_a_booking",
         "description": "Questions about calendars or meetings, and requests to cancel, move, hold or draft one.",
         "body": NOT_A_BOOKING, "route": "none", "read_only": True},
    ]


def router() -> str:
    return (PROGRAMS / "router.py").read_text(encoding="utf-8")


def hand_written_program() -> str:
    """The router with its memories packed in: what the decision tier runs."""
    return pack_memories(router(), memories())


def hand_written_component() -> str:
    """The same thing as a learning component: `learn` stores the memories and returns the router."""
    stored = [(m["name"], m["description"], m["body"], m["route"], m["read_only"]) for m in memories()]
    return (f"ROUTER = {router()!r}\nSTORED = {stored!r}\n\n\nasync def learn(lab):\n"
            "    for name, description, body, route, read_only in STORED:\n"
            "        lab.remember(name, description, body, route=route, read_only=read_only)\n"
            "    return ROUTER\n")
