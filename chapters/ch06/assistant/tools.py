"""Bind the calendar tools to one simulated task environment.

The descriptions identify each tool's action and arguments. They leave the
scheduling procedure to the agent or its recalled memory. Keeping that
procedure out of the tool descriptions lets the memory comparison measure
its contribution.
"""

from __future__ import annotations

from typing import Any

from helix.tools import Tool, tool


def make_tools(env: Any) -> list[Tool]:
    @tool(description="Read one person's calendar for one day (mon to fri), or for the whole week when day "
                      "is empty. Returns their events for the day. person is a full name such as "
                      "'Sam Ortiz'; empty means the user.")
    async def read_calendar(person: str = "", day: str = "") -> dict:
        return env.read_calendar(person, day)

    @tool(description="Create an event on the user's calendar and invite attendees by full name. day is "
                      "mon to fri, start is HH:MM, minutes is the length.")
    async def book_time(title: str, day: str, start: str, minutes: int,
                        attendees: list[str] | None = None) -> dict:
        return env.book_time(title, day, start, minutes, attendees)

    @tool(description="Cancel an event you booked for the current request, by its id.")
    async def cancel_event(event_id: str) -> dict:
        return env.cancel_event(event_id)

    return [read_calendar, book_time, cancel_event]
