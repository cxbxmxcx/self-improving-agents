"""The chapter 6 calendar assistant written on metacog's SDK surface, shared by the parity tests.

Only the calendar world, the prompt, and the tool descriptions come from the
book; the agent is metacog's. Parity holds only if it sends the book's exact
requests, so the recordings have a reply for every call.
"""

from __future__ import annotations

from typing import Any

from chapters.ch06.assistant.environment import SchedulingEnvironment
from chapters.ch06.assistant.harness import BOOTSTRAPPED_MEMORY, MODEL, PROMPT
from chapters.ch06.assistant.learning import CONTRACT
from chapters.ch06.assistant.tools import make_tools

from metacog import Agent, RunContextWrapper, function_tool
from metacog.meta import Procedure

DESCRIPTIONS = {t.name: t.description for t in make_tools(None)}


@function_tool(description_override=DESCRIPTIONS["read_calendar"], read_only=True)
async def read_calendar(ctx: RunContextWrapper[SchedulingEnvironment], person: str = "", day: str = "") -> dict:
    return ctx.context.read_calendar(person, day)


@function_tool(description_override=DESCRIPTIONS["book_time"])
async def book_time(ctx: RunContextWrapper[SchedulingEnvironment], title: str, day: str, start: str, minutes: int,
                    attendees: list[str] | None = None) -> dict:
    return ctx.context.book_time(title, day, start, minutes, attendees)


@function_tool(description_override=DESCRIPTIONS["cancel_event"])
async def cancel_event(ctx: RunContextWrapper[SchedulingEnvironment], event_id: str) -> dict:
    return ctx.context.cancel_event(event_id)


# The first task class's procedure, with the book's contract text and tool description.
FIND_SHARED_STARTS = Procedure.from_contract(CONTRACT, api=[read_calendar],
                                             description=BOOTSTRAPPED_MEMORY["description"])


def assistant(**kwargs: Any) -> Agent:
    return Agent(name="Maya's calendar assistant", instructions=PROMPT, model=MODEL,
                 tools=[read_calendar, book_time, cancel_event], **kwargs)


def check(ctx: RunContextWrapper[SchedulingEnvironment], result: Any) -> dict:
    """The study's checker: the simulated user's verdict on what was booked."""
    return ctx.context.check()
