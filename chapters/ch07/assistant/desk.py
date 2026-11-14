"""Set up Chapter 6's calendar assistant for the packaged harness and for decision programs.

Chapter 6 built this assistant's calendars, checker, tools and procedure, and nothing here changes them. This module
names those parts for helix's packaged metacognition harness, which holds the recorder, detector, learner, gate and
controller that listing 6.2 wired by hand. It also describes the same world to a decision program as a desk.

A desk gives a program the read tool, the trusted procedure, the one action it may take, and the checker's verdict.
The contract tells whoever writes a program what the job is and what must be handed off. A request that should not be
acted on is right only when nothing gets booked.
"""

from __future__ import annotations

from typing import Any

from chapters.ch06.assistant.environment import SchedulingEnvironment
from chapters.ch06.assistant.harness import (
    STARTING_MEMORY,
    STARTING_SAME_TIME,
    MODEL,
    PROMPT,
    ReadOnlyCalendar,
)
from chapters.ch06.assistant.learning import (
    CONTRACT,
    SAME_TIME_CONTRACT,
    starting_source,
)
from chapters.ch06.assistant.tools import make_tools
from helix.memory.code_skills import skill_tool
from helix.memory.learning_mechanisms import Contract
from helix.metacognition import (
    Agent,
    Procedure,
    RunContextWrapper,
    Verdict,
    function_tool,
)
from helix.metacognition.decide import KIT_GUIDE, RETURNS_ROUTED, Desk

DESCRIPTIONS = {tool.name: tool.description for tool in make_tools(None)}
# Requests the assistant must leave alone: they use the words of the work but ask for something else.
NOT_FOR_ACTION = ("near_miss", "out_of_scope")
# Who a verdict on such a request comes from. It judges whether the request was read right, so it can switch a
# decision program's action off, and it says nothing about the scheduling function the agent may have called.
INTENT = "intent"
JOB = """Serve one request to a personal calendar assistant. Its job is booking one meeting this week for the user and
the people the request names, at a start every one of them can make. A request for more than one meeting goes to the
agent. A request that asks about a meeting, or asks to cancel, move, hold or draft one, goes to the agent read-only.
Read tool: read_calendar(person, day), answering one person's events for a day (mon to fri), or for the week when day
is empty. An empty person means the user, a first name is enough when only one person has it, and an unknown name
answers {"error": ...}. Procedure: find_shared_starts(attendees, days, minutes, earliest, latest_end), answering
{"starts": {day: ["HH:MM", ...]}} with the starts everyone can make, earliest first. Its days are mon to fri, and
earliest and latest_end are HH:MM or empty. Action: book_time, with the arguments {"title": str, "day": str,
"start": "HH:MM", "minutes": int, "attendees": [full names]}."""
# Added to the job when the assistant also takes Chapter 6's second class of request.
SECOND_CLASS = """The assistant also takes requests for the same meeting at the same time on each of several days. Such a
request needs one booking for every day it names, so it goes to the agent. Procedure the agent has for it:
find_common_starts(attendees, days, minutes, earliest, latest_end), answering {"starts": ["HH:MM", ...]} with the
starts that work on every listed day, earliest first."""


# A plain function, not a coroutine: a decision program calls the agent's read tools directly.
@function_tool(description_override=DESCRIPTIONS["read_calendar"], read_only=True)
def read_calendar(ctx: RunContextWrapper[SchedulingEnvironment], person: str = "", day: str = "") -> dict:
    return ctx.context.read_calendar(person, day)


@function_tool(description_override=DESCRIPTIONS["book_time"])
async def book_time(ctx: RunContextWrapper[SchedulingEnvironment], title: str, day: str, start: str, minutes: int,
                    attendees: list[str] | None = None) -> dict:
    return ctx.context.book_time(title, day, start, minutes, attendees)


@function_tool(description_override=DESCRIPTIONS["cancel_event"])
async def cancel_event(ctx: RunContextWrapper[SchedulingEnvironment], event_id: str) -> dict:
    return ctx.context.cancel_event(event_id)


# Chapter 6's procedure, with its contract and the description the agent reads.
FIND_SHARED_STARTS = Procedure.from_contract(CONTRACT, api=[read_calendar],
                                             description=STARTING_MEMORY["description"])
# The procedure of Chapter 6's second class: the same start on every listed day.
FIND_COMMON_STARTS = Procedure.from_contract(SAME_TIME_CONTRACT, api=[read_calendar],
                                             description=STARTING_SAME_TIME["description"])


def assistant(model: str = MODEL, **kwargs: Any) -> Agent:
    """The calendar assistant on the packaged harness, with Chapter 6's prompt and tools."""
    return Agent(name="Maya's calendar assistant", instructions=PROMPT, model=model,
                 tools=[read_calendar, book_time, cancel_event], **kwargs)


def world_for(task: dict[str, Any]) -> SchedulingEnvironment:
    """A fresh copy of the task's week, remembering what kind of request this is."""
    world = SchedulingEnvironment(task)
    world.style = task.get("style", "plain")
    return world


def check(ctx: RunContextWrapper[SchedulingEnvironment], result: Any) -> Verdict:
    """The verdict on one run: Chapter 6's checker, or nothing booked when the request was not for action."""
    world = ctx.context
    if getattr(world, "style", "plain") in NOT_FOR_ACTION:
        booked = bool(world.booked)
        return Verdict(not booked, "it booked a meeting nobody asked for" if booked else "took no action", INTENT)
    return Verdict.coerce(world.check())


def desk(procedure: str | None = None) -> Desk:
    """The calendar world for a decision program. `procedure` is the trusted find_shared_starts, Chapter 6's by default.

    Inside the decision tier the loop calls the procedure the agent trusts at that moment, so this one is used only
    when a program is served on its own.
    """
    source = procedure or starting_source()

    def procedures(world: SchedulingEnvironment) -> dict[str, Any]:
        async def find_shared_starts(**arguments: Any) -> Any:
            tool = skill_tool("find_shared_starts", "", source, bind={"api": ReadOnlyCalendar(world)})
            return await tool.call(**arguments)

        return {"find_shared_starts": find_shared_starts}

    def book(world: SchedulingEnvironment, title: str, day: str, start: str, minutes: int,
             attendees: list[str] | None = None) -> dict:
        return world.book_time(title, day, start, minutes, attendees)

    return Desk(world=world_for, reads=lambda world: {"read_calendar": world.read_calendar}, procedures=procedures,
                actions={"book_time": book}, verdict=lambda world, reply: world.check())


def contract(second_class: bool = False) -> Contract:
    """What a decision program for this desk must do, for whoever writes one."""
    job = f"{JOB}\n{SECOND_CLASS}" if second_class else JOB
    return Contract(name="decide", signature="async def decide(request, kit) -> dict", description=job, api=KIT_GUIDE,
                    returns=RETURNS_ROUTED)
