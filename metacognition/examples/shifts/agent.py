"""The on-call desk, written on metacog: its tools, a trusted procedure, and how runs are judged."""

from __future__ import annotations

from typing import Any

from helix.metacognition import procedure

from examples.shifts.world import ShiftWorld
from metacog import Agent, RunContextWrapper, function_tool

MODEL = "gpt-5.4-mini"
PROMPT = """You run the on-call desk for a small engineering team. Do what the request asks with the tools: put one person
on one day's shift, or plan next week's rota and publish it. A plan must meet every constraint in the request and the
team's standing rules, which read_team gives you, so work it out before you publish, and publish only once. When you are
done, reply in one short sentence. If a request asks for anything else, answer it if you can and change nothing."""
AVAILABLE = '''def available_people(api, day: str) -> dict:
    team = api.read_team()
    leave = api.read_leave()["leave"]
    wanted = day.strip().capitalize()
    return {"people": [person for person in team["people"] if wanted not in leave.get(person, [])]}
'''


@function_tool(read_only=True)
def read_team(ctx: RunContextWrapper[ShiftWorld]) -> dict:
    """Read the team: each person's skills, and the standing rules every rota has to meet."""
    return ctx.context.read_team()


@function_tool(read_only=True)
def read_leave(ctx: RunContextWrapper[ShiftWorld]) -> dict:
    """Read who is on leave next week, and on which days."""
    return ctx.context.read_leave()


@function_tool(read_only=True)
def read_rota(ctx: RunContextWrapper[ShiftWorld]) -> dict:
    """Read the current rota: who is on call each day."""
    return ctx.context.read_rota()


@function_tool
def assign_shift(ctx: RunContextWrapper[ShiftWorld], day: str, person: str) -> dict:
    """Put one person on one day's on-call shift."""
    return ctx.context.assign_shift(day, person)


@function_tool
def publish_rota(ctx: RunContextWrapper[ShiftWorld], monday: str, tuesday: str, wednesday: str, thursday: str,
                 friday: str) -> dict:
    """Publish next week's rota: the person on call each day. Publish once."""
    return ctx.context.publish_rota(monday, tuesday, wednesday, thursday, friday)


@procedure(api=[read_team, read_leave])
def available_people(day: str) -> dict:
    """Work out who can take a shift on a day: everyone on the team who is not on leave that day.

    Returns: {"people": [str]}, the names of those available.
    """
    ...


def desk(model: str = MODEL, **kwargs: Any) -> Agent:
    return Agent(name="On-call desk", instructions=PROMPT, model=model,
                 tools=[read_team, read_leave, read_rota, assign_shift, publish_rota], **kwargs)


def check(ctx: RunContextWrapper[ShiftWorld], result: Any) -> dict:
    """Whether the change the desk made is what was asked for, and meets everything it had to."""
    return ctx.context.check()
