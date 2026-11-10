"""The support desk, written on metacog: its tools, a trusted procedure, and how runs are judged."""

from __future__ import annotations

from typing import Any

from helix.metacognition import procedure

from examples.triage.world import TriageWorld
from metacog import Agent, RunContextWrapper, function_tool

MODEL = "gpt-5.4-mini"
PROMPT = """You run a software company's support desk. Do what the request asks with the tools: triage a ticket, by reading
its thread and the policy and then setting its severity and team once with triage_ticket, or assign a ticket to a named
person. When you are done, reply in one short sentence. If a request asks for anything else, answer it if you can and change
nothing."""
PLAN_OF = '''def ticket_plan(api, ticket_id: str) -> dict:
    return {"plan": api.read_account(ticket_id).get("plan")}
'''


@function_tool(read_only=True)
def read_ticket(ctx: RunContextWrapper[TriageWorld], ticket_id: str) -> dict:
    """Read a ticket's thread: who said what, in order."""
    return ctx.context.read_ticket(ticket_id)


@function_tool(read_only=True)
def read_account(ctx: RunContextWrapper[TriageWorld], ticket_id: str) -> dict:
    """Read the plan of the customer behind a ticket: standard or enterprise."""
    return ctx.context.read_account(ticket_id)


@function_tool(read_only=True)
def read_policy(ctx: RunContextWrapper[TriageWorld]) -> dict:
    """Read the triage policy: how severity and team are decided."""
    return ctx.context.read_policy()


@function_tool
def triage_ticket(ctx: RunContextWrapper[TriageWorld], ticket_id: str, severity: str, team: str) -> dict:
    """Set a ticket's severity (P1 to P4) and its owning team. Do this once."""
    return ctx.context.triage_ticket(ticket_id, severity, team)


@function_tool
def assign_ticket(ctx: RunContextWrapper[TriageWorld], ticket_id: str, owner: str) -> dict:
    """Assign a ticket to a named person."""
    return ctx.context.assign_ticket(ticket_id, owner)


@procedure(api=[read_account])
def ticket_plan(ticket_id: str) -> dict:
    """Look up the plan of the customer behind a ticket.

    Returns: {"plan": str}, standard or enterprise.
    """
    ...


def desk(model: str = MODEL, **kwargs: Any) -> Agent:
    return Agent(name="Support desk", instructions=PROMPT, model=model,
                 tools=[read_ticket, read_account, read_policy, triage_ticket, assign_ticket], **kwargs)


def check(ctx: RunContextWrapper[TriageWorld], result: Any) -> dict:
    """Whether the change the desk made is what was asked for and what the policy gives."""
    return ctx.context.check()
