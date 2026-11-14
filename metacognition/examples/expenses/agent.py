"""Jordan's expense assistant, written on metacog: tools, the procedure worth learning, and how runs are judged."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from helix.memory.code_skills import skill_tool
from helix.metacognition import procedure

from examples.expenses.world import COMPANY, EMPLOYEE, ExpenseWorld, task_by_id
from metacog import Agent, RunContextWrapper, function_tool

MODEL = "gpt-5.4-mini"
PROMPT = f"""You are {EMPLOYEE}'s travel expense assistant at {COMPANY}. File the user's expense
claims with the tools: read the trip and its receipts, read the company expense policy, work out
the reimbursable amount in US dollars, and file one claim. When you are done, reply with the
amount you claimed."""
REFERENCE = (Path(__file__).parent / "memories" / "reimbursable_total.py").read_text(encoding="utf-8")


@function_tool(read_only=True)
def read_trip(ctx: RunContextWrapper[ExpenseWorld], trip_id: str) -> dict:
    """Read a trip's details and every receipt, with each amount in the currency it was paid in."""
    return ctx.context.read_trip(trip_id)


@function_tool(read_only=True)
def read_policy(ctx: RunContextWrapper[ExpenseWorld]) -> dict:
    """Read the company expense policy: allowances by city, company exchange rates, and the policy notes."""
    return ctx.context.read_policy()


@function_tool
def file_claim(ctx: RunContextWrapper[ExpenseWorld], trip_id: str, amount_usd: float) -> dict:
    """File the expense claim for a trip, in US dollars. File once per trip."""
    return ctx.context.file_claim(trip_id, amount_usd)


@procedure(api=[read_trip, read_policy])
def reimbursable_total(trip_id: str) -> dict:
    """Work out how much of a trip's receipts the company reimburses, in US dollars, under its expense policy.

    Returns: {"total_usd": float}, the reimbursable total rounded to the cent.
    """
    ...


def assistant(model: str = MODEL, **kwargs: Any) -> Agent:
    return Agent(name="Jordan's expense assistant", instructions=PROMPT, model=model,
                 tools=[read_trip, read_policy, file_claim], **kwargs)


def check(ctx: RunContextWrapper[ExpenseWorld], result: Any) -> dict:
    """The simulated finance team's verdict on what was filed."""
    return ctx.context.check()


class WorldAPI:
    """What a procedure reads in function-level practice: the case's own world, read-only."""

    def __init__(self, world: ExpenseWorld) -> None:
        self.read_trip = world.read_trip
        self.read_policy = world.read_policy


async def run_on_case(source: str, task: dict[str, Any]) -> Any:
    """A function's output on one case, under the code-skill guard."""
    world = ExpenseWorld(task)
    try:
        tool = skill_tool("reimbursable_total", "", source, bind={"api": WorldAPI(world)})
        return await tool.call(trip_id=task["trip"]["trip_id"])
    except Exception as exc:  # noqa: BLE001 - refused by the guard or not callable with the contract's arguments
        return {"error": f"reimbursable_total could not be called: {type(exc).__name__}: {exc}"}


async def practice(source: str, case_id: str) -> tuple[float, str, str]:
    """Chapter 6's act for this world: run the function on the case, file what it returns, ask the checker."""
    task = task_by_id(case_id)
    output = await run_on_case(source, task)
    observed = f"called with trip_id={task['trip']['trip_id']!r}; returned {json.dumps(output, default=str)[:600]}"
    if isinstance(output, dict) and "error" in output:
        return 0.0, f"the function crashed when run: {output['error']}", observed
    total = output.get("total_usd") if isinstance(output, dict) else None
    if not isinstance(total, (int, float)):
        return 0.0, f"the function returned {output!r}, not {{\"total_usd\": float}}", observed
    world = ExpenseWorld(task)
    world.file_claim(task["trip"]["trip_id"], float(total))
    verdict = world.check()
    return (1.0 if verdict["success"] else 0.0), verdict["reason"], observed
