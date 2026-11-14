"""The billing desk, written on metacog: its tools, the procedure worth learning, and how runs are judged."""

from __future__ import annotations

import json
from typing import Any

from helix.memory.code_skills import skill_tool
from helix.metacognition import procedure

from examples.billing.world import COMPANY, BillingWorld, task_by_id
from metacog import Agent, RunContextWrapper, function_tool

MODEL = "gpt-5.4-mini"
PROMPT = f"""You are the billing desk's assistant at {COMPANY}. Issue the credits you are asked to issue with the tools: read
the account, read the billing policy, work out the credit, and issue it once. When you are done, reply with the amount you
issued. If a request asks for anything else, answer it if you can and issue nothing."""


@function_tool(read_only=True)
def read_account(ctx: RunContextWrapper[BillingWorld], account_id: str) -> dict:
    """Read an account: its plan, term, seats, billing day, any promotion, and its recent change."""
    return ctx.context.read_account(account_id)


@function_tool(read_only=True)
def read_policy(ctx: RunContextWrapper[BillingWorld]) -> dict:
    """Read the billing policy: the price per seat of each plan, and the rules for credits."""
    return ctx.context.read_policy()


@function_tool
def issue_credit(ctx: RunContextWrapper[BillingWorld], account_id: str, amount_usd: float) -> dict:
    """Issue a credit to an account, in US dollars. Issue once per change."""
    return ctx.context.issue_credit(account_id, amount_usd)


@procedure(api=[read_account, read_policy])
def credit_due(account_id: str) -> dict:
    """Work out the credit an account is owed for its recent plan change, in US dollars, under the billing policy.

    Returns: {"credit_usd": float}, the credit rounded down to the cent.
    """
    ...


def assistant(model: str = MODEL, **kwargs: Any) -> Agent:
    return Agent(name="Billing desk", instructions=PROMPT, model=model, tools=[read_account, read_policy, issue_credit], **kwargs)


def check(ctx: RunContextWrapper[BillingWorld], result: Any) -> dict:
    """Finance's verdict on what was issued."""
    return ctx.context.check()


class WorldAPI:
    """What a procedure reads in function-level practice: the case's own world, read-only."""

    def __init__(self, world: BillingWorld) -> None:
        self.read_account = world.read_account
        self.read_policy = world.read_policy


async def run_on_case(source: str, task: dict[str, Any]) -> Any:
    """A function's output on one case, under the code-skill guard."""
    world = BillingWorld(task)
    try:
        tool = skill_tool("credit_due", "", source, bind={"api": WorldAPI(world)})
        return await tool.call(account_id=task["account"]["account_id"])
    except Exception as exc:  # noqa: BLE001 - refused by the guard or not callable with the contract's arguments
        return {"error": f"credit_due could not be called: {type(exc).__name__}: {exc}"}


async def practice(source: str, case_id: str) -> tuple[float, str, str]:
    """Run a candidate on one case, issue what it returns, and ask the checker."""
    task = task_by_id(case_id)
    output = await run_on_case(source, task)
    observed = f"called with account_id={task['account']['account_id']!r}; returned {json.dumps(output, default=str)[:600]}"
    if isinstance(output, dict) and "error" in output:
        return 0.0, f"the function crashed when run: {output['error']}", observed
    amount = output.get("credit_usd") if isinstance(output, dict) else None
    if not isinstance(amount, (int, float)):
        return 0.0, f"the function returned {output!r}, not {{\"credit_usd\": float}}", observed
    world = BillingWorld(task)
    world.issue_credit(task["account"]["account_id"], float(amount))
    verdict = world.check()
    return (1.0 if verdict["success"] else 0.0), verdict["reason"], observed
