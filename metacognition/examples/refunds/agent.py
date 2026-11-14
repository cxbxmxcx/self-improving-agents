"""The returns desk agent, written on metacog: tools, the procedure worth learning, and how runs are judged."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from helix.memory.code_skills import skill_tool
from helix.metacognition import procedure

from examples.refunds.world import STORE, ReturnsWorld, task_by_id
from metacog import Agent, RunContextWrapper, function_tool

MODEL = "gpt-5.4-mini"
PROMPT = f"""You work the returns desk at {STORE}. Process the return the user names with the tools: read
the return, its order and the customer, read the store's returns policy, and work out what the return gets.
Then act once: issue the refund (to the customer's card, or as store credit) or decline the return. When
you are done, reply with what you did."""
MEMORIES = Path(__file__).parent / "memories"
# The hand-written procedure for the policy as it stands, holiday rule included.
REFERENCE = (MEMORIES / "refund_decision.py").read_text(encoding="utf-8")
# The same procedure as written before the holiday policy existed: where D2's online loop starts.
REGULAR = (MEMORIES / "refund_decision_regular.py").read_text(encoding="utf-8")
# How the first pass's procedure names an action, and the action tools' terms for it (the second pass).
TOOL_TERMS = {"refund": "card", "credit": "store_credit", "decline": "decline"}


def in_tool_terms(source: str) -> str:
    """A procedure that answers {"action": "refund"} rewritten to answer {"method": "card"}: the same code, in
    the terms the action tools take. D2's second pass changes nothing else."""
    for action, method in TOOL_TERMS.items():
        source = source.replace(f'{{"action": "{action}"', f'{{"method": "{method}"')
    if '"action"' in source:
        raise ValueError("a decision is left in the procedure's own words")
    return source


@function_tool(read_only=True)
def read_return(ctx: RunContextWrapper[ReturnsWorld], return_id: str) -> dict:
    """Read a return request: its order and item, the customer's reason, whether the item was opened, any
    accessories missing from it, and the date it was requested."""
    return ctx.context.read_return(return_id)


@function_tool(read_only=True)
def read_order(ctx: RunContextWrapper[ReturnsWorld], order_id: str) -> dict:
    """Read an order: its customer, when it was ordered and delivered, and each item with its price, shipping,
    final-sale flag and accessories."""
    return ctx.context.read_order(order_id)


@function_tool(read_only=True)
def read_customer(ctx: RunContextWrapper[ReturnsWorld], customer_id: str) -> dict:
    """Read a customer's record, including their membership tier."""
    return ctx.context.read_customer(customer_id)


@function_tool(read_only=True)
def read_policy(ctx: RunContextWrapper[ReturnsWorld]) -> dict:
    """Read the store's returns policy: return windows by category, fees, and the policy notes."""
    return ctx.context.read_policy()


@function_tool
def issue_refund(ctx: RunContextWrapper[ReturnsWorld], return_id: str, amount_usd: float,
                 method: Literal["card", "store_credit"]) -> dict:
    """Issue the refund for a return, in US dollars: back to the customer's card, or as store credit."""
    return ctx.context.issue_refund(return_id, amount_usd, method)


@function_tool
def decline_return(ctx: RunContextWrapper[ReturnsWorld], return_id: str, reason: str) -> dict:
    """Decline a return, with the reason to give the customer."""
    return ctx.context.decline_return(return_id, reason)


READS = [read_return, read_order, read_customer, read_policy]


@procedure(api=READS)
def refund_decision(return_id: str) -> dict:
    """Work out what a return gets under the store's returns policy: a refund to the card, store credit, or a
    decline.

    Returns: {"action": "refund" | "credit" | "decline", "amount_usd": float}, the amount rounded to the cent
    (0 for a decline).
    """
    ...


@procedure(api=READS, name="refund_decision")
def refund_decision_in_tool_terms(return_id: str) -> dict:
    """Work out what a return gets under the store's returns policy, in the terms the action tools take.

    Returns: {"method": "card" | "store_credit" | "decline", "amount_usd": float}: the method to pass to
    issue_refund, or "decline" to call decline_return instead; the amount rounded to the cent (0 for a decline).
    """
    ...


@procedure(api=READS, name="refund_decision")
def refund_decision_to_act_on(return_id: str) -> dict:
    """Work out what a return gets under the store's returns policy, in the terms the action tools take. It
    reads the return, its order, the customer and the policy itself: call it first and act on its answer,
    without reading them again.

    Returns: {"method": "card" | "store_credit" | "decline", "amount_usd": float}: the method to pass to
    issue_refund, or "decline" to call decline_return instead; the amount rounded to the cent (0 for a decline).
    """
    ...


# The contracts D2 compares: the first pass's procedure answers in its own words, the second's in the action
# tools' terms, and the third also tells the agent to act on its answer rather than redo its reads. Each comes
# with its hand-written procedure and the one written before the holiday policy.
CONTRACTS = {
    "words": (refund_decision, REFERENCE, REGULAR),
    "tool": (refund_decision_in_tool_terms, in_tool_terms(REFERENCE), in_tool_terms(REGULAR)),
    "act": (refund_decision_to_act_on, in_tool_terms(REFERENCE), in_tool_terms(REGULAR)),
}


def returns_desk(**kwargs: Any) -> Agent:
    return Agent(name="Larkspur returns desk", instructions=PROMPT, model=MODEL,
                 tools=[*READS, issue_refund, decline_return], **kwargs)


def check(ctx: RunContextWrapper[ReturnsWorld], result: Any) -> dict:
    """The store's audit of what was done with the return."""
    return ctx.context.check()


class WorldAPI:
    """What a procedure reads in function-level practice: the case's own world, read-only."""

    def __init__(self, world: ReturnsWorld) -> None:
        self.read_return = world.read_return
        self.read_order = world.read_order
        self.read_customer = world.read_customer
        self.read_policy = world.read_policy


async def run_on_case(source: str, task: dict[str, Any]) -> Any:
    """A function's decision on one case, under the code-skill guard."""
    try:
        tool = skill_tool("refund_decision", "", source, bind={"api": WorldAPI(ReturnsWorld(task))})
        return await tool.call(return_id=task["return"]["return_id"])
    except Exception as exc:  # noqa: BLE001 - refused by the guard or not callable with the contract's arguments
        return {"error": f"refund_decision could not be called: {type(exc).__name__}: {exc}"}


def method_of(decision: Any) -> str | None:
    """The action tools' method for a decision in either contract: card, store_credit or decline."""
    if not isinstance(decision, dict):
        return None
    method = decision.get("method") if "method" in decision else TOOL_TERMS.get(decision.get("action"))
    return method if method in ("card", "store_credit", "decline") else None


def apply(world: ReturnsWorld, return_id: str, decision: dict[str, Any]) -> None:
    """Carry out a procedure's decision the way the agent would."""
    method, amount = method_of(decision), float(decision.get("amount_usd") or 0.0)
    if method == "decline":
        world.decline_return(return_id, "under the returns policy")
    else:
        world.issue_refund(return_id, amount, method)


async def practice(source: str, case_id: str) -> tuple[float, str, str]:
    """Chapter 6's act for this world: run the function on the case, carry out its decision, ask the audit."""
    task = task_by_id(case_id)
    decision = await run_on_case(source, task)
    return_id = task["return"]["return_id"]
    observed = f"called with return_id={return_id!r}; returned {json.dumps(decision, default=str)[:600]}"
    if isinstance(decision, dict) and "error" in decision:
        return 0.0, f"the function crashed when run: {decision['error']}", observed
    if method_of(decision) is None:
        return 0.0, f"the function returned {decision!r}, not an action and an amount", observed
    world = ReturnsWorld(task)
    apply(world, return_id, decision)
    verdict = world.check()
    return (1.0 if verdict["success"] else 0.0), verdict["reason"], observed
