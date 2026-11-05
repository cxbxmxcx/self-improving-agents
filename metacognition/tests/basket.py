"""A small scripted domain for metacognition tests: a basket total that drifts when tax appears.

v1 of the procedure sums prices; after the drift every item also carries a
tax, so v1's totals are rejected, the detector quarantines it, the agent
answers at the no-memory setting, and the learner writes v2, which the gate
admits. Agent and writer replies come from a ScriptedModel, in order.
"""

from __future__ import annotations

import asyncio

from helix.memory.code_skills import skill_tool

from metacog import Agent, RunConfig, RunContextWrapper, Runner, function_tool
from metacog.meta import GENERIC, Controller, Gate, Metacognition, ValueLearning, Verdict, procedure
from metacog.testing import call, text


@function_tool(read_only=True)
def price(ctx: RunContextWrapper[dict], item: str) -> int:
    """The price of one item."""
    return ctx.context["prices"][item]


@function_tool(read_only=True)
async def tax(ctx: RunContextWrapper[dict]) -> int:
    """The tax added to each item."""
    await asyncio.sleep(0)
    return ctx.context["tax"]


@procedure(api=[price, tax])
def basket_total(items: list[str]) -> int:
    """Add up what the listed items cost.

    Returns: the total as an integer.
    """
    ...


V1 = 'def basket_total(api, items: list[str]) -> int:\n    return sum(api.price(item) for item in items)\n'
V2 = ('def basket_total(api, items: list[str]) -> int:\n'
      '    return sum(api.price(item) + api.tax() for item in items)\n')
PRICES = {"apple": 2, "pear": 3}
WRITTEN = text(f"```python\n{V2}```")


def world(items: list[str], tax: int = 0) -> dict:
    return {"prices": PRICES, "tax": tax, "items": items, "expected": sum(PRICES[i] + tax for i in items)}


CASES = {"S1": world(["apple", "pear"]), "D1": world(["apple", "pear"], 1), "D2": world(["pear", "pear"], 1),
         "A": world(["apple", "pear"], 1), "B": world(["pear", "pear"], 1), "D3": world(["apple", "apple"], 1),
         "G1": world(["apple"], 1), "G2": world(["pear", "apple", "apple"], 1)}


class CaseAPI:
    def __init__(self, case: dict) -> None:
        self.case = case

    def price(self, item: str) -> int:
        return self.case["prices"][item]

    def tax(self) -> int:
        return self.case["tax"]


async def practice(source: str, case_id: str) -> tuple[float, str, str]:
    """Function-level practice, as chapter 6's act: run the source on the case's own world."""
    case = CASES[case_id]
    got = await skill_tool("basket_total", "", source, bind={"api": CaseAPI(case)}).call(items=case["items"])
    ok = got == case["expected"]
    return (1.0 if ok else 0.0), "accepted" if ok else f"returned {got}, expected {case['expected']}", \
        f"called with items={case['items']}, returned {got}"


def verifier(ctx: RunContextWrapper[dict], result) -> Verdict:
    expected = str(ctx.context["expected"])
    return Verdict(result.final_output.strip() == expected, f"the total is {expected}")


def make_meta(**kwargs) -> Metacognition:
    options = dict(verifier=verifier, learner=ValueLearning(GENERIC), practice=practice,
                   gate=Gate(cases=["G1", "G2"], bar=1.0),
                   controller=Controller({"memory": "none", "no_memory": "high"}),
                   recall_k=1, relearn_after=2, learning="inline")
    options.update(kwargs)
    return Metacognition([basket_total], **options)


def agent(meta: Metacognition | None) -> Agent:
    return Agent(name="Shop", instructions="Total the basket.", tools=[price, tax], model="gpt-5.4-mini",
                 metacognition=meta)


def uses(items: list[str], answer: int) -> list:
    """The agent's replies when it calls the procedure and reports its total."""
    return [call("basket_total", {"items": items}), text(str(answer))]


async def serve(shop: Agent, case_id: str):
    case = CASES[case_id]
    return await Runner.run(shop, f"What do {', '.join(case['items'])} cost?", context=case,
                            run_config=RunConfig(case_id=case_id, case_data=case))
