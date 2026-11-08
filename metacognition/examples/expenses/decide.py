"""D1's desk for decision programs: an expense world, its read tools, the trusted procedure, and file_claim."""

from __future__ import annotations

from typing import Any

from helix.memory.code_skills import skill_tool
from helix.metacognition.decide import Desk

from examples.expenses.agent import REFERENCE, WorldAPI
from examples.expenses.world import ExpenseWorld


def desk(procedure: str = REFERENCE) -> Desk:
    """`procedure` is the trusted reimbursable_total the program may call (the hand-written one by default)."""

    def procedures(world: ExpenseWorld) -> dict[str, Any]:
        async def reimbursable_total(trip_id: str) -> Any:
            tool = skill_tool("reimbursable_total", "", procedure, bind={"api": WorldAPI(world)})
            return await tool.call(trip_id=trip_id)

        return {"reimbursable_total": reimbursable_total}

    return Desk(world=ExpenseWorld,
                reads=lambda world: {"read_trip": world.read_trip, "read_policy": world.read_policy},
                procedures=procedures,
                actions={"file_claim": lambda world, trip_id, amount_usd: world.file_claim(trip_id, amount_usd)},
                verdict=lambda world, reply: world.check())
