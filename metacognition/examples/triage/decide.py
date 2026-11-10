"""D7's desk for decision programs: a support world, its read tools, the trusted procedure, and the two actions."""

from __future__ import annotations

from typing import Any

from helix.metacognition.decide import Desk

from examples.triage.world import TriageWorld


def desk() -> Desk:
    def procedures(world: TriageWorld) -> dict[str, Any]:
        async def ticket_plan(ticket_id: str) -> Any:
            return {"plan": world.read_account(ticket_id).get("plan")}

        return {"ticket_plan": ticket_plan}

    return Desk(world=TriageWorld,
                reads=lambda world: {"read_ticket": world.read_ticket, "read_account": world.read_account,
                                     "read_policy": world.read_policy},
                procedures=procedures,
                actions={"triage_ticket": lambda world, ticket_id, severity, team: world.triage_ticket(ticket_id, severity, team),
                         "assign_ticket": lambda world, ticket_id, owner: world.assign_ticket(ticket_id, owner)},
                verdict=lambda world, reply: world.check())
