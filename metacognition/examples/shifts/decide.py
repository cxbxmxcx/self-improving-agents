"""D5's desk for decision programs: an on-call world, its read tools, the trusted procedure, and the two actions."""

from __future__ import annotations

from typing import Any

from helix.metacognition.decide import Desk

from examples.shifts.world import ShiftWorld


def desk() -> Desk:
    def procedures(world: ShiftWorld) -> dict[str, Any]:
        async def available_people(day: str) -> Any:
            leave = world.read_leave()["leave"]
            wanted = day.strip().capitalize()
            return {"people": [person for person in world.read_team()["people"] if wanted not in leave.get(person, [])]}

        return {"available_people": available_people}

    return Desk(world=ShiftWorld,
                reads=lambda world: {"read_team": world.read_team, "read_leave": world.read_leave,
                                     "read_rota": world.read_rota},
                procedures=procedures,
                actions={"assign_shift": lambda world, day, person: world.assign_shift(day, person),
                         "publish_rota": lambda world, **days: world.publish_rota(**days)},
                verdict=lambda world, reply: world.check())
