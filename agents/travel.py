"""Travel task agent: assemble a trip with simulated flight/hotel/activity tools.

This is the Chapter 3 task agent. Unlike the RAG agent of Chapter 2, it has
several tools, so the natural-language tool *descriptions* drive which tool the
model picks and which constraints it passes. Vague descriptions make the agent
miss "nonstop", "under budget", or "two food activities"; the chapter's search
methods (GEPA especially) evolve the descriptions to fix that, graded by a
deterministic task-success signal (helix.signals.task_success).

What the chapter searches over is one TOOL_DESCRIPTION artifact at a time (e.g.
`prompt.tool.search_flights.description`). The implementations stay plain Python
in agents.travel_sim; only the description is the artifact under improvement.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from helix.agent import Agent
from helix.artifact import Artifact, Subtype, genesis
from helix.tools import TextDescriptionTool, tool

from agents.travel_sim import (
    FIXED_DESCRIPTIONS,
    GENESIS_DESCRIPTIONS,
    SEARCHABLE_TOOL_DESCRIPTION_IDS,
    TripState,
    make_tool_callables,
)

DEFAULT_MODEL = "claude-haiku-4-5"
SYSTEM_PROMPT_ID = "prompt.travel.system"

DEFAULT_SYSTEM_PROMPT = """\
You are a travel assistant. Plan and book a trip that satisfies the user's
request using the available tools. Search before you book, respect every stated
constraint (route, date, nonstop, budget, rating, activity type and count), and
finish by confirming the itinerary. Prefer the cheapest option that still meets
all constraints."""


def build_genesis_prompt() -> Artifact:
    return genesis(
        id=SYSTEM_PROMPT_ID,
        kind=Subtype.PROMPT,
        content=DEFAULT_SYSTEM_PROMPT,
        created_by="human",
    )


def genesis_descriptions() -> dict[str, Artifact]:
    """The genesis TOOL_DESCRIPTION artifacts for the searchable tools."""
    return {
        desc_id: genesis(id=desc_id, kind=Subtype.TOOL_DESCRIPTION, content=content)
        for desc_id, content in GENESIS_DESCRIPTIONS.items()
    }


def build_travel_agent(
    descriptions: dict[str, Artifact],
    trip: TripState,
    *,
    system_prompt: Artifact | None = None,
    model: str = DEFAULT_MODEL,
) -> Agent:
    """Build the travel agent with its tools bound to one TripState.

    Searchable tools (search_flights/hotels/activities) get artifact-backed
    descriptions from `descriptions`; the book/add tools keep fixed strings.
    A fresh TripState per build keeps each scenario run isolated.
    """
    callables = make_tool_callables(trip)
    tools: list[Any] = []
    for tool_name, desc_id in SEARCHABLE_TOOL_DESCRIPTION_IDS.items():
        tools.append(
            TextDescriptionTool(
                name=tool_name,
                fn=callables[tool_name],
                description_artifact=descriptions[desc_id],
            )
        )
    for tool_name, desc in FIXED_DESCRIPTIONS.items():
        tools.append(tool(description=desc)(callables[tool_name]))
    return Agent(
        system_prompt=system_prompt or build_genesis_prompt(),
        tools=tools,
        model=model,
    )


# ---------------------------------------------------------------------------
# Scenarios and the deterministic task-success check
# ---------------------------------------------------------------------------

@dataclass
class TravelScenario:
    id: str
    request: str
    constraints: dict[str, Any]

    def score(self, trip: TripState) -> float:
        """Fraction of the scenario's constraints the booked trip satisfies.

        Each specified group (flight/hotel/activities) contributes equally, and
        within a group each checked field contributes equally, so the score is a
        smooth [0, 1] gradient the search can climb rather than all-or-nothing.
        """
        groups: list[float] = []
        if "flight" in self.constraints:
            groups.append(_score_flight(trip.flight, self.constraints["flight"]))
        if "hotel" in self.constraints:
            groups.append(_score_hotel(trip.hotel, trip.hotel_nights, self.constraints["hotel"]))
        if "activities" in self.constraints:
            groups.append(_score_activities(trip.activities, self.constraints["activities"]))
        return sum(groups) / len(groups) if groups else 0.0


def _frac(checks: list[bool]) -> float:
    return sum(1 for c in checks if c) / len(checks) if checks else 1.0


def _score_flight(flight, c: dict) -> float:
    if flight is None:
        return 0.0
    checks: list[bool] = []
    if "origin" in c:
        checks.append(flight.origin == c["origin"])
    if "destination" in c:
        checks.append(flight.destination == c["destination"])
    if "date" in c:
        checks.append(flight.date == c["date"])
    if "nonstop" in c:
        checks.append((flight.stops == 0) == bool(c["nonstop"]))
    if "max_price" in c:
        checks.append(flight.price <= c["max_price"])
    if "airline" in c:
        checks.append(flight.airline == c["airline"])
    return _frac(checks)


def _score_hotel(hotel, nights: int, c: dict) -> float:
    if hotel is None:
        return 0.0
    checks: list[bool] = []
    if "city" in c:
        checks.append(hotel.city == c["city"])
    if "max_price_per_night" in c:
        checks.append(hotel.price_per_night <= c["max_price_per_night"])
    if "min_rating" in c:
        checks.append(hotel.rating >= c["min_rating"])
    if "nights" in c:
        checks.append(nights == c["nights"])
    if "amenities" in c:
        checks.append(all(a in hotel.amenities for a in c["amenities"]))
    return _frac(checks)


def _score_activities(activities: list, c: dict) -> float:
    city = c.get("city")
    category = c.get("category")
    count = int(c.get("count", 1))
    matching = [
        a for a in activities
        if (city is None or a.city == city) and (category is None or a.category == category)
    ]
    return 1.0 if len(matching) >= count else len(matching) / count


def load_travel_scenarios(path: str | Path) -> list[TravelScenario]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        TravelScenario(id=s["id"], request=s["request"], constraints=s["constraints"])
        for s in data["scenarios"]
    ]


def make_score_scenario(
    base_descriptions: dict[str, Artifact],
    *,
    system_prompt: Artifact | None = None,
    model: str = DEFAULT_MODEL,
):
    """Build the score_scenario coroutine TaskSuccessSignal consumes.

    For one candidate tool description, it substitutes the candidate for its own
    id, leaves the other descriptions at their current versions, runs the agent
    on the scenario against a fresh TripState, and scores the booked trip.
    """
    async def score_scenario(candidate: Artifact, scenario: TravelScenario) -> tuple[float, dict]:
        descriptions = dict(base_descriptions)
        descriptions[candidate.id] = candidate
        trip = TripState()
        agent = build_travel_agent(descriptions, trip, system_prompt=system_prompt, model=model)
        answer, _trajectory = await agent.run(scenario.request)
        return scenario.score(trip), {
            "booked": trip.summary(),
            "answer_excerpt": (answer or "")[:160],
        }

    return score_scenario
