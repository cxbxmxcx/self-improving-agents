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
from helix.eval.dataset import EvalQuestion, EvalSet
from helix.signal import (
    Cost,
    GapMeasurement,
    Preference,
    SignalKind,
    derive_signal_id,
)
from helix.tools import TextDescriptionTool, tool
from helix.trajectory import Trajectory

from agents.travel_sim import (
    FIXED_DESCRIPTIONS,
    GENESIS_DESCRIPTIONS,
    SEARCHABLE_TOOL_DESCRIPTION_IDS,
    TripState,
    make_tool_callables,
    reconstruct_trip,
)

DEFAULT_MODEL = "claude-haiku-4-5"
SYSTEM_PROMPT_ID = "prompt.travel.system"

DEFAULT_SYSTEM_PROMPT = """\
You are a travel assistant. Plan and book a trip that satisfies the user's
request using the available tools. A request may have several components: a
flight, a hotel, and one or more activities. Book every component the user asks
for before you finish; do not stop after the flight. Search before you book,
pass every stated constraint to the search tools (route, date, nonstop, budget,
rating, activity type and count), prefer the cheapest option that still meets
all constraints, and confirm the full itinerary at the end."""


# Real tool signatures, used to ground the description-mutation prompt so the
# proposer mentions the parameters that exist and never invents ones that do not.
TOOL_SCHEMAS: dict[str, str] = {
    "search_flights": (
        "origin (str, e.g. SFO), destination (str), date (str YYYY-MM-DD), "
        "nonstop (bool), max_price (int USD, 0 = no cap), cabin (str). "
        "Returns matching flights sorted cheapest first."
    ),
    "search_hotels": (
        "city (str), max_price_per_night (int USD, 0 = no cap), "
        "min_rating (float, 0 = no floor). Returns matching hotels."
    ),
    "search_activities": (
        "city (str), category (str, one of food|museum|outdoor|nightlife|"
        "landmark; empty = all). Returns matching activities."
    ),
}


def grounded_mutation_prompt(tool_name: str) -> str:
    """A mutation system prompt grounded in the tool's real signature.

    The framework's generic mutators see only the description text, so without
    this they invent plausible-but-wrong parameters. Feeding the real schema is
    what lets the search write descriptions the agent can actually act on.
    """
    return (
        f"You are improving the natural-language DESCRIPTION of the `{tool_name}` "
        f"tool that an LLM agent reads to decide how to call it.\n\n"
        f"The tool's real parameters: {TOOL_SCHEMAS[tool_name]}\n\n"
        f"Rewrite the description so the agent reliably passes every relevant "
        f"parameter whenever the user states a matching constraint. Name the "
        f"parameters explicitly. Never invent parameters the tool does not have. "
        f"Output only the new description, one or two sentences, no preamble."
    )


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
    *,
    system_prompt: Artifact | None = None,
    model: str = DEFAULT_MODEL,
) -> Agent:
    """Build the travel agent with stateless tools.

    Searchable tools (search_flights/hotels/activities) get artifact-backed
    descriptions from `descriptions`; the book/add tools keep fixed strings. The
    tools hold no state, so the agent clones cleanly under `with_artifacts`; the
    booked itinerary is read back from the trajectory via reconstruct_trip.
    """
    callables = make_tool_callables()
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
        return self.score_with_reasons(trip)[0]

    def score_with_reasons(self, trip: TripState) -> tuple[float, list[str]]:
        """Fraction of constraints satisfied, plus a reason for each failure.

        Each group (flight/hotel/activities) contributes equally; within a group
        each field contributes equally, so the score is a smooth gradient. The
        reasons feed the reflective search so it can discover the gotchas (the
        cabin multiplier, the 1-10 rating scale, the exact category values).
        """
        groups: list[float] = []
        reasons: list[str] = []
        if "flight" in self.constraints:
            f, r = _score_flight(trip, self.constraints["flight"])
            groups.append(f); reasons += r
        if "hotel" in self.constraints:
            f, r = _score_hotel(trip, self.constraints["hotel"])
            groups.append(f); reasons += r
        if "activities" in self.constraints:
            f, r = _score_activities(trip, self.constraints["activities"])
            groups.append(f); reasons += r
        return (sum(groups) / len(groups) if groups else 0.0), reasons


def _frac(checks: list[bool]) -> float:
    return sum(1 for c in checks if c) / len(checks) if checks else 1.0


def _score_flight(trip: TripState, c: dict) -> tuple[float, list[str]]:
    f = trip.flight
    if f is None:
        return 0.0, ["no flight booked"]
    price = trip.flight_price()
    checks: list[bool] = []
    reasons: list[str] = []

    def chk(ok: bool, msg: str) -> None:
        checks.append(ok)
        if not ok:
            reasons.append(msg)

    if "origin" in c:
        chk(f.origin == c["origin"], f"flight origin {f.origin} != {c['origin']}")
    if "destination" in c:
        chk(f.destination == c["destination"], f"flight destination {f.destination} != {c['destination']}")
    if "date" in c:
        chk(f.date == c["date"], f"flight date {f.date} != {c['date']}")
    if "nonstop" in c:
        chk((f.stops == 0) == bool(c["nonstop"]),
            "flight is not nonstop" if c["nonstop"] else "flight should allow stops")
    if "max_price" in c:
        chk(price <= c["max_price"],
            f"flight fare ${price} in {trip.flight_cabin} cabin exceeds budget ${c['max_price']}")
    return _frac(checks), reasons


def _score_hotel(trip: TripState, c: dict) -> tuple[float, list[str]]:
    h = trip.hotel
    if h is None:
        return 0.0, ["no hotel booked"]
    checks: list[bool] = []
    reasons: list[str] = []

    def chk(ok: bool, msg: str) -> None:
        checks.append(ok)
        if not ok:
            reasons.append(msg)

    if "city" in c:
        chk(h.city == c["city"], f"hotel city {h.city} != {c['city']}")
    if "max_price_per_night" in c:
        chk(h.price_per_night <= c["max_price_per_night"],
            f"hotel ${h.price_per_night}/night exceeds ${c['max_price_per_night']}")
    if "min_rating" in c:
        chk(h.rating >= c["min_rating"],
            f"hotel rating {h.rating}/10 is below required {c['min_rating']}")
    if "nights" in c:
        chk(trip.hotel_nights == c["nights"], f"booked {trip.hotel_nights} nights, wanted {c['nights']}")
    if "amenities" in c:
        chk(all(a in h.amenities for a in c["amenities"]), "hotel missing a requested amenity")
    return _frac(checks), reasons


def _score_activities(trip: TripState, c: dict) -> tuple[float, list[str]]:
    city = c.get("city")
    category = c.get("category")
    count = int(c.get("count", 1))
    matching = [
        a for a in trip.activities
        if (city is None or a.city == city) and (category is None or a.category == category)
    ]
    frac = 1.0 if len(matching) >= count else len(matching) / count
    reasons = ([] if frac >= 1.0
               else [f"only {len(matching)}/{count} {category or 'activities'} in {city} added"])
    return frac, reasons


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
        agent = build_travel_agent(descriptions, system_prompt=system_prompt, model=model)
        answer, trajectory = await agent.run(scenario.request)
        trip = reconstruct_trip(trajectory)
        score, reasons = scenario.score_with_reasons(trip)
        return score, {
            "booked": trip.summary(),
            "reasons": reasons,
            "answer_excerpt": (answer or "")[:160],
        }

    return score_scenario


def load_travel_eval_set(path: str | Path) -> EvalSet:
    """Travel scenarios as an EvalSet for the framework search path.

    Each scenario becomes an EvalQuestion whose `question` is the request and
    whose `reference_answer` is the JSON of the expected constraints. The
    framework round runs the agent on the question; TravelTaskJudge reads the
    constraints back out of reference_answer.
    """
    questions = [
        EvalQuestion(
            id=s.id, band=3, question=s.request,
            reference_answer=json.dumps(s.constraints),
        )
        for s in load_travel_scenarios(path)
    ]
    return EvalSet(questions=questions, description="travel task scenarios")


class TravelTaskJudge:
    """Pairwise ground-truth signal for the framework search path (§3.4).

    The framework round and GEPA judge candidate vs reference trajectories. This
    signal reconstructs each trip from its trajectory, scores both against the
    scenario constraints (passed as reference_answer JSON in ground_truth), and
    prefers the higher-scoring one. It also fills `score` with the candidate's
    task-success fraction so absolute consumers (GEPA's score sort) work too.
    """

    def __init__(self, version: int = 1) -> None:
        self._version = version

    @property
    def kind(self) -> SignalKind:
        return SignalKind.GROUND_TRUTH

    @property
    def cost_estimate(self) -> Cost:
        return Cost()

    @property
    def signal_id(self) -> str:
        return derive_signal_id("TravelTaskJudge", {"version": self._version})

    @property
    def signal_version(self) -> int:
        return self._version

    async def measure(
        self,
        candidate: Artifact,
        trajectory: Trajectory | None = None,
        reference: Artifact | None = None,
        ground_truth: Any | None = None,
    ) -> GapMeasurement:
        gt = ground_truth or {}
        try:
            constraints = json.loads(gt.get("reference_answer") or "{}")
        except (TypeError, ValueError):
            constraints = {}
        scenario = TravelScenario("_", "", constraints)
        cand_traj = gt.get("candidate_trajectory") or trajectory
        ref_traj = gt.get("reference_trajectory")
        cand, cand_reasons = (
            scenario.score_with_reasons(reconstruct_trip(cand_traj))
            if cand_traj is not None else (0.0, ["no candidate run"])
        )
        ref = scenario.score(reconstruct_trip(ref_traj)) if ref_traj is not None else 0.0
        if cand > ref:
            pref = Preference.LEFT
        elif cand < ref:
            pref = Preference.RIGHT
        else:
            pref = Preference.TIE
        # The failure reasons are what let a reflective mutator discover the
        # gotchas (cabin multiplier, 1-10 ratings, exact categories).
        why = ("; ".join(cand_reasons))[:240] or "all constraints satisfied"
        return GapMeasurement(
            score=cand,
            preference=pref,
            confidence=abs(cand - ref),
            feedback=f"task success {cand:.2f} (vs ref {ref:.2f}). Candidate misses: {why}",
            signal_id=self.signal_id,
            signal_version=self.signal_version,
            cost=Cost(),
        )
