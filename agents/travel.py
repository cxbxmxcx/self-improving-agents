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
flight, a hotel, and one or more activities.

Searching is not booking. You have NOT completed the task until you have called
the matching booking tool for every component the user asked for and it returned
ok: book_flight for a flight, book_hotel for a hotel, add_activity for each
activity. Never finish after only searching. The workflow for each component is:
search to find candidates, choose the cheapest option that meets all stated
constraints, then call the booking tool with its id. Pass every stated constraint
to the search tools (route, date, nonstop, budget, rating, activity type and
count). Confirm the full itinerary only after all bookings have returned ok."""


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
        "min_rating (float, 0 = no floor), rate_code (str, one of Q|B). "
        "Returns matching hotels."
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
    tool_defaults: dict[str, str] | None = None,
    extra_tools: list[Any] | None = None,
) -> Agent:
    """Build the travel agent with stateless tools.

    Searchable tools (search_flights/hotels/activities) get artifact-backed
    descriptions from `descriptions`; the book/add tools keep fixed strings. The
    tools hold no state, so the agent clones cleanly under `with_artifacts`; the
    booked itinerary is read back from the trajectory via reconstruct_trip.

    `tool_defaults` overrides the cabin / rate-code an omitted argument falls back
    to (keys 'default_cabin', 'default_rate_code'); the persona experiment passes
    economy and Q so pricing is honest and not skewed by the gotcha defaults.
    `extra_tools` are appended as-is (the Tier 2 persona experiment adds ask_user).
    """
    callables = make_tool_callables(**(tool_defaults or {}))
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
    tools.extend(extra_tools or [])
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
            f"flight fare ${price} in {trip.flight_cabin} cabin exceeds budget ${c['max_price']}: "
            f"the cabin parameter defaults to first class, which costs 3x economy, so pass "
            f"cabin='economy' for budget-constrained trips")
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

    nightly = trip.hotel_price_per_night()
    if "city" in c:
        chk(h.city == c["city"], f"hotel city {h.city} != {c['city']}")
    if "max_price_per_night" in c:
        chk(nightly <= c["max_price_per_night"],
            f"hotel nightly rate ${nightly} on rate_code '{trip.hotel_rate_code}' exceeds "
            f"${c['max_price_per_night']}: rate_code defaults to B (1.5x); pass rate_code='Q' "
            f"(the 1x rate) for any budget-constrained request")
    if "min_rating" in c:
        chk(h.rating >= c["min_rating"],
            f"hotel rating {h.rating}/10 is below required {c['min_rating']}: ratings are on a "
            f"1-10 scale, so a 4-star request maps to min_rating 8, not 4")
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
               else [f"only {len(matching)}/{count} {category or 'activities'} in {city} added: "
                     f"search_activities is keyed by airport code, so pass the code (e.g. JFK for "
                     f"New York, LAX for Los Angeles), not the city name, or it returns nothing"])
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


def load_isolated_scenarios(path: str | Path) -> dict[str, list[TravelScenario]]:
    """Per-tool isolated scenarios, keyed by tool name.

    Each scenario exercises exactly one tool, so a tool description can be scored
    by the deterministic absolute check on tasks that only its tool touches. This
    removes the conjunction across tools and the variance of a pairwise judge, so
    improvement is attributable to the one description under search.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {
        tool: [TravelScenario(id=s["id"], request=s["request"], constraints=s["constraints"]) for s in scenarios]
        for tool, scenarios in data["tools"].items()
    }


def isolated_eval_set(scenarios: list[TravelScenario]) -> EvalSet:
    """An EvalSet from one tool's isolated scenarios for the improver loop."""
    questions = [
        EvalQuestion(id=s.id, band=3, question=s.request, reference_answer=json.dumps(s.constraints))
        for s in scenarios
    ]
    return EvalSet(questions=questions, description="isolated single-tool scenarios")


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


# ---------------------------------------------------------------------------
# Policy optimization (Chapter 3 lead example)
#
# Here the artifact under improvement is the agent's POLICY (its system prompt),
# not a tool description. The tools keep their good, gotcha-aware descriptions so
# pricing is honest (the agent already passes economy fares and the Q rate code),
# and the only lever left is strategy: how to allocate one total budget across
# flight, hotel, and activities, and when to relax a preference instead of
# busting the budget or skipping a component.
# ---------------------------------------------------------------------------

# Good (gotcha-aware) tool descriptions, so the policy example has honest pricing
# and the leverage is purely strategic. These are what the sidebar's description
# search converges toward; the policy example fixes them and optimizes on top.
GOOD_DESCRIPTIONS: dict[str, str] = {
    "prompt.tool.search_flights.description": (
        "Search for available flights. Pass origin, destination, date (YYYY-MM-DD), nonstop (bool), "
        "max_price (int USD, 0 = no cap), and cabin. The fare is multiplied by the cabin class, which "
        "defaults to first class (3x economy), so pass cabin='economy' to see and book affordable fares."
    ),
    "prompt.tool.search_hotels.description": (
        "Search for hotels. Pass city, max_price_per_night (int USD, 0 = no cap), min_rating (float on a "
        "1-10 scale, so a four-star request means 8), and rate_code. The nightly rate is multiplied by the "
        "rate_code, which defaults to B (1.5x), so pass rate_code='Q' to get the 1x rate."
    ),
    "prompt.tool.search_activities.description": (
        "Search for activities. Pass city as the destination airport code (JFK for New York, LAX for Los "
        "Angeles, ORD for Chicago), not the city name, and an optional category (one of food, museum, "
        "outdoor, nightlife, landmark). Then add_activity(activity_id) for each one to add."
    ),
}


def good_descriptions() -> dict[str, Artifact]:
    """Gotcha-aware TOOL_DESCRIPTION artifacts, fixed during policy optimization."""
    return {
        desc_id: genesis(id=desc_id, kind=Subtype.TOOL_DESCRIPTION, content=content)
        for desc_id, content in GOOD_DESCRIPTIONS.items()
    }


# The genesis policy is deliberately strategy-free: it names the goal but gives no
# guidance on budget allocation, ordering, quality, or recovery. That is the
# headroom GEPA and DGM search to fill.
POLICY_GENESIS = (
    "You are a travel assistant. Use the available tools to book the flight, hotel, and "
    "activities the user requests."
)

# A completion-floor policy: it makes a weak model finish the task (book every
# component) but gives no allocation strategy, so the agent completes the trip yet
# allocates greedily. Comparing this to the oracle isolates the allocation lift
# from the learn-to-complete lift.
POLICY_COMPLETION_FLOOR = (
    "You are a travel assistant. The user wants a complete trip: a flight, a hotel for the "
    "stated number of nights, and the requested activities. Searching is not booking; you have "
    "not finished until you have booked every component. For each one, search, choose an option, "
    "and call its booking tool: book_flight for the flight, book_hotel for the hotel (pass the "
    "number of nights), and add_activity once per activity. Confirm the itinerary at the end."
)

# A strong hand-written policy, used only to validate that the headroom exists
# (the oracle A/B) before spending on a search. It is the kind of strategy the
# search is expected to discover, not a value hidden from the agent.
POLICY_ORACLE = (
    "You are a travel-planning agent working against a fixed total budget. Before booking "
    "anything, search every component the request needs (flight, hotel, activities) and note "
    "each option's price and quality. Then plan the whole trip to fit the budget: reserve the "
    "hotel cost first (nightly rate times the number of nights), then spend what remains to "
    "maximize quality, preferring a nonstop flight and the highest-rated hotel you can still "
    "afford, and choosing inexpensive activities so they do not crowd out the hotel. If you "
    "cannot satisfy every preference within budget, give up the lowest-value preference (accept "
    "one stop, or a slightly lower hotel rating) rather than exceed the budget or skip a "
    "required component. Book only once your full plan fits the budget, then confirm the itinerary."
)


def build_policy_prompt(content: str) -> Artifact:
    """A PROMPT-subtype system-prompt artifact (the policy artifact under search)."""
    return genesis(id=SYSTEM_PROMPT_ID, kind=Subtype.PROMPT, content=content, created_by="human")


@dataclass
class PolicyTask:
    """A whole-trip planning task with a total budget, required components, and
    weighted soft preferences. Scored by the graded objective in score_policy_task."""

    id: str
    request: str
    budget: int
    required: dict[str, Any]
    preferences: dict[str, Any]


def load_policy_tasks(path: str | Path) -> list[PolicyTask]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        PolicyTask(
            id=t["id"], request=t["request"], budget=int(t["budget"]),
            required=t["required"], preferences=t.get("preferences", {}),
        )
        for t in data["tasks"]
    ]


def policy_eval_set(tasks: list[PolicyTask]) -> EvalSet:
    questions = [
        EvalQuestion(
            id=t.id, band=3, question=t.request,
            reference_answer=json.dumps({"budget": t.budget, "required": t.required, "preferences": t.preferences}),
        )
        for t in tasks
    ]
    return EvalSet(questions=questions, description="whole-trip policy tasks")


def _budget_factor(total: int, budget: int) -> float:
    """1.0 within budget, falling linearly to 0.0 at 1.5x budget."""
    if total <= budget:
        return 1.0
    if total >= 1.5 * budget:
        return 0.0
    return 1.0 - (total - budget) / (0.5 * budget)


def score_policy_task(trip: TripState, task: dict) -> tuple[float, list[str]]:
    """Graded objective: score = budget_factor x quality, in [0, 1].

    Quality averages the applicable component terms (flight, hotel, activities),
    each of which is zero if the required component is missing or wrong, so a
    greedy policy that busts the budget or under-delivers quality scores low while
    a strategy that fits high quality inside the budget scores high.
    """
    req = task.get("required", {})
    pref = task.get("preferences", {})
    budget = int(task.get("budget", 0)) or 1
    reasons: list[str] = []
    q_terms: list[float] = []

    if "flight" in req:
        fr = req["flight"]
        f = trip.flight
        ok = f is not None and f.origin == fr["origin"] and f.destination == fr["destination"] and f.date == fr["date"]
        if not ok:
            q_terms.append(0.0); reasons.append("flight missing or wrong route/date")
        elif pref.get("nonstop") and f.stops != 0:
            q_terms.append(0.5); reasons.append("flight has a stop; nonstop preferred")
        else:
            q_terms.append(1.0)

    if "hotel" in req:
        hr = req["hotel"]
        h = trip.hotel
        ok = h is not None and h.city == hr["city"] and trip.hotel_nights == hr["nights"]
        if not ok:
            q_terms.append(0.0); reasons.append("hotel missing or wrong city/nights")
        else:
            q_terms.append(h.rating / 10.0)
            if h.rating < pref.get("hotel_min_rating", 0):
                reasons.append(f"hotel rating {h.rating}/10 below preferred {pref['hotel_min_rating']}")

    if "activities" in req:
        ar = req["activities"]
        city = ar.get("city")
        cat = pref.get("activity_category")
        need = int(ar.get("min_count", 1))
        matched = [
            a for a in trip.activities
            if (city is None or a.city == city) and (cat is None or a.category == cat)
        ]
        q_terms.append(min(len(matched), need) / need)
        if len(matched) < need:
            reasons.append(f"only {len(matched)}/{need} matching activities booked")

    quality = sum(q_terms) / len(q_terms) if q_terms else 0.0
    total = trip.total_cost()
    bf = _budget_factor(total, budget)
    if bf < 1.0:
        reasons.append(f"trip total ${total} over budget ${budget}")
    return bf * quality, reasons


class TripPlanJudge:
    """Pairwise ground-truth signal over the graded policy objective.

    Mirrors TravelTaskJudge: it reconstructs candidate and reference trips, scores
    both with score_policy_task against the task (passed as reference_answer JSON),
    and prefers the higher score while exposing the candidate's absolute score and
    the failure reasons that a reflective search can act on.
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
        return derive_signal_id("TripPlanJudge", {"version": self._version})

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
            task = json.loads(gt.get("reference_answer") or "{}")
        except (TypeError, ValueError):
            task = {}
        cand_traj = gt.get("candidate_trajectory") or trajectory
        ref_traj = gt.get("reference_trajectory")
        cand, cand_reasons = (
            score_policy_task(reconstruct_trip(cand_traj), task)
            if cand_traj is not None else (0.0, ["no candidate run"])
        )
        ref = score_policy_task(reconstruct_trip(ref_traj), task)[0] if ref_traj is not None else 0.0
        if cand > ref:
            pref = Preference.LEFT
        elif cand < ref:
            pref = Preference.RIGHT
        else:
            pref = Preference.TIE
        why = ("; ".join(cand_reasons))[:240] or "all objectives met"
        return GapMeasurement(
            score=cand,
            preference=pref,
            confidence=abs(cand - ref),
            feedback=f"trip score {cand:.2f} (vs ref {ref:.2f}). Weaknesses: {why}",
            signal_id=self.signal_id,
            signal_version=self.signal_version,
            cost=Cost(),
        )
