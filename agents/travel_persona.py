"""Persona-rubric optimization (Chapter 3): plan trips for users drawn from a set
of hidden personas, scored by a coarse rubric judge.

Different personas reward different trips, so there is no static answer to bake
into the prompt; the optimum is a strategy that tailors the trip to the traveler.
The judge returns a 1-to-5 rating (kept as a [0,1] fraction) plus one vague
sentence naming only the worst-matched dimension, never the target, so the
feedback cannot be copied into the prompt. A noise knob lets us raise the reward
noise to expose the robustness gap between SPO and the population/archive methods.

See chapters/ch03/PERSONA_EXPERIMENT_DESIGN.md for the full design.
"""

from __future__ import annotations

import contextvars
import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from helix.artifact import Artifact, Subtype, genesis
from helix.eval.dataset import EvalQuestion, EvalSet
from helix.signal import Cost, GapMeasurement, Preference, SignalKind, derive_signal_id
from helix.trajectory import Trajectory

from helix.tools import tool

from agents.travel import SYSTEM_PROMPT_ID, build_travel_agent
from agents.travel_sim import TripState, cabin_price, hotel_price, reconstruct_trip


@dataclass(frozen=True)
class Persona:
    name: str
    cue: str                      # the phrase that signals the persona in a Tier 1 request
    budget_cap: int               # trip total the persona is comfortable up to (USD)
    hotel_min_rating: float       # 0 means the persona does not care about rating
    hotel_amenity: str | None     # a preferred amenity, if any
    cat_weights: dict[str, float] # activity category -> preference weight in [-1, 1]
    pace: int                     # preferred number of activities
    nonstop: bool                 # whether a nonstop flight matters
    cabin: str                    # preferred flight cabin
    dim_weights: dict[str, float] # weight of each dimension in the overall rating


PERSONAS: dict[str, Persona] = {
    "family": Persona(
        name="family", cue="for my family", budget_cap=1000, hotel_min_rating=8.0, hotel_amenity="pool",
        cat_weights={"outdoor": 1.0, "landmark": 1.0, "museum": 0.6, "food": 0.2, "nightlife": -1.5},
        pace=2, nonstop=True, cabin="economy",
        dim_weights={"hotel": 0.30, "activities": 0.35, "flight": 0.10, "budget": 0.25}),
    "luxury": Persona(
        name="luxury", cue="as a luxury getaway for myself", budget_cap=2800, hotel_min_rating=9.0, hotel_amenity=None,
        cat_weights={"food": 1.0, "nightlife": 1.0, "museum": 0.8, "landmark": 0.2, "outdoor": 0.0},
        pace=3, nonstop=True, cabin="business",
        dim_weights={"hotel": 0.35, "activities": 0.30, "flight": 0.20, "budget": 0.15}),
    "solo": Persona(
        name="solo", cue="for a solo trip", budget_cap=720, hotel_min_rating=0.0, hotel_amenity=None,
        cat_weights={"nightlife": 1.0, "outdoor": 1.0, "landmark": 1.0, "museum": 0.8, "food": 0.4},
        pace=3, nonstop=False, cabin="economy",
        dim_weights={"hotel": 0.10, "activities": 0.40, "flight": 0.15, "budget": 0.35}),
    "foodie": Persona(
        name="foodie", cue="for a foodie weekend", budget_cap=1200, hotel_min_rating=8.0, hotel_amenity=None,
        cat_weights={"food": 1.0, "museum": 0.4, "landmark": 0.2, "nightlife": 0.2, "outdoor": -0.2},
        pace=3, nonstop=True, cabin="economy",
        dim_weights={"hotel": 0.20, "activities": 0.45, "flight": 0.10, "budget": 0.25}),
    "business": Persona(
        name="business", cue="for a business trip", budget_cap=1700, hotel_min_rating=9.0, hotel_amenity=None,
        cat_weights={"food": 0.5, "museum": 0.2, "landmark": 0.1, "outdoor": 0.0, "nightlife": -0.5},
        pace=0, nonstop=True, cabin="business",
        dim_weights={"hotel": 0.40, "activities": 0.15, "flight": 0.30, "budget": 0.15}),
}


def _clamp(x: float) -> float:
    return max(0.0, min(1.0, x))


def _hotel_dim(trip: TripState, p: Persona, city: str, nights: int) -> float:
    h = trip.hotel
    if h is None or h.city != city or trip.hotel_nights != nights:
        return 0.0
    if p.hotel_min_rating <= 0:
        base = 1.0
    elif h.rating >= p.hotel_min_rating:
        base = 1.0
    else:
        base = _clamp(1.0 - 0.4 * (p.hotel_min_rating - h.rating))  # 0.4 per star short
    if p.hotel_amenity and p.hotel_amenity not in h.amenities:
        base *= 0.5
    return _clamp(base)


def _activities_dim(trip: TripState, p: Persona, city: str) -> float:
    matched = [a for a in trip.activities if a.city == city]
    count = len(matched)
    if p.pace == 0:
        return 1.0 if count == 0 else _clamp(1.0 - 0.4 * count)
    if count == 0:
        return 0.0
    quality = _clamp(sum(p.cat_weights.get(a.category, 0.0) for a in matched) / count)
    # The persona's favourite category must appear, or the slate feels off.
    top_cat = max(p.cat_weights, key=lambda c: p.cat_weights[c])
    if not any(a.category == top_cat for a in matched):
        quality *= 0.5
    if count < p.pace:
        quality *= count / p.pace
    elif count > p.pace + 1:
        quality *= 0.6
    return _clamp(quality)


def _flight_dim(trip: TripState, p: Persona, city: str) -> float:
    f = trip.flight
    if f is None or f.destination != city:
        return 0.0
    nonstop = 1.0 if (not p.nonstop or f.stops == 0) else 0.35
    cabin = 1.0 if trip.flight_cabin == p.cabin else 0.3
    return (nonstop + cabin) / 2


def _budget_dim(trip: TripState, p: Persona) -> float:
    total = trip.total_cost()
    if total <= p.budget_cap:
        return 1.0
    return _clamp(1.0 - (total - p.budget_cap) / (0.3 * p.budget_cap))  # 0 at 1.3x cap


_FEEDBACK = {
    "hotel": "the hotel wasn't really my style",
    "activities": "the activities didn't suit me",
    "flight": "the flight wasn't ideal for me",
    "budget": "the trip cost more than I'd like",
}


def score_persona(trip: TripState, persona: Persona, required: dict) -> tuple[float, str]:
    """Coarse rubric: weighted per-dimension match in [0, 1], plus one vague line
    naming only the worst-matched dimension."""
    city = required["city"]
    nights = int(required["nights"])
    dims = {
        "hotel": _hotel_dim(trip, persona, city, nights),
        "activities": _activities_dim(trip, persona, city),
        "flight": _flight_dim(trip, persona, city),
        "budget": _budget_dim(trip, persona),
    }
    # Conjunctive: a weighted geometric mean, so a trip that is wrong on any one
    # dimension cannot be rescued by the others. A great hotel does not save a
    # trip whose activities are all wrong for the traveler.
    rating = math.exp(sum(persona.dim_weights[k] * math.log(max(v, 0.02)) for k, v in dims.items()))
    worst = min(dims, key=lambda k: dims[k])
    return _clamp(rating), _FEEDBACK[worst]


def _noise(seed_text: str, magnitude: float) -> float:
    """Deterministic pseudo-noise in [-magnitude, magnitude], keyed by content so
    runs are reproducible (no RNG, which would break resume)."""
    if magnitude <= 0:
        return 0.0
    digest = hashlib.sha256(seed_text.encode("utf-8")).hexdigest()
    frac = int(digest[:8], 16) / 0xFFFFFFFF  # [0, 1]
    return (frac * 2 - 1) * magnitude


# Neutral tool descriptions: they expose the levers (cabin, rate_code, airport
# code) factually so the policy, not the description, decides how to use them.
PERSONA_DESCRIPTIONS: dict[str, str] = {
    "prompt.tool.search_flights.description": (
        "Search flights. Params: origin, destination, date (YYYY-MM-DD), nonstop (bool), max_price "
        "(int USD, 0 = no cap), cabin (economy|business|first). Results list depart time, airline, stops, price."
    ),
    "prompt.tool.search_hotels.description": (
        "Search hotels. Params: city, max_price_per_night (int USD, 0 = no cap), min_rating (float, 1-10 "
        "scale), rate_code ('Q' prepaid or 'B' refundable). Results list rating, amenities, rate-adjusted price."
    ),
    "prompt.tool.search_activities.description": (
        "Search activities. Params: city as the destination airport code (e.g. JFK), category (one of food, "
        "museum, outdoor, nightlife, landmark). Call add_activity(activity_id) to add each one."
    ),
}


def persona_descriptions() -> dict[str, Artifact]:
    return {
        desc_id: genesis(id=desc_id, kind=Subtype.TOOL_DESCRIPTION, content=content)
        for desc_id, content in PERSONA_DESCRIPTIONS.items()
    }


# The genesis policy is a generic planner that does not tailor to the traveler.
POLICY_GENESIS_PERSONA = (
    "You are a travel assistant. Plan and book a trip (a flight, a hotel, and some activities) for "
    "the user's request, and confirm the itinerary."
)

# The oracle encodes the cue-to-preference mapping for all personas; the Tier 1
# search must discover this multi-branch strategy from coarse ratings.
POLICY_ORACLE_PERSONA = (
    "You are a travel assistant. Read the traveler type from the request and tailor the trip to it. "
    "A family wants a comfortable hotel rated 8+ (a pool is a plus), two daytime activities among "
    "outdoor, landmark, and museum, no nightlife, an economy nonstop flight, and a moderate total cost. "
    "A luxury traveler wants a top-rated hotel (9+), three activities among fine dining, museum, and "
    "nightlife, and a business-class nonstop flight; cost is no object. A solo traveler wants a modest, "
    "cheap hotel, three varied activities (museum, outdoor, nightlife, landmark), the cheapest flight even "
    "with a stop, and a low total cost. A foodie wants a comfortable hotel and three mostly-food activities "
    "with some culture, on an economy nonstop flight. A business traveler wants a convenient comfortable "
    "hotel rated 8+, a nonstop flight, and at most one activity. Book economy unless the traveler is "
    "luxury, prefer nonstop unless the traveler is solo, and keep the total within what the traveler would "
    "find reasonable."
)


def build_persona_policy(content: str) -> Artifact:
    return genesis(id=SYSTEM_PROMPT_ID, kind=Subtype.PROMPT, content=content, created_by="human")


@dataclass
class PersonaTask:
    id: str
    persona: str
    request: str
    required: dict[str, Any]


def load_persona_tasks(path: str | Path, split: str = "train") -> list[PersonaTask]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        PersonaTask(id=t["id"], persona=t["persona"], request=t["request"], required=t["required"])
        for t in data[split]
    ]


def persona_eval_set(tasks: list[PersonaTask]) -> EvalSet:
    questions = [
        EvalQuestion(id=t.id, band=3, question=t.request,
                     reference_answer=json.dumps({"persona": t.persona, "required": t.required}))
        for t in tasks
    ]
    return EvalSet(questions=questions, description="persona-rubric trip tasks")


class PersonaRubricJudge:
    """Pairwise rubric signal: rates candidate and reference trips against the
    active persona, prefers the higher rating, and returns the coarse feedback.
    `noise` adds reproducible jitter to the ratings to expose method robustness."""

    def __init__(self, version: int = 1, noise: float = 0.0) -> None:
        self._version = version
        self.noise = noise

    @property
    def kind(self) -> SignalKind:
        return SignalKind.GROUND_TRUTH

    @property
    def cost_estimate(self) -> Cost:
        return Cost()

    @property
    def signal_id(self) -> str:
        return derive_signal_id("PersonaRubricJudge", {"version": self._version, "noise": self.noise})

    @property
    def signal_version(self) -> int:
        return self._version

    def _rate(self, traj, persona: Persona, required: dict) -> tuple[float, str]:
        if traj is None:
            return 0.0, "no trip was booked"
        base, fb = score_persona(reconstruct_trip(traj), persona, required)
        rated = _clamp(base + _noise(f"{persona.name}:{base}", self.noise))
        return rated, fb

    async def measure(
        self,
        candidate: Artifact,
        trajectory: Trajectory | None = None,
        reference: Artifact | None = None,
        ground_truth: Any | None = None,
    ) -> GapMeasurement:
        gt = ground_truth or {}
        try:
            spec = json.loads(gt.get("reference_answer") or "{}")
        except (TypeError, ValueError):
            spec = {}
        persona = PERSONAS.get(spec.get("persona", ""), PERSONAS["family"])
        required = spec.get("required", {})
        cand, fb = self._rate(gt.get("candidate_trajectory") or trajectory, persona, required)
        ref = self._rate(gt.get("reference_trajectory"), persona, required)[0] if gt.get("reference_trajectory") else 0.0
        if cand > ref:
            pref = Preference.LEFT
        elif cand < ref:
            pref = Preference.RIGHT
        else:
            pref = Preference.TIE
        stars = round(1 + 4 * cand, 1)
        return GapMeasurement(
            score=cand,
            preference=pref,
            confidence=abs(cand - ref),
            feedback=f"{stars:.1f}/5 stars from the {persona.name} traveler: {fb}",
            signal_id=self.signal_id,
            signal_version=self.signal_version,
            cost=Cost(),
        )


# ---------------------------------------------------------------------------
# Tier 2: vague requests + an ask_user tool the agent must learn to use.
#
# The request carries no persona cue, so the agent cannot read the traveler from
# it. An ask_user tool, backed by a persona simulator that answers in character,
# lets a clarifying question live inside the tool loop. The hidden persona for the
# current run is injected through a contextvar that the eval harness sets before
# each agent.run, so the tool answers as that traveler without the agent seeing
# the persona. The genesis policy does not ask; the search must discover that it
# should, which is the strategy it already reached for in Tier 1 (where asking
# halted the agent because there was no tool to answer).
# ---------------------------------------------------------------------------

_CURRENT_PERSONA: contextvars.ContextVar = contextvars.ContextVar("current_persona", default=None)

# What each traveler says when asked about a topic. Vague enough to be realistic,
# specific enough that a good agent can map it to bookings.
PERSONA_ANSWERS: dict[str, dict[str, str]] = {
    "family": {
        "budget": "We'd like to keep it reasonable, mid-range, around a thousand dollars total.",
        "who": "Two adults and two young kids.",
        "activities": "Kid-friendly things: outdoor activities and sightseeing, a museum is fine, but no nightlife.",
        "hotel": "Somewhere comfortable, ideally with a pool for the kids.",
        "flight": "Nonstop please, economy is fine.",
        "default": "We just want an easy, fun family trip.",
    },
    "luxury": {
        "budget": "Money is not a concern; I want the best.",
        "who": "Just me.",
        "activities": "Fine dining, museums, and some nightlife.",
        "hotel": "The top-rated, most luxurious hotel available.",
        "flight": "Business class, nonstop.",
        "default": "I want a premium, indulgent experience.",
    },
    "solo": {
        "budget": "I'm on a tight budget, please keep it cheap.",
        "who": "Just me.",
        "activities": "A bit of everything: museums, outdoors, some nightlife, sightseeing.",
        "hotel": "Somewhere cheap and central, the rating doesn't matter much.",
        "flight": "The cheapest option, a stop is fine.",
        "default": "I'm a budget solo traveler who likes to explore.",
    },
    "foodie": {
        "budget": "Mid-range is fine, I'll spend a bit more on good food.",
        "who": "My partner and I.",
        "activities": "Mostly great food, with a little culture like a museum.",
        "hotel": "Comfortable, nothing fancy.",
        "flight": "Nonstop, economy.",
        "default": "We're here for the food, mainly.",
    },
    "business": {
        "budget": "The company is paying; comfort and convenience matter more than cost.",
        "who": "Just me, traveling for work.",
        "activities": "Not much, I'll be busy. Maybe one nice dinner.",
        "hotel": "A convenient, high-quality hotel near downtown.",
        "flight": "Business class, nonstop.",
        "default": "It's a work trip, keep it efficient.",
    },
}

_TOPIC_KEYWORDS = {
    "budget": ("budget", "spend", "afford", "cost", "price", "how much", "expensive", "cheap"),
    "who": ("who", "travel", "travelling", "traveling", "with you", "group", "how many", "kids", "children", "party"),
    "hotel": ("hotel", "stay", "accommodat", "room", "lodging"),
    "flight": ("flight", "fly", "nonstop", "non-stop", "stop", "cabin", "class", "airline"),
    "activities": ("activit", "do", "interest", "like", "enjoy", "see", "experience", "prefer", "nightlife", "food", "museum"),
}


def _simulate_answer(question: str, persona: Persona | None) -> str:
    if persona is None:
        return "I don't have a strong preference; whatever you'd recommend."
    q = question.lower()
    answers = PERSONA_ANSWERS[persona.name]
    for topic in ("budget", "who", "hotel", "flight", "activities"):
        if any(k in q for k in _TOPIC_KEYWORDS[topic]):
            return answers[topic]
    return answers["default"]


def make_ask_user_tool():
    """An ask_user(question) tool that answers as the current hidden persona."""

    async def ask_user(question: str) -> str:
        return _simulate_answer(question, _CURRENT_PERSONA.get())

    return tool(
        description=(
            "Ask the traveler one short clarifying question about their trip preferences "
            "(who is travelling, their budget, hotel style, activities they like or dislike, "
            "or flight preferences) and receive their answer. Use this before booking when the "
            "request does not already tell you what the traveler wants."
        )
    )(ask_user)


def build_persona_agent_t2(system_prompt_text: str, model: str = "claude-sonnet-4-6"):
    """A persona agent with the ask_user tool, honest pricing, and neutral descriptions."""
    return build_travel_agent(
        persona_descriptions(),
        system_prompt=build_persona_policy(system_prompt_text),
        model=model,
        tool_defaults={"default_cabin": "economy", "default_rate_code": "Q"},
        extra_tools=[make_ask_user_tool()],
    )


async def run_for_persona(agent, task: PersonaTask):
    """Run the agent on a Tier 2 task with the hidden persona set for ask_user."""
    token = _CURRENT_PERSONA.set(PERSONAS[task.persona])
    try:
        return await agent.run(task.request)
    finally:
        _CURRENT_PERSONA.reset(token)


# Genesis does not ask; the search must discover elicitation.
POLICY_GENESIS_T2 = (
    "You are a travel assistant. Plan and book a trip (a flight, a hotel, and some activities) "
    "for the user's request using the booking tools, and confirm the itinerary."
)

# The oracle elicits then tailors; this is the strategy the search should find.
POLICY_ORACLE_T2 = (
    "You are a travel assistant, and you do not yet know the traveler's preferences. Before "
    "booking anything, use the ask_user tool to ask a few short questions: who is travelling, "
    "their budget, what hotel style they want, and which activities they enjoy or want to avoid. "
    "Use their answers to tailor the trip: pick a hotel matching their style and budget, book "
    "activities in their stated interests and skip ones they dislike, fly economy unless they "
    "want a premium cabin, and keep the total within their budget. Book the flight, hotel, and "
    "activities with the booking tools, then confirm."
)
