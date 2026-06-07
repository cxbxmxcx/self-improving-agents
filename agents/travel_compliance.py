"""Hidden company travel-policy compliance: the rugged substrate for the
SPO -> GEPA -> DGM escalation example (Chapter 3 §3.4).

The agent plans the same trips, but now a company travel policy of several rules
governs HOW it books. The rules are not in the request or the tool schemas; the
agent can only learn them from the compliance signal's feedback, so the artifact
under improvement (the system prompt) must discover and encode them. Two pairs of
rules conflict, which makes the reward landscape rugged: greedily fixing one
violation regresses another, so a single-line hill-climber (SPO) gets trapped at a
local optimum, a reflective population (GEPA) can compose the joint fix, and the
archive (DGM) can revive a branch the population dropped.

The rules (with weights, so conflicts resolve by priority):
  R1 economy-only (1): flight cabin must be economy.
  R2 no-red-eye (2):   the flight must not depart 21:00-04:59.
  R3 prefer-JetBlue (1): when a JetBlue option exists for the route, book it.
  R4 refundable-long-stay (1.5): 4+ nights must use the refundable rate (B),
     shorter stays the prepaid rate (Q).
  R5 nightly-cap (1.5): the booked nightly rate must be <= $300.
  R6 activity-cap (1):  at most two activities.

Conflicts:
  - R2 vs R3 on LAX->JFK: the only JetBlue flight (FL107) is a red-eye; the
    daytime flight (FL113) is Delta. R2 (weight 2) outranks R3 (weight 1), so the
    compliant choice sacrifices the airline preference.
  - R4 vs R5 on a 4-night stay: the refundable rate is 1.5x, so a four-star hotel
    (HT204 at $369 refundable) busts the $300 cap; only a sub-$200-base hotel
    (HT202, HT205) satisfies both, a coordinated two-part choice.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from helix.artifact import Artifact, Subtype, genesis
from helix.eval.dataset import EvalQuestion, EvalSet
from helix.signal import Cost, GapMeasurement, Preference, SignalKind, derive_signal_id
from helix.trajectory import Trajectory

from agents.travel import SYSTEM_PROMPT_ID
from agents.travel_sim import FLIGHTS, TripState, reconstruct_trip

# A JetBlue option exists on these routes (so R3 applies there).
_JETBLUE_ROUTES = {(f.origin, f.destination) for f in FLIGHTS if f.airline == "JetBlue"}

SATISFIED, VIOLATED, NA = "satisfied", "violated", "na"


def _depart_hour(flight) -> int:
    return int(flight.depart.split(":")[0])


@dataclass(frozen=True)
class Rule:
    id: str
    weight: float
    summary: str
    # status(trip, task) -> SATISFIED | VIOLATED | NA
    status: Callable[[TripState, dict], str]


def _r1(trip: TripState, task: dict) -> str:
    if "flight" not in task.get("required", {}):
        return NA
    if trip.flight is None:
        return VIOLATED
    return SATISFIED if trip.flight_cabin == "economy" else VIOLATED


def _r2(trip: TripState, task: dict) -> str:
    if "flight" not in task.get("required", {}):
        return NA
    if trip.flight is None:
        return VIOLATED
    h = _depart_hour(trip.flight)
    return VIOLATED if (h >= 21 or h < 5) else SATISFIED


def _r3(trip: TripState, task: dict) -> str:
    fr = task.get("required", {}).get("flight")
    if fr is None or (fr["origin"], fr["destination"]) not in _JETBLUE_ROUTES:
        return NA
    if trip.flight is None:
        return VIOLATED
    return SATISFIED if trip.flight.airline == "JetBlue" else VIOLATED


def _r4(trip: TripState, task: dict) -> str:
    hr = task.get("required", {}).get("hotel")
    if hr is None:
        return NA
    if trip.hotel is None:
        return VIOLATED
    want = "B" if hr["nights"] >= 4 else "Q"
    return SATISFIED if trip.hotel_rate_code == want else VIOLATED


def _r5(trip: TripState, task: dict) -> str:
    if "hotel" not in task.get("required", {}):
        return NA
    if trip.hotel is None:
        return VIOLATED
    return SATISFIED if trip.hotel_price_per_night() <= 300 else VIOLATED


def _r6(trip: TripState, task: dict) -> str:
    if "activities" not in task.get("required", {}):
        return NA
    return SATISFIED if len(trip.activities) <= 2 else VIOLATED


COMPLIANCE_RULES: list[Rule] = [
    Rule("R1-economy", 1.0, "flight cabin must be economy", _r1),
    Rule("R2-no-red-eye", 2.0, "flight must not depart between 21:00 and 04:59", _r2),
    Rule("R3-prefer-jetblue", 1.0, "book JetBlue when it serves the route", _r3),
    Rule("R4-refundable-long-stay", 1.5, "4+ nights use refundable rate_code B, shorter stays prepaid Q", _r4),
    Rule("R5-nightly-cap", 1.5, "booked nightly rate must be <= $300", _r5),
    Rule("R6-activity-cap", 1.0, "book at most two activities", _r6),
]


def _completion(trip: TripState, task: dict) -> float:
    req = task.get("required", {})
    got = 0
    if "flight" in req:
        f = req["flight"]
        got += 1 if (trip.flight and trip.flight.origin == f["origin"]
                     and trip.flight.destination == f["destination"] and trip.flight.date == f["date"]) else 0
    if "hotel" in req:
        h = req["hotel"]
        got += 1 if (trip.hotel and trip.hotel.city == h["city"] and trip.hotel_nights == h["nights"]) else 0
    if "activities" in req:
        a = req["activities"]
        n = sum(1 for x in trip.activities if x.city == a.get("city"))
        got += 1 if n >= int(a.get("min_count", 1)) else 0
    return got / max(1, len(req))


def score_compliance(trip: TripState, task: dict) -> tuple[float, list[str]]:
    """score = completion x weighted compliance, in [0, 1].

    Compliance is the satisfied rule weight over the applicable rule weight, so a
    policy that resolves a conflict in favor of the higher-weight rule scores
    higher. The violation strings name each broken rule so the search can learn it.
    """
    completion = _completion(trip, task)
    sat = app = 0.0
    violations: list[str] = []
    for rule in COMPLIANCE_RULES:
        st = rule.status(trip, task)
        if st == NA:
            continue
        app += rule.weight
        if st == SATISFIED:
            sat += rule.weight
        else:
            violations.append(f"{rule.id}: {rule.summary}")
    compliance = (sat / app) if app else 1.0
    if completion < 1.0:
        violations.append(f"incomplete trip ({completion:.0%} of required components booked)")
    return completion * compliance, violations


@dataclass
class ComplianceTask:
    id: str
    request: str
    required: dict[str, Any]


def load_compliance_tasks(path: str | Path) -> list[ComplianceTask]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [ComplianceTask(id=t["id"], request=t["request"], required=t["required"]) for t in data["tasks"]]


def compliance_eval_set(tasks: list[ComplianceTask]) -> EvalSet:
    questions = [
        EvalQuestion(id=t.id, band=3, question=t.request, reference_answer=json.dumps({"required": t.required}))
        for t in tasks
    ]
    return EvalSet(questions=questions, description="company-policy compliance tasks")


# The genesis policy knows none of the company rules; the search must discover
# them from the compliance feedback. The neutral tool descriptions expose the
# levers (cabin, rate_code, depart time, airline) without saying how to use them.
POLICY_GENESIS_COMPLIANCE = (
    "You are a corporate travel assistant. Book the flight, hotel, and activities the user "
    "requests using the available tools, and confirm the itinerary."
)

# The fully-compliant policy, used only to validate the headroom (oracle A/B).
POLICY_ORACLE_COMPLIANCE = (
    "You are a corporate travel assistant. Book the requested flight, hotel, and activities, and "
    "follow the company travel policy when choosing options. Flights must be economy class and must "
    "not depart between 21:00 and 04:59 (no red-eyes); prefer JetBlue when it serves the route, but "
    "never take a red-eye to do so. Hotels of four nights or more must use the refundable rate "
    "(rate_code 'B'); shorter stays use the prepaid rate (rate_code 'Q'); and the booked nightly "
    "rate must not exceed $300, so for a long stay choose a hotel whose base rate is low enough that "
    "the refundable rate still fits. Book at most two activities. When two rules conflict, the "
    "no-red-eye rule outranks the airline preference."
)

COMPLIANCE_DESCRIPTIONS: dict[str, str] = {
    "prompt.tool.search_flights.description": (
        "Search flights. Params: origin, destination, date (YYYY-MM-DD), nonstop (bool), max_price "
        "(int USD, 0 = no cap), cabin (economy|business|first). Each result lists its depart time, "
        "airline, number of stops, and price."
    ),
    "prompt.tool.search_hotels.description": (
        "Search hotels. Params: city, max_price_per_night (int USD, 0 = no cap), min_rating (float), "
        "rate_code ('Q' prepaid or 'B' refundable). Results list the rate-adjusted nightly price."
    ),
    "prompt.tool.search_activities.description": (
        "Search activities. Params: city as the destination airport code (e.g. JFK), category (one of "
        "food, museum, outdoor, nightlife, landmark). Call add_activity(activity_id) to add each one."
    ),
}


def compliance_descriptions() -> dict[str, Artifact]:
    return {
        desc_id: genesis(id=desc_id, kind=Subtype.TOOL_DESCRIPTION, content=content)
        for desc_id, content in COMPLIANCE_DESCRIPTIONS.items()
    }


def build_compliance_policy(content: str) -> Artifact:
    return genesis(id=SYSTEM_PROMPT_ID, kind=Subtype.PROMPT, content=content, created_by="human")


class ComplianceJudge:
    """Pairwise ground-truth signal over the policy-compliance objective."""

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
        return derive_signal_id("ComplianceJudge", {"version": self._version})

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
        cand, cand_v = (
            score_compliance(reconstruct_trip(cand_traj), task)
            if cand_traj is not None else (0.0, ["no candidate run"])
        )
        ref = score_compliance(reconstruct_trip(ref_traj), task)[0] if ref_traj is not None else 0.0
        if cand > ref:
            pref = Preference.LEFT
        elif cand < ref:
            pref = Preference.RIGHT
        else:
            pref = Preference.TIE
        why = ("; ".join(cand_v))[:240] or "fully compliant"
        return GapMeasurement(
            score=cand,
            preference=pref,
            confidence=abs(cand - ref),
            feedback=f"compliance {cand:.2f} (vs ref {ref:.2f}). Policy violations: {why}",
            signal_id=self.signal_id,
            signal_version=self.signal_version,
            cost=Cost(),
        )
