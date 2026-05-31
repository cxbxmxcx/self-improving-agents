"""Travel task agent: gotcha headroom, scoring, the pairwise judge, agent build."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agents.travel import (
    TravelScenario,
    TravelTaskJudge,
    build_travel_agent,
    genesis_descriptions,
    load_travel_eval_set,
    load_travel_scenarios,
)
from agents.travel_sim import TripState, make_tool_callables, reconstruct_trip
from helix.artifact import Subtype, genesis
from helix.signal import Preference, SignalKind
from helix.tools import TextDescriptionTool
from helix.trajectory import StepKind, Trajectory

SCENARIOS = Path(__file__).resolve().parent.parent / "chapters" / "ch03" / "travel_scenarios.json"

_FLIGHT = {"origin": "SFO", "destination": "JFK", "date": "2026-06-15", "nonstop": True, "max_price": 500}
_TS_FLIGHT = TravelScenario("tf", "r", {"flight": _FLIGHT})


async def _traj(*bookings) -> Trajectory:
    tools = make_tool_callables()
    t = Trajectory(task="plan a trip")
    for name, args in bookings:
        res = await tools[name](**args)
        t.append(StepKind.TOOL_CALL, {"name": name, "arguments": args})
        t.append(StepKind.TOOL_RESULT, {"result": res})
    return t


async def _trip(*bookings) -> TripState:
    return reconstruct_trip(await _traj(*bookings))


# ---------------- the flight cabin gotcha ----------------

@pytest.mark.asyncio
async def test_economy_booking_meets_budget_but_first_class_does_not():
    eco = await _trip(("book_flight", {"flight_id": "FL101", "cabin": "economy"}))
    assert _TS_FLIGHT.score(eco) == 1.0                       # $412 <= $500

    first = await _trip(("book_flight", {"flight_id": "FL101"}))  # default first, $1236
    score, reasons = _TS_FLIGHT.score_with_reasons(first)
    assert score == 0.8                                       # 4 of 5 fields; budget fails
    assert any("exceeds budget" in r for r in reasons)


# ---------------- the hotel rating-scale gotcha ----------------

@pytest.mark.asyncio
async def test_hotel_rating_scale_gotcha():
    sc = TravelScenario("th", "r", {"hotel": {"city": "JFK", "min_rating": 8.0}})
    naive = await _trip(("book_hotel", {"hotel_id": "HT202", "nights": 2}))   # 7.8/10
    score, reasons = sc.score_with_reasons(naive)
    assert score == 0.5 and any("below required 8" in r for r in reasons)     # city ok, rating not
    good = await _trip(("book_hotel", {"hotel_id": "HT204", "nights": 2}))    # 8.2/10
    assert sc.score(good) == 1.0


# ---------------- the activity category gotcha ----------------

@pytest.mark.asyncio
async def test_activity_category_gotcha():
    sc = TravelScenario("ta", "r", {"activities": {"city": "JFK", "category": "food", "count": 2}})
    wrong = await _trip(("add_activity", {"activity_id": "AC303"}))  # a museum
    assert sc.score(wrong) == 0.0
    right = await _trip(
        ("add_activity", {"activity_id": "AC301"}),
        ("add_activity", {"activity_id": "AC302"}),
    )
    assert sc.score(right) == 1.0


@pytest.mark.asyncio
async def test_no_booking_scores_zero():
    assert _TS_FLIGHT.score(TripState()) == 0.0


# ---------------- the pairwise judge ----------------

@pytest.mark.asyncio
async def test_travel_task_judge_prefers_the_economy_booking():
    cand = await _traj(("book_flight", {"flight_id": "FL101", "cabin": "economy"}))  # in budget
    ref = await _traj(("book_flight", {"flight_id": "FL101"}))                       # first, over budget
    art = genesis("prompt.tool.search_flights.description", Subtype.TOOL_DESCRIPTION, "x")
    m = await TravelTaskJudge().measure(
        candidate=art,
        ground_truth={
            "reference_answer": json.dumps(_TS_FLIGHT.constraints),
            "candidate_trajectory": cand,
            "reference_trajectory": ref,
        },
    )
    assert m.preference == Preference.LEFT
    assert m.score == 1.0
    assert TravelTaskJudge().kind == SignalKind.GROUND_TRUTH
    assert "exceeds budget" not in m.feedback or "task success 1.00" in m.feedback


# ---------------- agent construction / loaders ----------------

def test_build_travel_agent_wires_searchable_tools_as_artifacts():
    agent = build_travel_agent(genesis_descriptions())
    by_name = {t.name: t for t in agent.tools.values()}
    assert {"search_flights", "search_hotels", "search_activities"} <= set(by_name)
    for name in ("search_flights", "search_hotels", "search_activities"):
        assert isinstance(by_name[name], TextDescriptionTool)


def test_load_travel_eval_set_carries_constraints_in_reference_answer():
    es = load_travel_eval_set(SCENARIOS)
    assert len(es) == 5
    assert json.loads(es.questions[0].reference_answer)  # constraints round-trip


def test_scenarios_use_the_gotcha_scales():
    scenarios = load_travel_scenarios(SCENARIOS)
    flights = [s.constraints["flight"] for s in scenarios if "flight" in s.constraints]
    hotels = [s.constraints["hotel"] for s in scenarios if "hotel" in s.constraints]
    assert all(f.get("max_price", 500) >= 500 for f in flights if "max_price" in f)
    assert all(h["min_rating"] >= 8.0 for h in hotels if "min_rating" in h)
