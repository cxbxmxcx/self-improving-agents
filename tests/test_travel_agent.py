"""Travel task agent: deterministic scoring, the pairwise judge, agent build."""

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

_TS1 = TravelScenario(
    "t1", "r",
    {"flight": {"origin": "SFO", "destination": "JFK", "date": "2026-06-15", "nonstop": True, "max_price": 450}},
)


async def _traj(*bookings) -> Trajectory:
    """A trajectory of (tool_name, args_dict) bookings, with real tool results."""
    tools = make_tool_callables()
    t = Trajectory(task="plan a trip")
    for name, args in bookings:
        res = await tools[name](**args)
        t.append(StepKind.TOOL_CALL, {"name": name, "arguments": args})
        t.append(StepKind.TOOL_RESULT, {"result": res})
    return t


async def _trip(*bookings) -> TripState:
    return reconstruct_trip(await _traj(*bookings))


@pytest.mark.asyncio
async def test_score_is_one_when_all_constraints_met():
    trip = await _trip(("book_flight", {"flight_id": "FL101"}))  # nonstop SFO->JFK, $412
    assert _TS1.score(trip) == 1.0


@pytest.mark.asyncio
async def test_score_is_partial_when_a_constraint_is_missed():
    trip = await _trip(("book_flight", {"flight_id": "FL102"}))  # connecting
    assert abs(_TS1.score(trip) - 0.8) < 1e-9  # 4 of 5 fields


@pytest.mark.asyncio
async def test_no_booking_scores_zero():
    assert _TS1.score(TripState()) == 0.0


@pytest.mark.asyncio
async def test_activity_count_gives_partial_credit():
    sc = TravelScenario("a", "r", {"activities": {"city": "JFK", "category": "food", "count": 2}})
    assert (await _trip(("add_activity", {"activity_id": "AC301"})) and
            sc.score(await _trip(("add_activity", {"activity_id": "AC301"}))) == 0.5)
    full = await _trip(
        ("add_activity", {"activity_id": "AC301"}),
        ("add_activity", {"activity_id": "AC302"}),
    )
    assert sc.score(full) == 1.0


@pytest.mark.asyncio
async def test_multi_group_scenario_averages_groups():
    sc = load_travel_scenarios(SCENARIOS)[1]  # TS2: flight + hotel
    trip = await _trip(("book_flight", {"flight_id": "FL101"}))  # flight only
    assert sc.score(trip) == 0.5  # flight 1.0, hotel 0.0


@pytest.mark.asyncio
async def test_travel_task_judge_prefers_the_better_trip():
    constraints = _TS1.constraints
    cand = await _traj(("book_flight", {"flight_id": "FL101"}))   # nonstop, satisfies
    ref = await _traj(("book_flight", {"flight_id": "FL102"}))    # connecting, weaker
    art = genesis("prompt.tool.search_flights.description", Subtype.TOOL_DESCRIPTION, "x")

    m = await TravelTaskJudge().measure(
        candidate=art,
        ground_truth={
            "reference_answer": json.dumps(constraints),
            "candidate_trajectory": cand,
            "reference_trajectory": ref,
        },
    )
    assert m.preference == Preference.LEFT
    assert m.score == 1.0
    assert TravelTaskJudge().kind == SignalKind.GROUND_TRUTH


def test_build_travel_agent_wires_searchable_tools_as_artifacts():
    agent = build_travel_agent(genesis_descriptions())
    by_name = {t.name: t for t in agent.tools.values()}
    assert {"search_flights", "search_hotels", "search_activities"} <= set(by_name)
    assert {"book_flight", "book_hotel", "add_activity"} <= set(by_name)
    for name in ("search_flights", "search_hotels", "search_activities"):
        assert isinstance(by_name[name], TextDescriptionTool)


def test_load_travel_eval_set_carries_constraints_in_reference_answer():
    es = load_travel_eval_set(SCENARIOS)
    assert len(es) == 5
    q = es.questions[0]
    assert q.question and json.loads(q.reference_answer)  # constraints round-trip
