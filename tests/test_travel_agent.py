"""Travel task agent: deterministic scenario scoring and agent construction."""

from __future__ import annotations

from pathlib import Path

import pytest

from agents.travel import (
    TravelScenario,
    build_travel_agent,
    genesis_descriptions,
    load_travel_scenarios,
)
from agents.travel_sim import TripState, make_tool_callables
from helix.tools import TextDescriptionTool

SCENARIOS = Path(__file__).resolve().parent.parent / "chapters" / "ch03" / "travel_scenarios.json"

_TS1 = TravelScenario(
    "t1", "r",
    {"flight": {"origin": "SFO", "destination": "JFK", "date": "2026-06-15", "nonstop": True, "max_price": 450}},
)


async def _trip_with(*calls) -> TripState:
    trip = TripState()
    tools = make_tool_callables(trip)
    for name, *args in calls:
        await tools[name](*args)
    return trip


@pytest.mark.asyncio
async def test_score_is_one_when_all_constraints_met():
    trip = await _trip_with(("book_flight", "FL101"))  # nonstop SFO->JFK, $412
    assert _TS1.score(trip) == 1.0


@pytest.mark.asyncio
async def test_score_is_partial_when_a_constraint_is_missed():
    trip = await _trip_with(("book_flight", "FL102"))  # connecting, $318
    # origin/destination/date/max_price pass, nonstop fails: 4/5.
    assert abs(_TS1.score(trip) - 0.8) < 1e-9


@pytest.mark.asyncio
async def test_no_booking_scores_zero():
    assert _TS1.score(TripState()) == 0.0


@pytest.mark.asyncio
async def test_activity_count_gives_partial_credit():
    sc = TravelScenario("a", "r", {"activities": {"city": "JFK", "category": "food", "count": 2}})
    trip = await _trip_with(("add_activity", "AC301"))  # one food activity of two
    assert sc.score(trip) == 0.5
    trip = await _trip_with(("add_activity", "AC301"), ("add_activity", "AC302"))
    assert sc.score(trip) == 1.0


@pytest.mark.asyncio
async def test_multi_group_scenario_averages_groups():
    sc = load_travel_scenarios(SCENARIOS)[1]  # TS2: flight + hotel
    trip = await _trip_with(("book_flight", "FL101"))  # flight group fully satisfied, no hotel
    # flight group 1.0, hotel group 0.0 -> 0.5
    assert sc.score(trip) == 0.5


def test_build_travel_agent_wires_searchable_tools_as_artifacts():
    agent = build_travel_agent(genesis_descriptions(), TripState())
    by_name = {t.name: t for t in agent.tools.values()}
    assert {"search_flights", "search_hotels", "search_activities"} <= set(by_name)
    assert {"book_flight", "book_hotel", "add_activity", "view_itinerary"} <= set(by_name)
    for name in ("search_flights", "search_hotels", "search_activities"):
        assert isinstance(by_name[name], TextDescriptionTool)


def test_scenarios_load_and_are_well_formed():
    scenarios = load_travel_scenarios(SCENARIOS)
    assert len(scenarios) == 5
    assert all(s.request and s.constraints for s in scenarios)
