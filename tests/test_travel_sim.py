"""The travel simulation: stateless tools + trip reconstruction (Ch 3 task agent)."""

from __future__ import annotations

import pytest

from agents.travel_sim import make_tool_callables, reconstruct_trip
from helix.trajectory import StepKind, Trajectory


@pytest.fixture
def tools():
    return make_tool_callables()


def _traj(*calls) -> Trajectory:
    """Build a trajectory from (tool_name, args, result) triples."""
    t = Trajectory(task="plan a trip")
    for name, args, result in calls:
        t.append(StepKind.TOOL_CALL, {"name": name, "arguments": args})
        t.append(StepKind.TOOL_RESULT, {"result": result})
    return t


@pytest.mark.asyncio
async def test_search_flights_filters_route_date_and_sorts_by_price(tools):
    res = await tools["search_flights"]("SFO", "JFK", "2026-06-15")
    assert [f["id"] for f in res] == ["FL102", "FL101", "FL103"]  # cheapest first


@pytest.mark.asyncio
async def test_search_flights_honors_nonstop_and_max_price(tools):
    res = await tools["search_flights"]("SFO", "JFK", "2026-06-15", nonstop=True, max_price=450)
    assert [f["id"] for f in res] == ["FL101"]  # FL102 connecting, FL103 over budget


@pytest.mark.asyncio
async def test_search_hotels_filters_and_sorts_by_rating(tools):
    res = await tools["search_hotels"]("JFK", max_price_per_night=300, min_rating=4.0)
    assert [h["id"] for h in res] == ["HT203", "HT204"]


@pytest.mark.asyncio
async def test_search_activities_filters_by_category(tools):
    food = await tools["search_activities"]("JFK", category="food")
    assert {a["id"] for a in food} == {"AC301", "AC302"}


@pytest.mark.asyncio
async def test_booking_tools_confirm_without_mutating_shared_state(tools):
    ok = await tools["book_flight"]("FL104")
    assert ok["ok"] and ok["booked"]["id"] == "FL104"
    assert (await tools["book_flight"]("nope"))["ok"] is False
    # Two callable sets are independent (no shared TripState).
    other = make_tool_callables()
    assert (await other["book_flight"]("FL101"))["booked"]["id"] == "FL101"


@pytest.mark.asyncio
async def test_reconstruct_trip_from_successful_bookings(tools):
    bf = await tools["book_flight"]("FL101")
    bh = await tools["book_hotel"]("HT203", 3)
    a1 = await tools["add_activity"]("AC301")
    traj = _traj(
        ("book_flight", {"flight_id": "FL101"}, bf),
        ("book_hotel", {"hotel_id": "HT203", "nights": 3}, bh),
        ("add_activity", {"activity_id": "AC301"}, a1),
    )
    trip = reconstruct_trip(traj)
    assert trip.flight.id == "FL101"
    assert trip.hotel.id == "HT203" and trip.hotel_nights == 3
    assert [a.id for a in trip.activities] == ["AC301"]
    assert trip.total_cost() == 412 + 295 * 3 + 35


@pytest.mark.asyncio
async def test_reconstruct_trip_ignores_failed_bookings_and_dedups(tools):
    traj = _traj(
        ("book_flight", {"flight_id": "nope"}, {"ok": False, "error": "no flight nope"}),
        ("add_activity", {"activity_id": "AC301"}, {"ok": True, "added": {"id": "AC301"}}),
        ("add_activity", {"activity_id": "AC301"}, {"ok": True, "added": {"id": "AC301"}}),
    )
    trip = reconstruct_trip(traj)
    assert trip.flight is None
    assert [a.id for a in trip.activities] == ["AC301"]  # duplicate ignored
