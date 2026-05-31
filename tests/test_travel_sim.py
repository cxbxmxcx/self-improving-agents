"""The travel simulation: gotchas, stateless tools, trip reconstruction."""

from __future__ import annotations

import pytest

from agents.travel_sim import cabin_price, make_tool_callables, reconstruct_trip
from helix.trajectory import StepKind, Trajectory


@pytest.fixture
def tools():
    return make_tool_callables()


def _traj(*calls) -> Trajectory:
    t = Trajectory(task="plan a trip")
    for name, args, result in calls:
        t.append(StepKind.TOOL_CALL, {"name": name, "arguments": args})
        t.append(StepKind.TOOL_RESULT, {"result": result})
    return t


@pytest.mark.asyncio
async def test_search_flights_default_cabin_is_first_class_and_pricier(tools):
    # Gotcha: default cabin is first class (3x). Same query, two cabins.
    eco = await tools["search_flights"]("SFO", "JFK", "2026-06-15", cabin="economy")
    first = await tools["search_flights"]("SFO", "JFK", "2026-06-15")  # default first
    assert eco[0]["price"] * 3 == first[0]["price"]
    assert first[0]["cabin"] == "first" and eco[0]["cabin"] == "economy"


@pytest.mark.asyncio
async def test_max_price_hides_affordable_fares_under_default_cabin(tools):
    # Under default (first) cabin, FL101 is $1236, so a $450 cap returns nothing;
    # in economy the same cap returns it.
    none_first = await tools["search_flights"]("SFO", "JFK", "2026-06-15", nonstop=True, max_price=450)
    some_eco = await tools["search_flights"]("SFO", "JFK", "2026-06-15", nonstop=True, max_price=450, cabin="economy")
    assert none_first == []
    assert [f["id"] for f in some_eco] == ["FL101"]


@pytest.mark.asyncio
async def test_search_hotels_rating_is_on_a_ten_point_scale(tools):
    # Gotcha: a "4-star" request as min_rating=4 returns everything (all >= 6.4);
    # the real floor for 4 stars is min_rating=8.
    loose = await tools["search_hotels"]("JFK", max_price_per_night=300, min_rating=4.0)
    strict = await tools["search_hotels"]("JFK", max_price_per_night=300, min_rating=8.0)
    assert {h["id"] for h in strict} == {"HT203", "HT204"}  # 8.8, 8.2
    assert len(loose) > len(strict)


@pytest.mark.asyncio
async def test_search_activities_exact_category_required(tools):
    assert {a["id"] for a in await tools["search_activities"]("JFK", category="food")} == {"AC301", "AC302"}
    assert await tools["search_activities"]("JFK", category="dining") == []  # synonym matches nothing


@pytest.mark.asyncio
async def test_reconstruct_trip_captures_cabin_and_price():
    bf = {"ok": True, "booked": {"id": "FL101"}, "cabin": "economy", "price": cabin_price(412, "economy")}
    trip = reconstruct_trip(_traj(("book_flight", {"flight_id": "FL101", "cabin": "economy"}, bf)))
    assert trip.flight.id == "FL101" and trip.flight_cabin == "economy"
    assert trip.flight_price() == 412  # economy = base

    bf_first = {"ok": True, "booked": {"id": "FL101"}, "cabin": "first"}
    trip2 = reconstruct_trip(_traj(("book_flight", {"flight_id": "FL101"}, bf_first)))
    assert trip2.flight_cabin == "first" and trip2.flight_price() == 412 * 3
