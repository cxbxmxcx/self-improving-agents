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
async def test_search_hotels_default_rate_code_is_pricier(tools):
    # Gotcha: the nightly rate is multiplied by the rate code, default B (1.5x).
    # Under a $260 cap, no four-star hotel fits on the default code, but the Q
    # (1x) code brings HT204 ($246) within budget.
    none_default = await tools["search_hotels"]("JFK", max_price_per_night=260, min_rating=8.0)
    some_q = await tools["search_hotels"]("JFK", max_price_per_night=260, min_rating=8.0, rate_code="Q")
    assert none_default == []
    assert "HT204" in {h["id"] for h in some_q}
    assert some_q[0]["rate_code"] == "Q"


@pytest.mark.asyncio
async def test_search_activities_city_must_be_the_airport_code(tools):
    # Gotcha: activities are keyed by airport code, so the city name returns [].
    assert await tools["search_activities"]("New York", category="food") == []
    assert {a["id"] for a in await tools["search_activities"]("JFK", category="food")} == {"AC301", "AC302"}
    assert await tools["search_activities"]("JFK", category="dining") == []  # synonym matches nothing


@pytest.mark.asyncio
async def test_reconstruct_trip_captures_hotel_rate_code(tools):
    bh = await tools["book_hotel"]("HT204", nights=2, rate_code="Q")
    trip = reconstruct_trip(_traj(("book_hotel", {"hotel_id": "HT204", "nights": 2, "rate_code": "Q"}, bh)))
    assert trip.hotel.id == "HT204" and trip.hotel_rate_code == "Q"
    assert trip.hotel_price_per_night() == 246  # Q = base
    bh_default = await tools["book_hotel"]("HT204", nights=2)  # default B
    trip2 = reconstruct_trip(_traj(("book_hotel", {"hotel_id": "HT204", "nights": 2}, bh_default)))
    assert trip2.hotel_rate_code == "B" and trip2.hotel_price_per_night() == round(246 * 1.5)


@pytest.mark.asyncio
async def test_reconstruct_trip_captures_cabin_and_price():
    bf = {"ok": True, "booked": {"id": "FL101"}, "cabin": "economy", "price": cabin_price(412, "economy")}
    trip = reconstruct_trip(_traj(("book_flight", {"flight_id": "FL101", "cabin": "economy"}, bf)))
    assert trip.flight.id == "FL101" and trip.flight_cabin == "economy"
    assert trip.flight_price() == 412  # economy = base

    bf_first = {"ok": True, "booked": {"id": "FL101"}, "cabin": "first"}
    trip2 = reconstruct_trip(_traj(("book_flight", {"flight_id": "FL101"}, bf_first)))
    assert trip2.flight_cabin == "first" and trip2.flight_price() == 412 * 3
