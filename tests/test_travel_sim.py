"""The travel simulation: deterministic tools over a fixed dataset (Ch 3 task agent)."""

from __future__ import annotations

import pytest

from agents.travel_sim import TripState, make_tool_callables


@pytest.fixture
def tools_and_trip():
    trip = TripState()
    return make_tool_callables(trip), trip


@pytest.mark.asyncio
async def test_search_flights_filters_route_date_and_sorts_by_price(tools_and_trip):
    tools, _ = tools_and_trip
    res = await tools["search_flights"]("SFO", "JFK", "2026-06-15")
    ids = [f["id"] for f in res]
    assert ids == ["FL102", "FL101", "FL103"]  # cheapest first
    assert all(f["origin"] == "SFO" and f["destination"] == "JFK" for f in res)


@pytest.mark.asyncio
async def test_search_flights_honors_nonstop_and_max_price(tools_and_trip):
    tools, _ = tools_and_trip
    res = await tools["search_flights"]("SFO", "JFK", "2026-06-15", nonstop=True, max_price=450)
    ids = [f["id"] for f in res]
    assert ids == ["FL101"]  # FL102 is connecting, FL103 is over $450


@pytest.mark.asyncio
async def test_book_flight_records_to_trip(tools_and_trip):
    tools, trip = tools_and_trip
    out = await tools["book_flight"]("FL104")
    assert out["ok"] and trip.flight is not None
    assert trip.flight.id == "FL104"
    assert (await tools["book_flight"]("nope"))["ok"] is False


@pytest.mark.asyncio
async def test_search_hotels_filters_and_sorts_by_rating(tools_and_trip):
    tools, _ = tools_and_trip
    res = await tools["search_hotels"]("JFK", max_price_per_night=300, min_rating=4.0)
    ids = [h["id"] for h in res]
    # 4.0+ rating, <= $300/night, highest rating first.
    assert ids == ["HT203", "HT204"]


@pytest.mark.asyncio
async def test_book_hotel_records_nights_and_cost(tools_and_trip):
    tools, trip = tools_and_trip
    await tools["book_hotel"]("HT202", 3)
    assert trip.hotel.id == "HT202" and trip.hotel_nights == 3
    assert trip.total_cost() == 189 * 3


@pytest.mark.asyncio
async def test_search_activities_filters_by_category(tools_and_trip):
    tools, _ = tools_and_trip
    food = await tools["search_activities"]("JFK", category="food")
    assert {a["id"] for a in food} == {"AC301", "AC302"}
    assert all(a["category"] == "food" for a in food)


@pytest.mark.asyncio
async def test_add_activity_dedups_and_view_itinerary(tools_and_trip):
    tools, trip = tools_and_trip
    await tools["add_activity"]("AC301")
    await tools["add_activity"]("AC301")  # duplicate ignored
    await tools["add_activity"]("AC303")
    assert [a.id for a in trip.activities] == ["AC301", "AC303"]
    summary = await tools["view_itinerary"]()
    assert summary["total_cost"] == 35 + 30
    assert len(summary["activities"]) == 2
