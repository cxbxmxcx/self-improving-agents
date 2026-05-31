"""A deterministic travel-booking simulation for the Chapter 3 task agent.

No network, no API keys, no LLM: an in-memory dataset of flights, hotels, and
activities plus a stateful booking session. The agent's tools read this dataset
and write bookings into a TripState; a deterministic check then scores whether
the assembled itinerary satisfied the request. This is what lets Chapter 3 grade
tool-description optimization with a ground-truth signal rather than a judge.

The simulation is deliberately discriminating: prices, stops, ratings, and
categories vary enough that constraints like "cheapest nonstop", "under $300 a
night", "4-star or better", or "two food activities" each select a different
subset. A vague tool description makes the agent miss those constraints, which
is the failure mode the chapter's search methods learn to fix.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Flight:
    id: str
    origin: str
    destination: str
    date: str          # ISO yyyy-mm-dd
    depart: str        # HH:MM local
    arrive: str
    stops: int
    price: int         # USD
    airline: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "origin": self.origin, "destination": self.destination,
            "date": self.date, "depart": self.depart, "arrive": self.arrive,
            "stops": self.stops, "nonstop": self.stops == 0,
            "price": self.price, "airline": self.airline,
        }


@dataclass(frozen=True)
class Hotel:
    id: str
    city: str
    name: str
    price_per_night: int
    rating: float          # stars, 1.0 - 5.0
    neighborhood: str
    amenities: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "city": self.city, "name": self.name,
            "price_per_night": self.price_per_night, "rating": self.rating,
            "neighborhood": self.neighborhood, "amenities": list(self.amenities),
        }


@dataclass(frozen=True)
class Activity:
    id: str
    city: str
    name: str
    category: str          # food | museum | outdoor | nightlife | landmark
    price: int
    duration_hrs: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "city": self.city, "name": self.name,
            "category": self.category, "price": self.price,
            "duration_hrs": self.duration_hrs,
        }


# Flights span three origins, three destinations, two dates, with a mix of
# nonstop/connecting and price so constraints discriminate.
FLIGHTS: list[Flight] = [
    Flight("FL101", "SFO", "JFK", "2026-06-15", "07:00", "15:30", 0, 412, "JetBlue"),
    Flight("FL102", "SFO", "JFK", "2026-06-15", "09:30", "21:10", 1, 318, "United"),
    Flight("FL103", "SFO", "JFK", "2026-06-15", "13:00", "21:25", 0, 489, "Delta"),
    Flight("FL104", "SFO", "JFK", "2026-06-16", "08:00", "16:20", 0, 377, "Alaska"),
    Flight("FL105", "SFO", "LAX", "2026-06-15", "06:30", "08:00", 0, 96, "Southwest"),
    Flight("FL106", "SFO", "LAX", "2026-06-15", "11:00", "12:35", 0, 142, "Delta"),
    Flight("FL107", "LAX", "JFK", "2026-06-15", "22:00", "06:30", 0, 268, "JetBlue"),
    Flight("FL108", "SEA", "JFK", "2026-06-15", "07:45", "16:10", 0, 401, "Alaska"),
    Flight("FL109", "SEA", "JFK", "2026-06-15", "10:15", "21:55", 1, 295, "United"),
    Flight("FL110", "SFO", "ORD", "2026-06-15", "06:00", "12:20", 0, 211, "United"),
    Flight("FL111", "SFO", "ORD", "2026-06-15", "14:30", "22:40", 1, 178, "American"),
    Flight("FL112", "SEA", "LAX", "2026-06-15", "09:00", "11:35", 0, 124, "Alaska"),
]

HOTELS: list[Hotel] = [
    Hotel("HT201", "JFK", "Gramercy Park Hotel", 410, 4.6, "Gramercy", ("gym", "bar", "wifi")),
    Hotel("HT202", "JFK", "Pod Times Square", 189, 3.9, "Midtown", ("wifi", "rooftop")),
    Hotel("HT203", "JFK", "The Standard High Line", 295, 4.4, "Meatpacking", ("gym", "pool", "wifi", "bar")),
    Hotel("HT204", "JFK", "Hotel Indigo LES", 246, 4.1, "Lower East Side", ("wifi", "gym")),
    Hotel("HT205", "JFK", "Brooklyn Budget Inn", 119, 3.2, "Williamsburg", ("wifi",)),
    Hotel("HT206", "LAX", "Shore Hotel Santa Monica", 339, 4.5, "Santa Monica", ("pool", "wifi", "beach")),
    Hotel("HT207", "LAX", "Freehand Los Angeles", 175, 4.0, "Downtown", ("pool", "wifi", "bar")),
    Hotel("HT208", "ORD", "The Robey", 232, 4.3, "Wicker Park", ("gym", "wifi", "rooftop")),
    Hotel("HT209", "ORD", "Chicago Getaway Hostel", 88, 3.4, "Lincoln Park", ("wifi",)),
]

ACTIVITIES: list[Activity] = [
    Activity("AC301", "JFK", "Katz's Delicatessen", "food", 35, 1.5),
    Activity("AC302", "JFK", "Le Bernardin Tasting", "food", 210, 2.5),
    Activity("AC303", "JFK", "The Met", "museum", 30, 3.0),
    Activity("AC304", "JFK", "MoMA", "museum", 28, 2.5),
    Activity("AC305", "JFK", "Central Park Bike Tour", "outdoor", 45, 2.0),
    Activity("AC306", "JFK", "Statue of Liberty Ferry", "landmark", 24, 3.5),
    Activity("AC307", "JFK", "Comedy Cellar", "nightlife", 30, 2.0),
    Activity("AC308", "LAX", "Grand Central Market", "food", 25, 1.5),
    Activity("AC309", "LAX", "Getty Center", "museum", 0, 3.0),
    Activity("AC310", "LAX", "Griffith Observatory Hike", "outdoor", 0, 2.5),
    Activity("AC311", "ORD", "Art Institute of Chicago", "museum", 32, 3.0),
    Activity("AC312", "ORD", "Deep Dish Pizza Tour", "food", 55, 2.0),
]

_FLIGHTS_BY_ID = {f.id: f for f in FLIGHTS}
_HOTELS_BY_ID = {h.id: h for h in HOTELS}
_ACTIVITIES_BY_ID = {a.id: a for a in ACTIVITIES}


# ---------------------------------------------------------------------------
# Booking session (the agent's mutable state for one trip)
# ---------------------------------------------------------------------------

@dataclass
class TripState:
    """What the agent has booked so far for one trip. The ground-truth check
    reads this after a run to score the itinerary against the request."""

    flight: Flight | None = None
    hotel: Hotel | None = None
    hotel_nights: int = 0
    activities: list[Activity] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        return {
            "flight": self.flight.as_dict() if self.flight else None,
            "hotel": self.hotel.as_dict() if self.hotel else None,
            "hotel_nights": self.hotel_nights,
            "activities": [a.as_dict() for a in self.activities],
            "total_cost": self.total_cost(),
        }

    def total_cost(self) -> int:
        total = 0
        if self.flight:
            total += self.flight.price
        if self.hotel:
            total += self.hotel.price_per_night * max(0, self.hotel_nights)
        total += sum(a.price for a in self.activities)
        return total


# ---------------------------------------------------------------------------
# Tool implementations (plain Python, bound to a TripState)
# ---------------------------------------------------------------------------

def make_tool_callables(trip: TripState) -> dict[str, Any]:
    """Return the seven raw async tool callables bound to one TripState.

    The search tools read the dataset; the book/add tools mutate `trip`. Each
    is a normal async function with type hints, so TextDescriptionTool can
    derive its argument schema. The descriptions are supplied separately as
    artifacts (genesis_descriptions), which is what the chapter searches over.
    """

    async def search_flights(
        origin: str,
        destination: str,
        date: str,
        nonstop: bool = False,
        max_price: int = 0,
    ) -> list[dict]:
        out = [
            f for f in FLIGHTS
            if f.origin == origin and f.destination == destination and f.date == date
        ]
        if nonstop:
            out = [f for f in out if f.stops == 0]
        if max_price:
            out = [f for f in out if f.price <= max_price]
        out.sort(key=lambda f: f.price)
        return [f.as_dict() for f in out]

    async def book_flight(flight_id: str) -> dict:
        f = _FLIGHTS_BY_ID.get(flight_id)
        if f is None:
            return {"ok": False, "error": f"no flight {flight_id}"}
        trip.flight = f
        return {"ok": True, "booked": f.as_dict()}

    async def search_hotels(
        city: str,
        max_price_per_night: int = 0,
        min_rating: float = 0.0,
    ) -> list[dict]:
        out = [h for h in HOTELS if h.city == city]
        if max_price_per_night:
            out = [h for h in out if h.price_per_night <= max_price_per_night]
        if min_rating:
            out = [h for h in out if h.rating >= min_rating]
        out.sort(key=lambda h: (-h.rating, h.price_per_night))
        return [h.as_dict() for h in out]

    async def book_hotel(hotel_id: str, nights: int) -> dict:
        h = _HOTELS_BY_ID.get(hotel_id)
        if h is None:
            return {"ok": False, "error": f"no hotel {hotel_id}"}
        trip.hotel = h
        trip.hotel_nights = nights
        return {"ok": True, "booked": h.as_dict(), "nights": nights}

    async def search_activities(city: str, category: str = "") -> list[dict]:
        out = [a for a in ACTIVITIES if a.city == city]
        if category:
            out = [a for a in out if a.category == category]
        out.sort(key=lambda a: a.price)
        return [a.as_dict() for a in out]

    async def add_activity(activity_id: str) -> dict:
        a = _ACTIVITIES_BY_ID.get(activity_id)
        if a is None:
            return {"ok": False, "error": f"no activity {activity_id}"}
        if a.id not in {x.id for x in trip.activities}:
            trip.activities.append(a)
        return {"ok": True, "added": a.as_dict()}

    async def view_itinerary() -> dict:
        return trip.summary()

    return {
        "search_flights": search_flights,
        "book_flight": book_flight,
        "search_hotels": search_hotels,
        "book_hotel": book_hotel,
        "search_activities": search_activities,
        "add_activity": add_activity,
        "view_itinerary": view_itinerary,
    }


# ---------------------------------------------------------------------------
# Genesis tool descriptions (intentionally vague; the chapter improves them)
# ---------------------------------------------------------------------------

# Each id is a TOOL_DESCRIPTION artifact id. The genesis content is terse and
# omits the constraint parameters on purpose, so a vanilla agent misses
# "nonstop", "max_price", "min_rating", and "category" until search improves
# the description. Chapter 3 targets these ids one at a time.
GENESIS_DESCRIPTIONS: dict[str, str] = {
    "prompt.tool.search_flights.description": "Search for flights.",
    "prompt.tool.search_hotels.description": "Search for hotels in a city.",
    "prompt.tool.search_activities.description": "Find things to do in a city.",
}

# Tools whose descriptions are not under search keep a fixed, adequate string.
FIXED_DESCRIPTIONS: dict[str, str] = {
    "book_flight": "Book a flight by its id. Call after search_flights.",
    "book_hotel": "Book a hotel by its id for a number of nights.",
    "add_activity": "Add an activity to the itinerary by its id.",
    "view_itinerary": "Show the flight, hotel, and activities booked so far.",
}

# Maps the searchable tool name to its description-artifact id.
SEARCHABLE_TOOL_DESCRIPTION_IDS: dict[str, str] = {
    "search_flights": "prompt.tool.search_flights.description",
    "search_hotels": "prompt.tool.search_hotels.description",
    "search_activities": "prompt.tool.search_activities.description",
}
