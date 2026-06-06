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

# Gotcha: ratings are on a 1-10 scale, not the 1-5 a user means by "stars".
# A "4-star" request maps to min_rating 8, which only the description reveals.
HOTELS: list[Hotel] = [
    Hotel("HT201", "JFK", "Gramercy Park Hotel", 410, 9.2, "Gramercy", ("gym", "bar", "wifi")),
    Hotel("HT202", "JFK", "Pod Times Square", 189, 7.8, "Midtown", ("wifi", "rooftop")),
    Hotel("HT203", "JFK", "The Standard High Line", 295, 8.8, "Meatpacking", ("gym", "pool", "wifi", "bar")),
    Hotel("HT204", "JFK", "Hotel Indigo LES", 246, 8.2, "Lower East Side", ("wifi", "gym")),
    Hotel("HT205", "JFK", "Brooklyn Budget Inn", 119, 6.4, "Williamsburg", ("wifi",)),
    Hotel("HT206", "LAX", "Shore Hotel Santa Monica", 339, 9.0, "Santa Monica", ("pool", "wifi", "beach")),
    Hotel("HT207", "LAX", "Freehand Los Angeles", 175, 8.0, "Downtown", ("pool", "wifi", "bar")),
    Hotel("HT208", "ORD", "The Robey", 232, 8.6, "Wicker Park", ("gym", "wifi", "rooftop")),
    Hotel("HT209", "ORD", "Chicago Getaway Hostel", 88, 6.8, "Lincoln Park", ("wifi",)),
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

# Gotcha: the flight price shown is multiplied by the cabin. The default cabin
# is first class (3x), so an agent that does not pass cabin="economy" sees only
# fares it cannot afford. Only the description reveals this.
CABIN_MULTIPLIER: dict[str, float] = {"economy": 1.0, "business": 2.0, "first": 3.0}
DEFAULT_CABIN = "first"

# Gotcha: the nightly rate shown is multiplied by the rate code. The codes are
# opaque on purpose ("Q" is the 1x rate, "B" is the 1.5x default) so the
# improver's proposer has no real-world meaning to lean on, unlike the airline
# "advance vs flexible" framing that carries a refundability tradeoff. An agent
# that does not pass rate_code="Q" sees only inflated nightly rates, so a
# four-star hotel never fits a tight budget. Only the description reveals that
# the cheaper Q code exists and that the default B is not it.
RATE_CODE_MULTIPLIER: dict[str, float] = {"Q": 1.0, "B": 1.5}
DEFAULT_RATE_CODE = "B"

# Gotcha: the city parameter is the destination airport code, not the city name.
# search_activities("New York", ...) silently returns [] because activities are
# keyed by code ("JFK"). Only the description reveals the New York=JFK vocabulary.
CITY_CODES: dict[str, str] = {
    "new york": "JFK", "new york city": "JFK", "nyc": "JFK", "manhattan": "JFK",
    "los angeles": "LAX", "la": "LAX",
    "chicago": "ORD",
    "seattle": "SEA",
    "san francisco": "SFO",
}

# Gotcha: category must be one of these exact values. A request for "dining" or
# "restaurants" matches nothing unless mapped to "food".
ACTIVITY_CATEGORIES = ("food", "museum", "outdoor", "nightlife", "landmark")


def cabin_price(base: int, cabin: str) -> int:
    return round(base * CABIN_MULTIPLIER.get(cabin, CABIN_MULTIPLIER[DEFAULT_CABIN]))


def hotel_price(base: int, rate_code: str) -> int:
    return round(base * RATE_CODE_MULTIPLIER.get(rate_code, RATE_CODE_MULTIPLIER[DEFAULT_RATE_CODE]))


# ---------------------------------------------------------------------------
# Booking session (the agent's mutable state for one trip)
# ---------------------------------------------------------------------------

@dataclass
class TripState:
    """What the agent has booked so far for one trip. The ground-truth check
    reads this after a run to score the itinerary against the request."""

    flight: Flight | None = None
    flight_cabin: str = DEFAULT_CABIN
    hotel: Hotel | None = None
    hotel_nights: int = 0
    hotel_rate_code: str = DEFAULT_RATE_CODE
    activities: list[Activity] = field(default_factory=list)

    def flight_price(self) -> int:
        """The fare actually booked, after the cabin multiplier."""
        return cabin_price(self.flight.price, self.flight_cabin) if self.flight else 0

    def hotel_price_per_night(self) -> int:
        """The nightly rate actually booked, after the rate-plan multiplier."""
        return hotel_price(self.hotel.price_per_night, self.hotel_rate_code) if self.hotel else 0

    def summary(self) -> dict[str, Any]:
        return {
            "flight": self.flight.as_dict() if self.flight else None,
            "flight_cabin": self.flight_cabin if self.flight else None,
            "flight_price": self.flight_price(),
            "hotel": self.hotel.as_dict() if self.hotel else None,
            "hotel_nights": self.hotel_nights,
            "hotel_rate_code": self.hotel_rate_code if self.hotel else None,
            "hotel_price_per_night": self.hotel_price_per_night(),
            "activities": [a.as_dict() for a in self.activities],
            "total_cost": self.total_cost(),
        }

    def total_cost(self) -> int:
        total = 0
        if self.flight:
            total += self.flight_price()
        if self.hotel:
            total += self.hotel_price_per_night() * max(0, self.hotel_nights)
        total += sum(a.price for a in self.activities)
        return total


# ---------------------------------------------------------------------------
# Tool implementations (plain Python, stateless)
# ---------------------------------------------------------------------------

def make_tool_callables() -> dict[str, Any]:
    """Return the six raw async tool callables.

    The tools are stateless: search tools read the dataset, and the book/add
    tools just confirm the booking. The itinerary is not held in a shared
    object; it is the trajectory itself, which `reconstruct_trip` reads back.
    This keeps the agent clonable by the framework's `with_artifacts` without
    leaking booking state across eval runs. Each is a typed async function so
    TextDescriptionTool can derive its argument schema.
    """

    async def search_flights(
        origin: str,
        destination: str,
        date: str,
        nonstop: bool = False,
        max_price: int = 0,
        cabin: str = DEFAULT_CABIN,
    ) -> list[dict]:
        out = [
            f for f in FLIGHTS
            if f.origin == origin and f.destination == destination and f.date == date
        ]
        if nonstop:
            out = [f for f in out if f.stops == 0]
        # The fare shown is the cabin-adjusted price (default cabin is first
        # class), and max_price filters on that, so the default cabin can hide
        # every affordable fare.
        priced = [(f, cabin_price(f.price, cabin)) for f in out]
        if max_price:
            priced = [(f, p) for f, p in priced if p <= max_price]
        priced.sort(key=lambda fp: fp[1])
        return [{**f.as_dict(), "price": p, "cabin": cabin} for f, p in priced]

    async def book_flight(flight_id: str, cabin: str = DEFAULT_CABIN) -> dict:
        f = _FLIGHTS_BY_ID.get(flight_id)
        if f is None:
            return {"ok": False, "error": f"no flight {flight_id}"}
        return {"ok": True, "booked": f.as_dict(), "cabin": cabin,
                "price": cabin_price(f.price, cabin)}

    async def search_hotels(
        city: str,
        max_price_per_night: int = 0,
        min_rating: float = 0.0,
        rate_code: str = DEFAULT_RATE_CODE,
    ) -> list[dict]:
        # Hotels accept the city name or its code, so the only hotel gotcha is the
        # rate code; the airport-code vocabulary is left for search_activities.
        code = CITY_CODES.get(city.strip().lower(), city)
        out = [h for h in HOTELS if h.city == code]
        if min_rating:
            out = [h for h in out if h.rating >= min_rating]
        # The nightly rate shown is the rate-code-adjusted price (default code is
        # B, 1.5x), and max_price_per_night filters on that, so the default code
        # can hide every hotel that fits a tight nightly budget.
        priced = [(h, hotel_price(h.price_per_night, rate_code)) for h in out]
        if max_price_per_night:
            priced = [(h, p) for h, p in priced if p <= max_price_per_night]
        priced.sort(key=lambda hp: (-hp[0].rating, hp[1]))
        return [{**h.as_dict(), "price_per_night": p, "rate_code": rate_code} for h, p in priced]

    async def book_hotel(hotel_id: str, nights: int = 1, rate_code: str = DEFAULT_RATE_CODE) -> dict:
        h = _HOTELS_BY_ID.get(hotel_id)
        if h is None:
            return {"ok": False, "error": f"no hotel {hotel_id}"}
        return {"ok": True, "booked": h.as_dict(), "nights": nights, "rate_code": rate_code,
                "price_per_night": hotel_price(h.price_per_night, rate_code)}

    async def search_activities(city: str, category: str = "") -> list[dict]:
        # Gotcha: city must be the destination airport code (e.g. "JFK"), not the
        # city name. A name like "New York" matches nothing and returns [], so the
        # agent has nothing to add unless the description reveals the code.
        out = [a for a in ACTIVITIES if a.city == city]
        if category:
            out = [a for a in out if a.category == category]
        out.sort(key=lambda a: a.price)
        return [a.as_dict() for a in out]

    async def add_activity(activity_id: str) -> dict:
        a = _ACTIVITIES_BY_ID.get(activity_id)
        if a is None:
            return {"ok": False, "error": f"no activity {activity_id}"}
        return {"ok": True, "added": a.as_dict()}

    return {
        "search_flights": search_flights,
        "book_flight": book_flight,
        "search_hotels": search_hotels,
        "book_hotel": book_hotel,
        "search_activities": search_activities,
        "add_activity": add_activity,
    }


# ---------------------------------------------------------------------------
# Reconstruct the booked trip from a trajectory (the itinerary is the run)
# ---------------------------------------------------------------------------

def reconstruct_trip(trajectory: Any) -> TripState:
    """Rebuild the TripState a run booked by reading its trajectory.

    Pairs each TOOL_CALL with the TOOL_RESULT that follows it and applies the
    successful book_flight / book_hotel / add_activity calls. This is how the
    ground-truth signals grade what the agent actually did, independent of any
    in-process state, which is what makes the task agent compose with the
    framework's clone-per-run eval.
    """
    trip = TripState()
    pending_name: str | None = None
    pending_args: dict[str, Any] = {}
    for step in getattr(trajectory, "steps", []):
        kind = getattr(getattr(step, "kind", None), "value", getattr(step, "kind", None))
        payload = getattr(step, "payload", {}) or {}
        if kind == "tool_call":
            pending_name = payload.get("name")
            pending_args = payload.get("arguments", {}) or {}
        elif kind == "tool_result" and pending_name is not None:
            result = payload.get("result", {})
            ok = isinstance(result, dict) and result.get("ok")
            if ok and pending_name == "book_flight":
                f = _FLIGHTS_BY_ID.get(pending_args.get("flight_id"))
                if f is not None:
                    trip.flight = f
                    trip.flight_cabin = pending_args.get("cabin", DEFAULT_CABIN)
            elif ok and pending_name == "book_hotel":
                h = _HOTELS_BY_ID.get(pending_args.get("hotel_id"))
                if h is not None:
                    trip.hotel = h
                    trip.hotel_nights = int(pending_args.get("nights", 0) or 0)
                    trip.hotel_rate_code = pending_args.get("rate_code", DEFAULT_RATE_CODE)
            elif ok and pending_name == "add_activity":
                a = _ACTIVITIES_BY_ID.get(pending_args.get("activity_id"))
                if a is not None and a.id not in {x.id for x in trip.activities}:
                    trip.activities.append(a)
            pending_name = None
            pending_args = {}
    return trip


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
}

# Maps the searchable tool name to its description-artifact id.
SEARCHABLE_TOOL_DESCRIPTION_IDS: dict[str, str] = {
    "search_flights": "prompt.tool.search_flights.description",
    "search_hotels": "prompt.tool.search_hotels.description",
    "search_activities": "prompt.tool.search_activities.description",
}
