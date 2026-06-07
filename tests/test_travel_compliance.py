"""The hidden company-policy compliance substrate: rules and conflicts.

Deterministic, no LLM. A fully-compliant trip scores 1.0; the two rule conflicts
(no-red-eye vs prefer-JetBlue on LAX->JFK, and refundable-long-stay vs nightly-cap
on a 4-night stay) cap the achievable score below 1.0, which is the ruggedness.
"""

from __future__ import annotations

from agents.travel_compliance import score_compliance
from agents.travel_sim import ACTIVITIES, FLIGHTS, HOTELS, TripState

_BY_F = {f.id: f for f in FLIGHTS}
_BY_H = {h.id: h for h in HOTELS}
_BY_A = {a.id: a for a in ACTIVITIES}


def _trip(flight_id, cabin, hotel_id, nights, rate, activity_ids):
    t = TripState()
    t.flight, t.flight_cabin = _BY_F[flight_id], cabin
    t.hotel, t.hotel_nights, t.hotel_rate_code = _BY_H[hotel_id], nights, rate
    t.activities = [_BY_A[a] for a in activity_ids]
    return t


def test_fully_compliant_trip_scores_one():
    # SFO->JFK 2 nights: economy, JetBlue daytime FL101, prepaid Q, nightly under cap,
    # at most two activities. Every rule satisfied.
    task = {"required": {"flight": {"origin": "SFO", "destination": "JFK", "date": "2026-06-15"},
                         "hotel": {"city": "JFK", "nights": 2}, "activities": {"city": "JFK", "min_count": 1}}}
    trip = _trip("FL101", "economy", "HT204", 2, "Q", ["AC301"])
    score, violations = score_compliance(trip, task)
    assert score == 1.0
    assert violations == []


def test_no_red_eye_outranks_jetblue_on_the_conflict_route():
    # LAX->JFK: the only JetBlue flight (FL107) is a red-eye; FL113 is Delta daytime.
    # R2 (no-red-eye, weight 2) outranks R3 (prefer-JetBlue, weight 1), so the daytime
    # Delta flight scores higher than the JetBlue red-eye.
    task = {"required": {"flight": {"origin": "LAX", "destination": "JFK", "date": "2026-06-15"},
                         "hotel": {"city": "JFK", "nights": 2}, "activities": {"city": "JFK", "min_count": 1}}}
    daytime = _trip("FL113", "economy", "HT204", 2, "Q", ["AC301"])   # Delta daytime, violates R3
    redeye = _trip("FL107", "economy", "HT204", 2, "Q", ["AC301"])    # JetBlue red-eye, violates R2
    assert score_compliance(daytime, task)[0] > score_compliance(redeye, task)[0]
    assert score_compliance(daytime, task)[0] < 1.0  # the conflict caps it below 1.0


def test_long_stay_needs_refundable_under_the_cap():
    # 4 nights requires the refundable rate (B); but B is 1.5x, so a four-star hotel
    # busts the $300 cap. Only a cheaper-base hotel satisfies both rules.
    task = {"required": {"flight": {"origin": "SFO", "destination": "JFK", "date": "2026-06-15"},
                         "hotel": {"city": "JFK", "nights": 4}, "activities": {"city": "JFK", "min_count": 1}}}
    ok = _trip("FL101", "economy", "HT202", 4, "B", ["AC301"])        # base 189 -> B 283 <= 300
    bust = _trip("FL101", "economy", "HT204", 4, "B", ["AC301"])      # base 246 -> B 369 > 300
    assert score_compliance(ok, task)[0] > score_compliance(bust, task)[0]


def test_activity_cap_penalizes_more_than_two():
    task = {"required": {"flight": {"origin": "SFO", "destination": "JFK", "date": "2026-06-15"},
                         "hotel": {"city": "JFK", "nights": 2}, "activities": {"city": "JFK", "min_count": 1}}}
    two = _trip("FL101", "economy", "HT204", 2, "Q", ["AC301", "AC303"])
    three = _trip("FL101", "economy", "HT204", 2, "Q", ["AC301", "AC303", "AC305"])
    assert score_compliance(two, task)[0] > score_compliance(three, task)[0]
