"""The persona-rubric substrate: multi-modal scoring and the Tier 2 simulator.

Deterministic, no LLM. The rubric must be multi-modal (no single trip pleases
every persona) and conjunctive (a wrong dimension is not rescued by the others),
which is what makes the search problem real.
"""

from __future__ import annotations

from agents.travel_persona import (
    PERSONAS, _CURRENT_PERSONA, _simulate_answer, score_persona,
)
from agents.travel_sim import ACTIVITIES, FLIGHTS, HOTELS, TripState

_BY_F = {f.id: f for f in FLIGHTS}
_BY_H = {h.id: h for h in HOTELS}
_BY_A = {a.id: a for a in ACTIVITIES}
REQ = {"city": "JFK", "nights": 2}


def _trip(flight_id, cabin, hotel_id, rate, activity_ids):
    t = TripState()
    t.flight, t.flight_cabin = _BY_F[flight_id], cabin
    t.hotel, t.hotel_nights, t.hotel_rate_code = _BY_H[hotel_id], 2, rate
    t.activities = [_BY_A[a] for a in activity_ids]
    return t


def test_score_persona_returns_fraction_and_feedback():
    s, fb = score_persona(_trip("FL101", "economy", "HT204", "Q", ["AC303"]), PERSONAS["family"], REQ)
    assert 0.0 <= s <= 1.0
    assert isinstance(fb, str) and fb


def test_persona_objective_is_multi_modal():
    # A luxury-ideal trip (top hotel, business cabin, fine dining + nightlife) should
    # please luxury but fail the family (nightlife penalty, no economy nonstop) and
    # the budget-conscious solo traveler, so no single trip maxes everyone.
    luxe = _trip("FL101", "business", "HT201", "Q", ["AC302", "AC307", "AC303"])
    assert score_persona(luxe, PERSONAS["luxury"], REQ)[0] > 0.7
    assert score_persona(luxe, PERSONAS["family"], REQ)[0] < 0.4
    assert score_persona(luxe, PERSONAS["solo"], REQ)[0] < 0.4


def test_scoring_is_conjunctive_a_wrong_dimension_is_not_rescued():
    # Family gets its ideal activities and hotel but a nightlife activity it dislikes
    # should drag the whole rating down, not be averaged away.
    good = _trip("FL101", "economy", "HT203", "Q", ["AC305", "AC306"])      # outdoor + landmark
    bad = _trip("FL101", "economy", "HT203", "Q", ["AC307", "AC307"])       # nightlife (disliked)
    assert score_persona(good, PERSONAS["family"], REQ)[0] > score_persona(bad, PERSONAS["family"], REQ)[0]


def test_no_booking_scores_near_zero():
    assert score_persona(TripState(), PERSONAS["family"], REQ)[0] < 0.1


def test_simulator_answers_in_character_per_persona():
    assert "no concern" in _simulate_answer("What's your budget?", PERSONAS["luxury"]).lower() \
        or "best" in _simulate_answer("What's your budget?", PERSONAS["luxury"]).lower()
    assert "cheap" in _simulate_answer("What's your budget?", PERSONAS["solo"]).lower() \
        or "tight" in _simulate_answer("What's your budget?", PERSONAS["solo"]).lower()
    assert "kid" in _simulate_answer("Who is travelling?", PERSONAS["family"]).lower()


def test_simulator_reads_the_current_persona_contextvar():
    token = _CURRENT_PERSONA.set(PERSONAS["business"])
    try:
        assert _CURRENT_PERSONA.get().name == "business"
    finally:
        _CURRENT_PERSONA.reset(token)
    assert _CURRENT_PERSONA.get() is None
