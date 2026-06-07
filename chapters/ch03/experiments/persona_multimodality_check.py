"""Deterministic gate (no LLM): confirm the persona objective is multi-modal with
real headroom before spending on a search. For each persona it brute-forces the
best achievable rating and an untailored generic trip, then cross-scores each
persona's best trip against the others to show no single trip pleases everyone.
"""
import itertools
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agents.travel_persona import PERSONAS, score_persona
from agents.travel_sim import ACTIVITIES, FLIGHTS, HOTELS, TripState

CITY, NIGHTS, DATE, ORIGIN = "JFK", 2, "2026-06-15", "SFO"
REQ = {"city": CITY, "nights": NIGHTS}
flights = [f for f in FLIGHTS if f.origin == ORIGIN and f.destination == CITY and f.date == DATE]
hotels = [h for h in HOTELS if h.city == CITY]
acts = [a for a in ACTIVITIES if a.city == CITY]


def trip(f, cabin, h, rate, chosen):
    t = TripState()
    t.flight, t.flight_cabin = f, cabin
    t.hotel, t.hotel_nights, t.hotel_rate_code = h, NIGHTS, rate
    t.activities = list(chosen)
    return t


def best_for(p):
    best = (-1.0, None, "")
    for f in flights:
        for cabin in ("economy", "business"):
            for h in hotels:
                for rate in ("Q", "B"):
                    for k in range(0, p.pace + 2):
                        for chosen in itertools.combinations(acts, k):
                            s, _ = score_persona(trip(f, cabin, h, rate, chosen), p, REQ)
                            if s > best[0]:
                                best = (s, trip(f, cabin, h, rate, chosen),
                                        f"{f.id} {cabin} {h.id}(r{h.rating}) {rate} {[a.category for a in chosen]}")
    return best


def main():
    generic = trip(flights[0], "economy", next(h for h in hotels if h.id == "HT204"), "Q",
                   [a for a in acts if a.id in ("AC303", "AC306")])
    bests = {name: best_for(p) for name, p in PERSONAS.items()}
    print(f"{'persona':<10}{'ceiling':>9}{'generic':>9}   best trip")
    for name, p in PERSONAS.items():
        s, _, desc = bests[name]
        g, _ = score_persona(generic, p, REQ)
        print(f"{name:<10}{s:>9.3f}{g:>9.3f}   {desc}")
    print("\ncross-persona: row persona's BEST trip scored by column persona (multi-modality)")
    names = list(PERSONAS)
    print(f"{'best/judge':<12}" + "".join(f"{n:>9}" for n in names))
    for rn in names:
        row = bests[rn][1]
        cells = "".join(f"{score_persona(row, PERSONAS[cn], REQ)[0]:>9.3f}" for cn in names)
        print(f"{rn:<12}{cells}")


if __name__ == "__main__":
    main()
