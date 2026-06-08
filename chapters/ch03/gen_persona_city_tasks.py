"""Generate the persona-city task set and its train/test split.

Five personas crossed with five structurally identical cities gives 25 trips. We
hold out one distinct city per persona (a city-balanced stratified split, so the
test set covers every persona once and every city once), which makes the held-out
score a real generalization measure: a memorized city cannot help, only a policy
that learned the persona-to-trip strategy transfers. The cities are parallel by
construction (agents.travel_sim), so a low test score means the strategy did not
generalize, not that the city was impoverished.

Run once to (re)write travel_persona_city_tasks.json:
    python chapters/ch03/gen_persona_city_tasks.py
"""
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agents.travel_persona import PERSONAS
from agents.travel_sim import _PERSONA_CITIES  # {code: City name}

# One held-out city per persona: balanced so each city is the test city exactly
# once and each persona is tested exactly once (20% per persona, 5 of 25 total).
HELD_OUT = {
    "family": "MIA", "luxury": "DEN", "solo": "BOS", "foodie": "ATL", "business": "AUS",
}

OUT = Path(__file__).parent / "travel_persona_city_tasks.json"


def request(persona: str, code: str, city: str) -> str:
    cue = PERSONAS[persona].cue
    return (f"Plan a 2-day trip to {city} (airport {code}) from SFO on 2026-06-15 "
            f"{cue}, with a hotel and some things to do.")


def main() -> None:
    train, test = [], []
    for persona in PERSONAS:
        for code, city in _PERSONA_CITIES.items():
            task = {
                "id": f"{persona}-{code.lower()}",
                "persona": persona,
                "request": request(persona, code, city),
                "required": {"city": code, "nights": 2},
            }
            (test if HELD_OUT[persona] == code else train).append(task)

    data = {
        "description": (
            "Persona-city tasks: five personas crossed with five structurally "
            "identical cities (agents.travel_sim parallel inventory). One distinct "
            "city is held out per persona (city-balanced stratified 20% split), so "
            "the test score measures whether the persona-tailoring strategy "
            "generalizes to an unseen city, not city-specific inventory luck. "
            "All trips are 2 nights from SFO; the airport code is given in each "
            "request to remove the city-code gotcha from this generalization test."
        ),
        "held_out": HELD_OUT,
        "train": train,
        "test": test,
    }
    OUT.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT}  train={len(train)} test={len(test)}")


if __name__ == "__main__":
    main()
