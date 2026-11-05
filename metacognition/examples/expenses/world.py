"""A simulated expense-claim world: one employee's trips, a company policy, and a checker.

The second task class for metacog's evidence (HARNESS_PLAN.md M5), built to
the study's rule that memory matters only when it carries knowledge the
model cannot derive (study E0, E6). The conventions live only in the policy
data the tools return, never in the tool descriptions:

- every foreign receipt converts at the company rate for the month the trip
  starts, whatever date is on the receipt;
- meals are reimbursed up to the city's daily allowance for each calendar day;
- hotels up to the city's nightly allowance for each night;
- a taxi over 40 US dollars needs a note saying why, or it is not reimbursed;
- alcohol is never reimbursed.

The drift variant changes one rule, as the client's note changed in the
study's E5: receipts convert at the rate for the month on each receipt.
Drift trips always cross a month boundary, so a stale procedure is rejected.

The checker accepts exactly one claim within five cents of what the policy
allows, and when it rejects it names the rule a wrong total most likely broke,
without revealing the allowed amount.
"""

from __future__ import annotations

import copy
import random
from collections import defaultdict
from datetime import date, timedelta
from typing import Any

EMPLOYEE = "Jordan Ellis"
COMPANY = "Brightline Labs"
TAXI_LIMIT_USD = 40.0
TOLERANCE = 0.05

CITIES = {
    "Lisbon": {"currency": "EUR", "meals": 70.0, "hotel": 190.0},
    "London": {"currency": "GBP", "meals": 85.0, "hotel": 260.0},
    "Mexico City": {"currency": "MXN", "meals": 55.0, "hotel": 160.0},
    "Osaka": {"currency": "JPY", "meals": 60.0, "hotel": 170.0},
    "Toronto": {"currency": "CAD", "meals": 75.0, "hotel": 220.0},
}
# Company rates: US dollars per unit, fixed for each month.
RATES = {
    "2026-09": {"EUR": 1.12, "GBP": 1.31, "MXN": 0.052, "JPY": 0.0068, "CAD": 0.74},
    "2026-10": {"EUR": 1.09, "GBP": 1.27, "MXN": 0.055, "JPY": 0.0071, "CAD": 0.72},
}
NOTES = {
    "rates": "Convert every foreign receipt at the company rate for the month the trip starts, whatever date "
             "is on the receipt.",
    "meals": "Meals are reimbursed up to the city's daily meal allowance for each calendar day; meal spending "
             "above a day's allowance is not reimbursed.",
    "hotel": "Hotels are reimbursed up to the city's nightly hotel allowance for each night.",
    "taxi": "A taxi over 40 US dollars is reimbursed only if its receipt has a note saying why; without a note it "
            "is not reimbursed.",
    "alcohol": "Alcohol is never reimbursed: leave out any receipt whose note mentions alcohol.",
    "total": "Claim the sum of what is reimbursed, in US dollars, rounded to the cent.",
}
DRIFT_RATES = "Convert each foreign receipt at the company rate for the month on the receipt."
MEAL_NOTES = ["", "", "lunch", "breakfast", "dinner"]
ALCOHOL_NOTES = ["team dinner, wine (alcohol)", "bar tab (alcohol)", "client dinner with beer (alcohol)"]
TAXI_NOTES = ["airport, late flight", "client visit across town", "equipment pickup"]


# ------------------------------------------------------------------ the policy's arithmetic

def reimbursable(trip: dict[str, Any], *, drift: bool = False, ignore: str = "") -> float:
    """What the policy allows for a trip, in dollars. `ignore` names one rule to break, for diagnosis."""
    city = CITIES[trip["city"]]
    start_month = trip["start"][:7]
    meals: dict[str, float] = defaultdict(float)
    total = 0.0
    for receipt in trip["receipts"]:
        receipt_month = receipt["date"][:7]
        by_receipt = drift != (ignore == "rates")  # the rule in force, or the other one when ignoring it
        usd = receipt["amount"] * RATES[receipt_month if by_receipt else start_month][receipt["currency"]]
        note = receipt["note"].strip()
        if "alcohol" in note.lower() and ignore != "alcohol":
            continue
        if receipt["category"] == "meal":
            meals[receipt["date"]] += usd
        elif receipt["category"] == "hotel":
            total += usd if ignore == "hotel" else min(usd, city["hotel"])
        elif receipt["category"] == "taxi":
            if usd > TAXI_LIMIT_USD and not note and ignore != "taxi":
                continue
            total += usd
    total += sum(spent if ignore == "meals" else min(spent, city["meals"]) for spent in meals.values())
    return round(total, 2)


DIAGNOSES = {
    "rates": ("it converted receipts at the rate for the month on each receipt, not the month the trip started",
              "it converted every receipt at the trip's starting month, not the month on each receipt"),
    "meals": ("it reimbursed meal spending above a day's meal allowance",) * 2,
    "hotel": ("it reimbursed hotel nights above the nightly allowance",) * 2,
    "taxi": ("it reimbursed a taxi over 40 US dollars that has no note",) * 2,
    "alcohol": ("it reimbursed a receipt for alcohol",) * 2,
}


def diagnose(trip: dict[str, Any], amount: float, *, drift: bool = False) -> str:
    for rule, messages in DIAGNOSES.items():
        if abs(reimbursable(trip, drift=drift, ignore=rule) - amount) <= TOLERANCE:
            return messages[1 if drift else 0]
    return "the amount is not what the expense policy allows"


# ------------------------------------------------------------------ the world the tools see

class ExpenseWorld:
    def __init__(self, task: dict[str, Any]) -> None:
        self.task = task
        self.trip = task["trip"]
        self.drift = bool(task.get("drift"))
        self.claims: list[tuple[str, float]] = []

    def read_trip(self, trip_id: str) -> dict[str, Any]:
        if trip_id.strip().upper() != self.trip["trip_id"]:
            return {"error": f"no trip {trip_id!r} for {EMPLOYEE}"}
        return copy.deepcopy(self.trip)

    def read_policy(self) -> dict[str, Any]:
        notes = [DRIFT_RATES if key == "rates" and self.drift else text for key, text in NOTES.items()]
        return {"company": COMPANY,
                "allowances_usd": {name: {"meals_per_day": c["meals"], "hotel_per_night": c["hotel"]}
                                   for name, c in CITIES.items()},
                "company_rates_usd_per_unit": copy.deepcopy(RATES), "notes": notes}

    def file_claim(self, trip_id: str, amount_usd: float) -> dict[str, Any]:
        self.claims.append((trip_id.strip().upper(), float(amount_usd)))
        return {"claim_id": f"C-{len(self.claims)}", "trip_id": trip_id, "amount_usd": round(float(amount_usd), 2),
                "status": "filed"}

    def check(self) -> dict[str, Any]:
        mine = [amount for trip_id, amount in self.claims if trip_id == self.trip["trip_id"]]
        if len(self.claims) != len(mine):
            return {"success": False, "reason": "a claim was filed for a different trip"}
        if not mine:
            return {"success": False, "reason": "no claim was filed"}
        if len(mine) > 1:
            return {"success": False, "reason": f"{len(mine)} claims were filed for one trip"}
        allowed = reimbursable(self.trip, drift=self.drift)
        if abs(mine[0] - allowed) <= TOLERANCE:
            return {"success": True, "reason": "accepted"}
        return {"success": False, "reason": f"the claim was {mine[0]:.2f} USD, but "
                                            f"{diagnose(self.trip, mine[0], drift=self.drift)}"}


# ------------------------------------------------------------------ trips

def _local(usd: float, currency: str, month: str) -> float:
    amount = usd / RATES[month][currency]
    return float(round(amount)) if currency == "JPY" else round(amount, 2)


def _trip(rng: random.Random, trip_id: str, *, crosses_month: bool) -> dict[str, Any]:
    city_name = rng.choice(sorted(CITIES))
    city = CITIES[city_name]
    nights = rng.randint(2, 4)
    # A crossing trip starts late enough in September (30 days) to end in October.
    start = date(2026, 9, rng.randint(31 - nights, 30)) if crosses_month else date(2026, 9, rng.randint(4, 20))
    days = [start + timedelta(days=d) for d in range(nights + 1)]
    month = start.isoformat()[:7]
    receipts: list[dict[str, Any]] = []

    def add(day: date, category: str, usd: float, note: str = "") -> None:
        receipts.append({"date": day.isoformat(), "category": category, "note": note,
                         "amount": _local(usd, city["currency"], month), "currency": city["currency"]})

    for day in days:
        for _ in range(rng.randint(1, 3)):
            add(day, "meal", rng.uniform(0.25, 0.75) * city["meals"], rng.choice(MEAL_NOTES))
    for night in days[:-1]:
        add(night, "hotel", city["hotel"] * rng.uniform(0.8, 1.25))
    for _ in range(rng.randint(1, 2)):
        add(rng.choice(days), "taxi", rng.uniform(15, 75), rng.choice(["", "", *TAXI_NOTES]))
    if rng.random() < 0.5:
        add(rng.choice(days), "meal", rng.uniform(20, 60), rng.choice(ALCOHOL_NOTES))
    receipts.sort(key=lambda r: r["date"])
    for number, receipt in enumerate(receipts, 1):
        receipt["id"] = f"R{number}"
    return {"trip_id": trip_id, "traveler": EMPLOYEE, "city": city_name, "start": days[0].isoformat(),
            "end": days[-1].isoformat(), "receipts": [{"id": r.pop("id"), **r} for r in receipts]}


def _rules_that_matter(trip: dict[str, Any], drift: bool) -> set[str]:
    allowed = reimbursable(trip, drift=drift)
    return {rule for rule in DIAGNOSES if abs(reimbursable(trip, drift=drift, ignore=rule) - allowed) > 1.0}


def task(number: int, *, drift: bool = False) -> dict[str, Any]:
    """One request, from its number alone: E-numbers before the policy change, F-numbers after."""
    rng = random.Random(number * 7919 + (104729 if drift else 0))
    trip_id = f"T-{number:03d}{'D' if drift else ''}"
    for _ in range(200):
        trip = _trip(rng, trip_id, crosses_month=drift or rng.random() < 0.5)
        rules = _rules_that_matter(trip, drift)
        # Keep trips where several conventions change the answer; drift trips must hinge on the changed rule.
        if len(rules) >= 3 and (not drift or "rates" in rules):
            break
    text = (f"Please file my expense claim for trip {trip_id}: {trip['city']}, {trip['start']} to "
            f"{trip['end']}.")
    return {"id": f"{'F' if drift else 'E'}{number:03d}", "text": text, "trip": trip, "drift": drift,
            "expected_usd": reimbursable(trip, drift=drift)}


SPLITS = {"dev": range(1, 17), "validation": range(101, 117), "test": range(201, 217)}
DRIFT_SPLITS = {"stream": range(701, 721), "gate": range(801, 807)}


def task_set(split: str) -> list[dict[str, Any]]:
    if split in DRIFT_SPLITS:
        return [task(n, drift=True) for n in DRIFT_SPLITS[split]]
    return [task(n) for n in SPLITS[split]]


def task_by_id(task_id: str) -> dict[str, Any]:
    return task(int(task_id[1:]), drift=task_id.startswith("F"))
