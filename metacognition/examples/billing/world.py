"""D6, a billing desk: work out the prorated credit an account is owed after a plan change, under company conventions.

The request is simple and the calculation is not. A credit depends on the billing cycle the change falls in, the term, the
promotion, whether seats were removed or the account cancelled, and two exceptions. The conventions are written down in the
billing policy, so a model that reads carefully can follow them, and a procedure learned from experience can compute them.

The words of the requests come from banks.json, written by an agent that saw neither the method under test nor its results.
"""

from __future__ import annotations

import copy
import json
import math
import random
from datetime import date, timedelta
from functools import cache
from pathlib import Path
from typing import Any

COMPANY = "Larkspur Software"
PRICES = {"starter": 12.0, "team": 24.0, "business": 39.0, "enterprise": 68.0}
STYLES = ("casual", "shorthand", "distractor", "buried")
TOLERANCE = 0.01
NOTES = ["A billing cycle starts on the account's billing day and ends the day before the billing day of the next month.",
         "A credit covers the days after the change date through the end of the cycle. The change date itself counts as used.",
         "A cycle's charge is the number of seats times the plan's price per seat. On a seat reduction, only the seats "
         "removed count.",
         "Annual accounts pay for ten months out of twelve, so one cycle's charge on an annual account is ten twelfths of "
         "what it would be on a monthly account.",
         "Any promotional discount comes off the cycle's charge first.",
         "A credit is worked out per day: the cycle's charge divided by the number of days in that cycle, times the days "
         "covered.",
         "A change made in the first two days of a cycle credits the whole of the cycle's charge.",
         "On Enterprise plans the last three days of a cycle are not refundable, so three days come off the days covered, "
         "never going below zero.",
         "A credit is rounded down to the cent."]
DIAGNOSES = {"seats": "on a seat reduction only the seats removed count",
             "annual": "an annual account's cycle is charged at ten twelfths",
             "promo": "the promotional discount comes off the charge first",
             "change_day": "the change date itself counts as used",
             "grace": "a change in the first two days of a cycle credits the whole charge",
             "tail": "the last three days of an Enterprise cycle are not refundable",
             "floor": "a credit is rounded down to the cent"}
SEARCH_BASES = range(201, 205)
GATE_BASES = range(205, 209)
SPLITS = {"dev": range(1, 17), "validation": range(101, 117), "test": range(201, 217)}


@cache
def banks() -> dict[str, Any]:
    return json.loads((Path(__file__).parent / "banks.json").read_text(encoding="utf-8"))


def _cycle(billing_day: int, on: date) -> tuple[date, date]:
    """The first and last day of the billing cycle that a date falls in."""
    month_start = date(on.year, on.month, 1)
    if on.day >= billing_day:
        start = date(on.year, on.month, billing_day)
    else:
        last_month = month_start - timedelta(days=1)
        start = date(last_month.year, last_month.month, billing_day)
    following = date(start.year + (start.month == 12), start.month % 12 + 1, billing_day)
    return start, following - timedelta(days=1)


def credit_for(account: dict[str, Any], *, ignore: str | None = None) -> float:
    """What the policy says the account is owed, or what it would come to with one convention missed."""
    change = account["change"]
    on = date.fromisoformat(change["date"])
    start, end = _cycle(account["billing_day"], on)
    seats = change["seats_removed"] if change["type"] == "seat_reduction" and ignore != "seats" else account["seats"]
    charge = seats * PRICES[account["plan"]]
    if account["term"] == "annual" and ignore != "annual":
        charge *= 10 / 12
    if ignore != "promo":
        charge *= 1 - account["promo_percent"] / 100
    if (on - start).days < 2 and ignore != "grace":
        amount = charge
    else:
        covered = (end - on).days + (ignore == "change_day")
        if account["plan"] == "enterprise" and ignore != "tail":
            covered = max(0, covered - 3)
        amount = charge * covered / ((end - start).days + 1)
    return round(amount, 2) if ignore == "floor" else math.floor(amount * 100 + 1e-9) / 100


def diagnose(account: dict[str, Any], amount: float) -> str:
    for rule, message in DIAGNOSES.items():
        if abs(credit_for(account, ignore=rule) - amount) <= TOLERANCE:
            return message
    return "the amount is not what the billing policy gives"


def _account(rng: random.Random, account_id: str) -> dict[str, Any]:
    while True:
        plan = rng.choice(sorted(PRICES))
        seats = rng.randint(3, 40)
        billing_day = rng.randint(1, 28)
        start = date(2026, rng.randint(1, 10), billing_day)
        following = date(start.year, start.month + 1, billing_day)
        on = start + timedelta(days=rng.randrange((following - start).days))
        cancels = rng.random() < 0.5
        change = {"type": "cancellation" if cancels else "seat_reduction", "date": on.isoformat(),
                  "seats_removed": seats if cancels else rng.randint(1, seats - 1)}
        account = {"account_id": account_id, "customer": rng.choice(["Halden Foods", "Corvid Labs", "Mirelle Studio", "Ostrand Freight",
                                                                     "Tessel Health", "Quillon Press"]),
                   "plan": plan, "term": rng.choice(["monthly", "annual"]), "seats": seats, "billing_day": billing_day,
                   "promo_percent": rng.choice([0, 0, 10, 20]), "change": change}
        if credit_for(account) >= 1.0:
            return account


def task(number: int) -> dict[str, Any]:
    account_id = f"A-{500 + number}"
    account = _account(random.Random(f"billing-{number}"), account_id)
    change = "cancellation" if account["change"]["type"] == "cancellation" else "seat reduction"
    return {"id": f"B{number:03d}", "base": f"B{number:03d}", "kind": "credit", "account": account,
            "expected_usd": credit_for(account), "text": f"Issue the credit {account_id} is owed for its recent {change}."}


def task_by_id(task_id: str) -> dict[str, Any]:
    return task(int(task_id[1:]))


def task_set(split: str) -> list[dict[str, Any]]:
    return [task(n) for n in SPLITS[split]]


def _audience(number: int) -> str:
    if number < 200:
        return "dev"
    return "search" if number in SEARCH_BASES else "gate" if number in GATE_BASES else "held"


def _fields(base: dict[str, Any]) -> dict[str, str]:
    account_id = base["account"]["account_id"]
    digits = account_id.split("-")[1]
    return {"account": account_id, "number": digits, "other": f"A-{int(digits) + 40}"}


def worded(split: str) -> list[dict[str, Any]]:
    """A split's requests as worded, four to a base, with the out-of-scope ones."""
    words, out = banks(), []
    for number in SPLITS[split]:
        base = task(number)
        for style in STYLES:
            choices = words["requests"][_audience(number)][style]
            text = choices[number % len(choices)].format(**_fields(base))
            out.append({**base, "id": f"{base['id']}~{style}", "style": style, "expect": "act", "text": text})
    bases = task_set(split)[:2]
    for n, template in enumerate(words["out_of_scope"][split]):
        base = bases[n % 2]
        out.append({**base, "id": f"{base['id']}~out{n}", "style": "out_of_scope", "expect": "handoff",
                    "text": template.format(account=base["account"]["account_id"])})
    return out


def _near_misses(split: str, audience: str, tag: str) -> list[dict[str, Any]]:
    """Each near miss on two of a split's requests: the words of the work, but nothing should be issued."""
    return [{**base, "id": f"{base['id']}~{tag}{n}", "text": template.format(account=base["account"]["account_id"]),
             "style": "near_miss", "expect": "handoff"}
            for base in task_set(split)[:2] for n, template in enumerate(banks()["near_miss"][audience])]


def load(split: str) -> list[dict[str, Any]]:
    """The requests of a split, by the names the decision loop uses for them."""
    near = {"gate": ("gate", "gate"), "stress": ("search", "near"), "stress2": ("held", "fresh")}
    if split in near:
        return _near_misses("test", *near[split])
    if split == "dev+near":
        return worded("dev") + _near_misses("dev", "dev", "neardev")
    return worded(split)


class BillingWorld:
    """One request's account, the billing policy, and the credits the desk issued."""

    def __init__(self, task_: dict[str, Any]) -> None:
        self.task = task_
        self.account = task_["account"]
        self.style = task_.get("style", "plain")
        self.credits: list[tuple[str, float]] = []

    def read_account(self, account_id: str) -> dict[str, Any]:
        if account_id.strip().upper() != self.account["account_id"]:
            return {"error": f"no account {account_id!r}"}
        return copy.deepcopy(self.account)

    def read_policy(self) -> dict[str, Any]:
        return {"company": COMPANY, "price_usd_per_seat": dict(PRICES), "notes": list(NOTES)}

    def issue_credit(self, account_id: str, amount_usd: float) -> dict[str, Any]:
        self.credits.append((account_id.strip().upper(), float(amount_usd)))
        return {"credit_id": f"CR-{len(self.credits)}", "account_id": account_id, "amount_usd": round(float(amount_usd), 2),
                "status": "issued"}

    def check(self) -> dict[str, Any]:
        mine = [amount for account_id, amount in self.credits if account_id == self.account["account_id"]]
        if len(self.credits) != len(mine):
            return {"success": False, "reason": "a credit was issued to a different account"}
        if not mine:
            return {"success": False, "reason": "no credit was issued"}
        if len(mine) > 1:
            return {"success": False, "reason": f"{len(mine)} credits were issued for one change"}
        if abs(mine[0] - credit_for(self.account)) <= TOLERANCE:
            return {"success": True, "reason": "accepted"}
        return {"success": False, "reason": f"the credit was {mine[0]:.2f} USD, but {diagnose(self.account, mine[0])}"}
