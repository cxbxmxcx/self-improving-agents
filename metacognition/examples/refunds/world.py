"""A simulated returns desk: an online store's orders, its customers, a returns policy, and a checker.

D2's world (DOMAINS.md). As in the expense world, the conventions live only in
the policy the tools return, never in the tool descriptions:

- a return window counts from the delivery date, by category: electronics 30
  days, home 45, apparel 60, beauty 14; members (tier "plus") get 15 more days;
- inside the window the item's price goes back to the card; up to 30 days
  after it, the same amount is store credit; after that the return is declined;
- opened electronics carry a 15% restocking fee on the item's price, and opened
  beauty items can't be returned;
- each accessory missing from the return is deducted at its listed value;
- shipping is never refunded, and final-sale items can't be returned;
- except that an item that arrived damaged is refunded in full, shipping
  included, to the card, within 90 days of delivery, whatever else applies.

The drift variant is a holiday policy, of the kind stores announce in November:
items delivered from November 15 through December 31 can be returned for a card
refund until January 31, and their restocking fee is waived. Drift returns are
requested in January for holiday deliveries and always hinge on that rule, so a
stale procedure is wrong on them.

The checker accepts exactly one action on the return that matches what the
policy gives, to the cent. When it rejects, it says what was done and names the
rule the action most likely broke, without revealing the right answer.
"""

from __future__ import annotations

import copy
import random
from datetime import date, timedelta
from typing import Any

STORE = "Larkspur Home & Tech"
TOLERANCE = 0.01

WINDOWS = {"electronics": 30, "home": 45, "apparel": 60, "beauty": 14}
MEMBER_EXTRA_DAYS = 15
LATE_CREDIT_DAYS = 30
RESTOCKING_RATE = 0.15
DAMAGED_DAYS = 90
HOLIDAY_START, HOLIDAY_END, HOLIDAY_DEADLINE = date(2026, 11, 15), date(2026, 12, 31), date(2027, 1, 31)

NOTES = {
    "window": "Return windows count from the delivery date, not the order date.",
    "members": "Members (tier plus) get 15 extra days on every return window.",
    "late": "Inside the window, refund the item's price to the card. Up to 30 days after the window closes, give "
            "the same amount as store credit instead. After that, decline the return.",
    "restocking": "Opened electronics carry a 15% restocking fee on the item's price.",
    "beauty": "Opened beauty items can't be returned.",
    "accessories": "Deduct the listed value of every accessory missing from the return.",
    "shipping": "Shipping is never refunded.",
    "final_sale": "Final-sale items can't be returned.",
    "damaged": "An item that arrived damaged is refunded in full to the card, its shipping included, within 90 days "
               "of delivery, whatever else applies.",
}
HOLIDAY = ("Holiday returns: items delivered from November 15 through December 31 can be returned for a refund to "
           "the card until January 31, and their restocking fee is waived.")

CATALOG = {
    "electronics": [("wireless headphones", 89, 249, [("charging case", 25.0), ("USB-C cable", 9.0)]),
                    ("smart speaker", 59, 199, [("power adapter", 15.0)]),
                    ("e-reader", 99, 179, [("cover", 19.0)]),
                    ("action camera", 149, 349, [("mount kit", 22.0), ("spare battery", 29.0)])],
    "home": [("blender", 49, 129, [("travel cup", 12.0)]), ("desk lamp", 29, 79, []),
             ("cookware set", 79, 199, [("glass lids", 18.0)])],
    "apparel": [("rain jacket", 69, 189, []), ("running shoes", 59, 149, []), ("wool sweater", 49, 119, [])],
    "beauty": [("hair dryer", 39, 129, [("diffuser", 14.0)]), ("skincare set", 25, 79, []), ("perfume", 45, 110, [])],
}
SHIPPING = (4.99, 7.99, 12.99)
REASONS = ("changed my mind", "wrong size", "not what I expected", "arrived damaged")
NAMES = ("Alex Kim", "Sam Rivera", "Priya Nair", "Taylor Brooks", "Morgan Diaz", "Casey Wong", "Riley Chen",
         "Dana Okafor", "Jamie Silva", "Avery Patel")


def _day(text: str) -> date:
    return date.fromisoformat(text)


# ------------------------------------------------------------------ the policy's arithmetic

def outcome(task: dict[str, Any], *, drift: bool | None = None, ignore: str = "") -> tuple[str, float]:
    """What the policy gives a return: ("refund" to the card, "credit", or "decline", amount in dollars).

    `ignore` names one rule to break, the way a wrong decision most likely broke it, for diagnosis.
    """
    ret, order, customer = task["return"], task["order"], task["customer"]
    item = next(i for i in order["items"] if i["sku"] == ret["sku"])
    drift = task["drift"] if drift is None else drift
    delivered, requested = _day(order["delivered_on"]), _day(ret["requested_on"])
    since = (requested - (_day(order["ordered_on"]) if ignore == "window" else delivered)).days
    holiday = drift and ignore != "holiday" and HOLIDAY_START <= delivered <= HOLIDAY_END
    if ret["reason"] == "arrived damaged" and ignore != "damaged":
        if (requested - delivered).days <= DAMAGED_DAYS:
            return "refund", round(item["price"] + item["shipping"], 2)
        return "decline", 0.0
    if item["final_sale"] and ignore != "final_sale":
        return "decline", 0.0
    if item["category"] == "beauty" and ret["opened"] and ignore != "beauty":
        return "decline", 0.0
    amount = item["price"] + (item["shipping"] if ignore == "shipping" else 0.0)
    if item["category"] == "electronics" and ret["opened"] and not holiday and ignore != "restocking":
        amount -= RESTOCKING_RATE * item["price"]
    if ignore != "accessories":
        amount -= sum(a["value"] for a in item["accessories"] if a["name"] in ret["missing_accessories"])
    amount = round(max(amount, 0.0), 2)
    extra = MEMBER_EXTRA_DAYS if customer["tier"] == "plus" and ignore != "members" else 0
    window = WINDOWS[item["category"]] + extra
    if since <= window or (holiday and requested <= HOLIDAY_DEADLINE):
        return "refund", amount
    if since <= window + LATE_CREDIT_DAYS:
        if ignore == "late":
            return "refund", amount
        if ignore == "late_decline":
            return "decline", 0.0
        return "credit", amount
    return "decline", 0.0


# The mistakes a checker recognizes, most telling first.
DIAGNOSES = {
    "holiday": "it applied the regular windows and fees to a holiday delivery",
    "damaged": "it applied the normal rules to an item that arrived damaged",
    "final_sale": "it accepted a final-sale item",
    "beauty": "it accepted an opened beauty item",
    "shipping": "it refunded the shipping",
    "restocking": "it didn't take the restocking fee on opened electronics",
    "accessories": "it didn't deduct the missing accessories",
    "window": "it counted the return window from the order date, not the delivery date",
    "members": "it didn't give the member's 15 extra days",
    "late": "it refunded to the card after the return window had closed",
    "late_decline": "it declined a return that was due store credit",
}


def _same(a: tuple[str, float], b: tuple[str, float]) -> bool:
    return a[0] == b[0] and abs(a[1] - b[1]) <= TOLERANCE


def diagnose(task: dict[str, Any], action: tuple[str, float]) -> str:
    for rule, message in DIAGNOSES.items():
        if _same(outcome(task, ignore=rule), action):
            return message
    return "that is not what the returns policy gives"


def describe(action: tuple[str, float]) -> str:
    kind, amount = action
    if kind == "decline":
        return "the return was declined"
    return f"{amount:.2f} USD went {'back to the card' if kind == 'refund' else 'out as store credit'}"


# ------------------------------------------------------------------ the world the tools see

class ReturnsWorld:
    def __init__(self, task: dict[str, Any]) -> None:
        self.task = task
        self.drift = bool(task.get("drift"))
        self.actions: list[tuple[str, str, float]] = []  # (return id, kind, amount)

    def read_return(self, return_id: str) -> dict[str, Any]:
        if return_id.strip().upper() != self.task["return"]["return_id"]:
            return {"error": f"no return {return_id!r}"}
        return copy.deepcopy(self.task["return"])

    def read_order(self, order_id: str) -> dict[str, Any]:
        if order_id.strip().upper() != self.task["order"]["order_id"]:
            return {"error": f"no order {order_id!r}"}
        return copy.deepcopy(self.task["order"])

    def read_customer(self, customer_id: str) -> dict[str, Any]:
        if customer_id.strip().upper() != self.task["customer"]["customer_id"]:
            return {"error": f"no customer {customer_id!r}"}
        return copy.deepcopy(self.task["customer"])

    def read_policy(self) -> dict[str, Any]:
        return {"store": STORE, "return_window_days": dict(WINDOWS), "member_extra_days": MEMBER_EXTRA_DAYS,
                "store_credit_days_after_window": LATE_CREDIT_DAYS,
                "restocking_fee_opened_electronics": RESTOCKING_RATE, "damaged_refund_days": DAMAGED_DAYS,
                "notes": [*NOTES.values(), *([HOLIDAY] if self.drift else [])]}

    def issue_refund(self, return_id: str, amount_usd: float, method: str) -> dict[str, Any]:
        if method not in ("card", "store_credit"):
            return {"error": "method must be card or store_credit"}
        kind = "refund" if method == "card" else "credit"
        self.actions.append((return_id.strip().upper(), kind, round(float(amount_usd), 2)))
        return {"refund_id": f"RF-{len(self.actions)}", "return_id": return_id,
                "amount_usd": round(float(amount_usd), 2), "method": method, "status": "issued"}

    def decline_return(self, return_id: str, reason: str) -> dict[str, Any]:
        self.actions.append((return_id.strip().upper(), "decline", 0.0))
        return {"return_id": return_id, "status": "declined", "reason": reason}

    def expected(self) -> tuple[str, float]:
        return outcome(self.task)

    def check(self) -> dict[str, Any]:
        return_id = self.task["return"]["return_id"]
        mine = [(kind, amount) for rid, kind, amount in self.actions if rid == return_id]
        if len(self.actions) != len(mine):
            return {"success": False, "reason": "an action was taken on a different return"}
        if not mine:
            return {"success": False, "reason": "nothing was done with the return"}
        if len(mine) > 1:
            return {"success": False, "reason": f"{len(mine)} actions were taken on one return"}
        if _same(mine[0], self.expected()):
            return {"success": True, "reason": "accepted"}
        return {"success": False, "reason": f"{describe(mine[0])}, but {diagnose(self.task, mine[0])}"}


# ------------------------------------------------------------------ returns

def _item(rng: random.Random, category: str, *, final_sale: bool) -> dict[str, Any]:
    name, low, high, accessories = rng.choice(CATALOG[category])
    return {"sku": f"SKU-{rng.randint(1000, 9999)}", "name": name, "category": category,
            "price": round(rng.randint(low, high) - 0.01, 2), "shipping": rng.choice(SHIPPING),
            "final_sale": final_sale, "accessories": [{"name": n, "value": v} for n, v in accessories]}


def _case(rng: random.Random, number: int, drift: bool) -> dict[str, Any]:
    category = rng.choice(sorted(WINDOWS))
    item = _item(rng, category, final_sale=rng.random() < 0.12)
    others = [_item(rng, c, final_sale=False) for c in rng.sample(sorted(WINDOWS), rng.randint(0, 2))]
    items = [item, *others]
    rng.shuffle(items)
    tier = "plus" if rng.random() < 0.35 else "standard"
    window = WINDOWS[category] + (MEMBER_EXTRA_DAYS if tier == "plus" else 0)
    if drift:  # a holiday delivery, returned in January
        delivered = HOLIDAY_START + timedelta(days=rng.randint(0, (HOLIDAY_END - HOLIDAY_START).days))
        requested = max(delivered + timedelta(days=3), date(2027, 1, rng.randint(2, 31)))
    else:  # near the window's edges, where the conventions decide the outcome
        delivered = date(2026, 8, 3) + timedelta(days=rng.randint(0, 70))
        since = rng.choice([rng.randint(3, window - 3), window - rng.randint(0, 6), window + rng.randint(1, 12),
                            window + rng.randint(13, LATE_CREDIT_DAYS), window + LATE_CREDIT_DAYS + rng.randint(1, 15)])
        requested = delivered + timedelta(days=max(since, 1))
    ordered = delivered - timedelta(days=rng.randint(2, 9))
    reason = rng.choices(REASONS, weights=(5, 3, 3, 1))[0]
    missing = [a["name"] for a in item["accessories"] if rng.random() < 0.4]
    suffix = "H" if drift else ""
    customer = {"customer_id": f"C-{rng.randint(100, 999)}{suffix}", "name": rng.choice(NAMES), "tier": tier}
    order = {"order_id": f"O-{number:03d}{suffix}", "customer_id": customer["customer_id"],
             "ordered_on": ordered.isoformat(), "delivered_on": delivered.isoformat(), "items": items}
    ret = {"return_id": f"RET-{number:03d}{suffix}", "order_id": order["order_id"], "sku": item["sku"],
           "reason": reason, "opened": rng.random() < 0.55, "missing_accessories": missing,
           "requested_on": requested.isoformat()}
    return {"return": ret, "order": order, "customer": customer, "drift": drift}


def rules_that_matter(task: dict[str, Any]) -> set[str]:
    expected = outcome(task)
    return {rule for rule in DIAGNOSES if not _same(outcome(task, ignore=rule), expected)}


def conventions_that_matter(task: dict[str, Any]) -> set[str]:
    """The policy conventions that change this return's outcome. Shipping changes every refund, so it doesn't
    count, and the two ways to get the late band wrong are one convention."""
    return {"late" if rule == "late_decline" else rule for rule in rules_that_matter(task)} - {"shipping"}


def task(number: int, *, drift: bool = False) -> dict[str, Any]:
    """One request, from its number alone: R-numbers before the holiday policy, H-numbers after."""
    rng = random.Random(number * 6151 + (49157 if drift else 0))
    # Regular returns rotate through the three outcomes; a holiday return is always a card refund.
    target = "refund" if drift else ("refund", "credit", "decline")[number % 3]
    for _ in range(2000):
        case = _case(rng, number, drift)
        kinds = conventions_that_matter(case)
        # A refund or credit should turn on two conventions. One is enough when it overrides the rest: a
        # decline (final sale, an opened beauty item, too late), damage on arrival, or the holiday rule.
        needed = 1 if target == "decline" or drift or "damaged" in kinds else 2
        if outcome(case)[0] == target and len(kinds) >= needed and (not drift or "holiday" in kinds):
            break
    else:
        raise RuntimeError(f"no return {number} meets the generator's conditions")
    ret = case["return"]
    text = f"Please process return {ret['return_id']} for order {ret['order_id']}."
    return {"id": f"{'H' if drift else 'R'}{number:03d}", "text": text, **case, "expected": outcome(case)}


SPLITS = {"dev": range(1, 17), "validation": range(101, 117), "test": range(201, 217)}
DRIFT_SPLITS = {"stream": range(701, 761), "gate": range(801, 807)}


def task_set(split: str) -> list[dict[str, Any]]:
    if split in DRIFT_SPLITS:
        return [task(n, drift=True) for n in DRIFT_SPLITS[split]]
    return [task(n) for n in SPLITS[split]]


def task_by_id(task_id: str) -> dict[str, Any]:
    return task(int(task_id[1:]), drift=task_id.startswith("H"))
