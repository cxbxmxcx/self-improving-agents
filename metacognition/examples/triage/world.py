"""D7, a support desk: assigning a ticket is routine, but triaging one means weighing what a thread really establishes.

A ticket's thread has a customer's impressions, a first note from engineering that is sometimes wrong, and later notes that
correct it. The severity follows from what engineering has established and from a policy written in prose, and the team
follows from the part of the product that is affected, with one exception.

The words of the requests and the sentences of the threads come from banks.json, written by an agent that saw neither the
method under test nor its results.
"""

from __future__ import annotations

import json
import random
from functools import cache
from pathlib import Path
from typing import Any

OWNERS = ("Dana", "Imran", "Keiko", "Luis", "Mira", "Tobias")
AREAS = ("billing", "login", "search", "mobile")
TEAMS = {"billing": "payments", "login": "identity", "search": "discovery", "mobile": "apps"}
SECURITY_TEAM = "secops"
SEVERITIES = ("P1", "P2", "P3", "P4")
STYLES = ("casual", "shorthand", "distractor", "buried")
POLICY = ["Judge a ticket by what engineering has established. A customer's impression doesn't count, and where engineering "
          "has spoken more than once on a point, its latest statement stands.",
          "P1: any security exposure, any loss of customer data in production, or production being down for all users with "
          "no workaround.",
          "P2: anything else in production with no workaround.",
          "P3: anything else in production, where a workaround exists.",
          "P4: anything that affects only staging.",
          "A ticket from an enterprise customer is one level more severe than these rules give, except that P1 stays P1.",
          f"Team: a security exposure goes to {SECURITY_TEAM} whatever part of the product it is in. Otherwise billing goes to "
          f"{TEAMS['billing']}, login to {TEAMS['login']}, search to {TEAMS['search']}, and the mobile app to {TEAMS['mobile']}."]
SEARCH_BASES = range(201, 205)
GATE_BASES = range(205, 209)
SPLITS = {"dev": range(1, 17), "test": range(201, 217)}
OTHERWISE = {"environment": ("production", "staging"), "scope": ("all", "some", "one"), "workaround": ("yes", "no")}


@cache
def banks() -> dict[str, Any]:
    return json.loads((Path(__file__).parent / "banks.json").read_text(encoding="utf-8"))


def severity_of(facts: dict[str, str], plan: str) -> str:
    production, workaround = facts["environment"] == "production", facts["workaround"] == "yes"
    if facts["security"] == "yes" or (facts["data_loss"] == "yes" and production) or (
            production and facts["scope"] == "all" and not workaround):
        return "P1"
    level = 1 if production and not workaround else 2 if production else 3
    return SEVERITIES[level - 1 if plan == "enterprise" else level]


def team_of(facts: dict[str, str]) -> str:
    return SECURITY_TEAM if facts["security"] == "yes" else TEAMS[facts["area"]]


def _audience(number: int) -> str:
    if number < 200:
        return "dev"
    return "search" if number in SEARCH_BASES else "gate" if number in GATE_BASES else "held"


def _facts(rng: random.Random) -> dict[str, str]:
    return {"environment": rng.choice(("production", "production", "staging")), "scope": rng.choice(OTHERWISE["scope"]),
            "workaround": rng.choice(OTHERWISE["workaround"]), "data_loss": "yes" if rng.random() < 0.15 else "no",
            "security": "yes" if rng.random() < 0.15 else "no", "area": rng.choice(AREAS)}


def _thread(rng: random.Random, facts: dict[str, str], words: dict[str, Any]) -> list[dict[str, str]]:
    """A customer's impressions, engineering's first note, and a later note that corrects it when it was wrong."""

    def line(voice: str, fact: str, value: str | None = None) -> str:
        return rng.choice(words[voice][fact][facts[fact] if value is None else value])

    def different(fact: str) -> str:
        return rng.choice([value for value in OTHERWISE[fact] if value != facts[fact]])

    def impression(fact: str) -> str:
        return facts[fact] if rng.random() < 0.55 else different(fact)

    points = ("environment", "scope", "workaround")
    corrected = rng.choice(points) if rng.random() < 0.4 else None
    first_note = {fact: different(fact) if fact == corrected else facts[fact] for fact in points}
    last_note = [line("engineer", "area"), line("engineer", "data_loss"), line("engineer", "security")]
    if corrected is not None:
        last_note.append(line("engineer", corrected))
    return [{"from": "customer", "text": " ".join([rng.choice(words["openers"]), line("customer", "area"),
                                                   line("customer", "environment", impression("environment")),
                                                   line("customer", "scope", impression("scope"))])},
            {"from": "support", "text": rng.choice(words["acks"])},
            {"from": "engineer", "text": " ".join(line("engineer", fact, first_note[fact]) for fact in points)},
            {"from": "customer", "text": line("customer", "workaround", impression("workaround"))},
            {"from": "engineer", "text": " ".join(last_note)}]


def _quiet_thread(rng: random.Random, words: dict[str, Any]) -> list[dict[str, str]]:
    return [{"from": "customer", "text": rng.choice(words["openers"])}, {"from": "support", "text": rng.choice(words["acks"])}]


def _ticket(number: int, seed: str, audience: str) -> dict[str, Any]:
    """A ticket's world. Development and search requests share one set of thread sentences."""
    rng = random.Random(seed)
    words = banks()["thread"]["dev" if audience in ("dev", "search") else audience]
    facts = _facts(rng)
    plan = rng.choice(("standard", "standard", "enterprise"))
    return {"ticket_id": f"TK-{500 + number}", "plan": plan, "facts": facts, "severity": severity_of(facts, plan),
            "team": team_of(facts), "thread": _thread(rng, facts, words), "quiet": _quiet_thread(rng, words)}


def _assign_base(number: int) -> dict[str, Any]:
    rng = random.Random(f"assign-{number}")
    ticket = _ticket(number, f"ticket-{number}", _audience(number))
    return {"id": f"K{number:03d}", "base": f"K{number:03d}", "kind": "assign", "owner": rng.choice(OWNERS), "ticket": ticket,
            "text": f"Assign {ticket['ticket_id']}."}


def _triage_task(number: int, style: str) -> dict[str, Any]:
    audience = _audience(number)
    ticket = _ticket(number, f"triage-{number}-{style}", audience)
    fields = {"ticket": ticket["ticket_id"], "number": ticket["ticket_id"].split("-")[1],
              "other": f"TK-{540 + number}"}
    choices = banks()["triage_requests"][audience][style]
    return {"id": f"K{number:03d}~{style}", "base": f"K{number:03d}", "kind": "triage", "style": style, "expect": "act",
            "ticket": ticket, "text": choices[number % len(choices)].format(**fields)}


def _assign_task(number: int, style: str) -> dict[str, Any]:
    base = _assign_base(number)
    ticket = base["ticket"]
    fields = {"ticket": ticket["ticket_id"], "number": ticket["ticket_id"].split("-")[1], "other": f"TK-{540 + number}",
              "owner": base["owner"]}
    choices = banks()["assign_requests"][_audience(number)][style]
    return {**base, "id": f"K{number:03d}~{style}", "style": style, "expect": "act",
            "text": choices[number % len(choices)].format(**fields)}


def task_set(split: str) -> list[dict[str, Any]]:
    """A split's base requests: odd numbers assign a ticket, even numbers triage one."""
    return [_assign_base(n) if n % 2 else {"id": f"K{n:03d}", "base": f"K{n:03d}", "kind": "triage", "text": "Triage a ticket."}
            for n in SPLITS[split]]


@cache
def worded(split: str) -> tuple[dict[str, Any], ...]:
    """A split's requests as worded, four to a base, with the out-of-scope ones."""
    out: list[dict[str, Any]] = []
    for number in SPLITS[split]:
        make = _assign_task if number % 2 else _triage_task
        out += [make(number, style) for style in STYLES]
    bases = [t for t in task_set(split) if t["kind"] == "assign"][:2]
    for n, template in enumerate(banks()["out_of_scope"][split]):
        base = bases[n % 2]
        out.append({**base, "id": f"{base['id']}~out{n}", "style": "out_of_scope", "expect": "handoff",
                    "text": template.format(ticket=base["ticket"]["ticket_id"])})
    return tuple(out)


def _near_misses(split: str, audience: str, tag: str) -> list[dict[str, Any]]:
    """Each near miss on two of a split's tickets: the words of the work, but nothing should change."""
    bases = [t for t in task_set(split) if t["kind"] == "assign"][:2]
    return [{**base, "id": f"{base['id']}~{tag}{n}", "style": "near_miss", "expect": "handoff",
             "text": template.format(ticket=base["ticket"]["ticket_id"], owner=base["owner"])}
            for base in bases for n, template in enumerate(banks()["near_miss"][audience])]


def load(split: str) -> list[dict[str, Any]]:
    """The requests of a split, by the names the decision loop uses for them."""
    near = {"gate": ("gate", "gate"), "stress": ("search", "near"), "stress2": ("held", "fresh")}
    if split in near:
        return _near_misses("test", *near[split])
    if split == "dev+near":
        return list(worded("dev")) + _near_misses("dev", "dev", "neardev")
    return list(worded(split))


class TriageWorld:
    """One request's ticket, the triage policy, and what the desk did."""

    def __init__(self, task: dict[str, Any]) -> None:
        self.task = task
        self.ticket = task["ticket"]
        self.style = task.get("style", "plain")
        self.actions: list[tuple[str, dict[str, str]]] = []

    def _known(self, ticket_id: str) -> bool:
        return ticket_id.strip().upper() == self.ticket["ticket_id"]

    def read_ticket(self, ticket_id: str) -> dict[str, Any]:
        if not self._known(ticket_id):
            return {"error": f"no ticket {ticket_id!r}"}
        thread = self.ticket["thread"] if self.task["kind"] == "triage" else self.ticket["quiet"]
        return {"ticket_id": self.ticket["ticket_id"], "thread": [dict(message) for message in thread]}

    def read_account(self, ticket_id: str) -> dict[str, Any]:
        return {"plan": self.ticket["plan"]} if self._known(ticket_id) else {"error": f"no ticket {ticket_id!r}"}

    def read_policy(self) -> dict[str, Any]:
        return {"rules": list(POLICY)}

    def triage_ticket(self, ticket_id: str, severity: str, team: str) -> dict[str, Any]:
        arguments = {"ticket_id": ticket_id.strip().upper(), "severity": severity.strip().upper(), "team": team.strip().lower()}
        self.actions.append(("triage_ticket", arguments))
        return {"status": "triaged", **arguments}

    def assign_ticket(self, ticket_id: str, owner: str) -> dict[str, Any]:
        arguments = {"ticket_id": ticket_id.strip().upper(), "owner": owner.strip().capitalize()}
        self.actions.append(("assign_ticket", arguments))
        return {"status": "assigned", **arguments}

    def check(self) -> dict[str, Any]:
        if not self.actions:
            return {"success": False, "reason": "nothing was changed"}
        if len(self.actions) > 1:
            return {"success": False, "reason": f"{len(self.actions)} changes were made where one was asked for"}
        name, arguments = self.actions[0]
        if arguments["ticket_id"] != self.ticket["ticket_id"]:
            return {"success": False, "reason": f"it changed {arguments['ticket_id']}, a different ticket"}
        if self.task["kind"] == "assign":
            if name != "assign_ticket":
                return {"success": False, "reason": f"it called {name} on a request to assign the ticket"}
            return ({"success": True, "reason": "accepted"} if arguments["owner"] == self.task["owner"]
                    else {"success": False, "reason": f"it assigned the ticket to {arguments['owner']}"})
        if name != "triage_ticket":
            return {"success": False, "reason": f"it called {name} on a request to triage the ticket"}
        if arguments["severity"] != self.ticket["severity"]:
            return {"success": False, "reason": f"it set severity {arguments['severity']}, which isn't what the policy gives"}
        if arguments["team"] != self.ticket["team"]:
            return {"success": False, "reason": f"it sent the ticket to {arguments['team']}, the wrong team"}
        return {"success": True, "reason": "accepted"}
