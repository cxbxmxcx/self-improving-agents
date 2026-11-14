"""Requests after four of the user's colleagues add scheduling rules to their calendar notes.

The week, people, conventions, and requests are the first class's (generator.py). Four
colleagues add a rule to their calendar note, the way people do:

    Leo Chen      Mornings are my focus time: please don't book me before 10:00.
    Priya Shah    I keep 12:00 to 13:00 free for lunch.
    Omar Haddad   No-meeting Wednesdays for me, please.
    Sam Ortiz     School pickup: I'm offline from 15:30 on Tuesdays and Thursdays.

Dana Park's note (15 minutes between meetings) is unchanged; the starting memory already reads it.
The checker refuses a booking that breaks a rule, with the reason a person would give. A task is
kept only when it has a solution, few starts work, and ignoring one named colleague's rule makes
the first booking wrong; seeds take turns naming Leo, Priya, Omar, and Sam, so every split needs
all four rules.

Listing 6.3 serves the first two requests after the rules appear (the first_contact split) with
the starting memory, then the stream while that memory is quarantined; the gate's held-back
cases and the test requests are separate seeds.
"""

from __future__ import annotations

import copy
import functools
import random
from typing import Any

from chapters.ch06.assistant import generator as base
from chapters.ch06.assistant.environment import (
    DAY_NAMES,
    SchedulingEnvironment,
    to_hhmm,
    to_minutes,
)

NOTES = {
    "leo": "Mornings are my focus time: please don't book me before 10:00.",
    "priya": "I keep 12:00 to 13:00 free for lunch.",
    "omar": "No-meeting Wednesdays for me, please.",
    "sam": "School pickup: I'm offline from 15:30 on Tuesdays and Thursdays.",
}
RULES = {
    "leo": {"not_before": "10:00"},
    "priya": {"keep_free": ["12:00", "13:00"]},
    "omar": {"no_days": ["wed"]},
    "sam": {"end_by": {"tue": "15:30", "thu": "15:30"}},
}
ORDER = ("leo", "priya", "omar", "sam")  # seeds take turns requiring each rule to decide the answer
# The first requests the starting memory serves, the stream served while it is quarantined,
# the gate's held-back cases, and test requests no learner or gate sees.
SPLITS = {"first_contact": range(17, 19), "stream": range(1, 17), "gate": range(801, 807),
          "test": range(201, 225)}
PREFIX = "R"


def with_rules(world: dict[str, Any], people: tuple[str, ...] = ORDER) -> dict[str, Any]:
    world = copy.deepcopy(world)
    for pid in people:
        world["profiles"][pid].update(note=NOTES[pid], rule=copy.deepcopy(RULES[pid]))
    return world


def without_rule(world: dict[str, Any], pid: str) -> dict[str, Any]:
    world = copy.deepcopy(world)
    world["profiles"][pid].pop("rule", None)
    return world


class RulesEnvironment(SchedulingEnvironment):
    """The first class's calendar week, whose checker also honors each colleague's rule."""

    def rule_refusal(self, person: str, day: str, start: int, minutes: int) -> str | None:
        rule = self.profiles[person].get("rule")
        if not rule:
            return None
        name = self.name(person)
        local = start + self.shift(person)
        end = local + minutes
        when = f"{DAY_NAMES[day]} {to_hhmm(start)}"
        if rule.get("not_before") and local < to_minutes(rule["not_before"]):
            return f"{name} asked not to be booked before {rule['not_before']}, and {when} is earlier"
        if rule.get("keep_free"):
            a, b = (to_minutes(t) for t in rule["keep_free"])
            if local < b and a < end:
                return (f"{name} keeps {rule['keep_free'][0]} to {rule['keep_free'][1]} free for lunch,"
                        f" and {when} overlaps it")
        if day in rule.get("no_days", ()):
            return f"{name} takes no meetings on {DAY_NAMES[day]}s"
        cutoff = rule.get("end_by", {}).get(day)
        if cutoff and end > to_minutes(cutoff):
            return f"{name} is offline from {cutoff} on {DAY_NAMES[day]}s, and {when} ends after that"
        return None

    def acceptable(self, day: str, start: int, attendees: list[str], minutes: int,
                   ignore: str | None = None) -> str | None:
        refused = super().acceptable(day, start, attendees, minutes, ignore=ignore)
        if refused:
            return refused
        for person in attendees:
            reason = self.rule_refusal(person, day, start, minutes)
            if reason:
                return reason
        return None


def _starts(task: dict[str, Any]) -> dict[str, list[int]]:
    return base._starts(RulesEnvironment(task), task["truth"])


def booked(found: dict[str, list[Any]], truth: dict[str, Any]) -> list[tuple[str, Any]]:
    """What booking the earliest start on the first `count` listed days with starts would book."""
    out: list[tuple[str, Any]] = []
    for day in truth["days"]:
        if found.get(day) and len(out) < truth["count"]:
            out.append((day, found[day][0]))
    return out


def deciding_rules(task: dict[str, Any]) -> list[str]:
    """The colleagues whose rule, ignored alone, makes the first booking wrong."""
    truth = task["truth"]
    right = booked(_starts(task), truth)
    invited = set(truth["attendees"])
    decided = []
    for pid in ORDER:
        if base.PEOPLE[pid] not in invited or not task["world"]["profiles"][pid].get("rule"):
            continue
        alt = _starts({**task, "world": without_rule(task["world"], pid)})
        if booked(alt, truth) != right:
            decided.append(pid)
    return decided


def solutions(task: dict[str, Any]) -> dict[str, list[str]]:
    return {d: [to_hhmm(s) for s in v] for d, v in _starts(task).items() if v}


def generate(seed: int) -> dict[str, Any]:
    return copy.deepcopy(_generate(seed))


@functools.cache
def _generate(seed: int, max_tries: int = 40000) -> dict[str, Any]:
    rng = random.Random(seed)
    need = ORDER[seed % len(ORDER)]
    count = 2 if seed % 3 == 0 else 1
    for _ in range(max_tries):
        world = with_rules(base._world(rng))
        text, truth = base._request(rng, count)
        if base.PEOPLE[need] not in truth["attendees"]:
            continue
        task = {"id": f"{PREFIX}{seed:03d}", "seed": seed, "class": "team_rules", "text": text,
                "world": world, "truth": truth}
        shared = _starts(task)
        if sum(1 for v in shared.values() if v) < truth["count"]:
            continue  # no solution
        if sum(len(v) for v in shared.values()) > 8:
            continue  # scarce, so a guess rarely lands
        decided = deciding_rules(task)
        if need not in decided:
            continue
        task["decided_by"] = decided
        return task
    raise RuntimeError(f"seed {seed}: no task met the constraints in {max_tries} tries")


def task_set(split: str) -> list[dict[str, Any]]:
    return [generate(seed) for seed in SPLITS[split]]
