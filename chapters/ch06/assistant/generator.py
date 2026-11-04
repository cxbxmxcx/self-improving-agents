"""Generate the first scheduling class from fixed seeds.

Each seed creates a fresh week of calendars and a request for one or two
meetings under day, duration, and time constraints. The request's parameters
provide the checker's hidden task specification. Retain cases with a solution,
few valid starts, and a naive interpretation that gives a wrong first answer.

The calendars expose offsets, working hours, free-status events, and an
external client's buffer between meetings. Seeds require different conventions
to affect the answer, giving each task set coverage of those rules. This
construction tests whether a procedure handles the conventions; a recorded
control separately measures what a writer produces without experience.
"""

from __future__ import annotations

import copy
import functools
import random
from typing import Any

from chapters.ch06.assistant.environment import (BUFFER, DAY_NAMES, DAYS, SchedulingEnvironment, offset_minutes,
                                                 to_hhmm, to_minutes)

WEEK = {"mon": "2026-10-19", "tue": "2026-10-20", "wed": "2026-10-21", "thu": "2026-10-22", "fri": "2026-10-23"}
HOURS = ("09:00", "17:00")
USER = "maya"
PEOPLE = {
    "maya": "Maya Rivera", "sam": "Sam Ortiz", "dana": "Dana Park", "priya": "Priya Shah",
    "leo": "Leo Chen", "ana": "Ana Silva", "omar": "Omar Haddad",
}
# Between meetings, not around the working day: the first draft said "free before
# and after", which left a meeting at the start of her day ambiguous.
NOTE = "External client. Please leave 15 minutes between any meeting with me and my other meetings."
PROFILES = {
    "maya": {"utc_offset": "-04:00"}, "sam": {"utc_offset": "-04:00"}, "priya": {"utc_offset": "-04:00"},
    "leo": {"utc_offset": "-04:00"}, "omar": {"utc_offset": "-03:00"}, "ana": {"utc_offset": "+01:00"},
    "dana": {"utc_offset": "-07:00", "external": True, "note": NOTE},
}
FREE_TITLES = ("Lunch and learn (optional)", "Gym", "Reading time", "Office hours (drop-in)")
TOPICS = ("project sync", "design review", "planning session", "budget check-in", "launch review",
          "hiring debrief")
# The task sets of section 7 of the plan: learning, selection, and a held-out test.
SPLITS = {"dev": range(1, 17), "validation": range(101, 117), "test": range(201, 217)}
STEP = 15  # starts fall on quarter hours
# The calendar conventions, in the order seeds take turns requiring them.
CONVENTIONS = ("timezone", "show_as", "buffer", "hours")


def profiles() -> dict[str, dict[str, Any]]:
    return {pid: {"hours": list(HOURS), **profile} for pid, profile in PROFILES.items()}


def _busy_day(rng: random.Random, open_: int, close: int, density: int) -> list[tuple[int, int]]:
    """Non-overlapping busy blocks for one person on one day, on their own clock."""
    blocks: list[tuple[int, int]] = []
    for _ in range(density * 3):
        if len(blocks) >= density:
            break
        minutes = rng.choice((30, 45, 60, 60, 90))
        start = rng.randrange(open_, close - minutes + 1, STEP)
        if all(start + minutes <= s or e <= start for s, e in blocks):
            blocks.append((start, start + minutes))
    return sorted(blocks)


def _world(rng: random.Random) -> dict[str, Any]:
    open_, close = (to_minutes(h) for h in HOURS)
    people = profiles()
    events, number = [], 1
    for pid in PEOPLE:
        # Calendars are drawn on each owner's clock, then stored on Maya's.
        shift = offset_minutes(people[pid]["utc_offset"]) - offset_minutes(people[USER]["utc_offset"])
        low, high = (1, 3) if pid == USER else (3, 5)
        for day in DAYS:
            for s, e in _busy_day(rng, open_, close, rng.randint(low, high)):
                events.append({"id": f"E{number:02d}", "owner": pid, "title": "Focus time" if pid == USER else "Busy",
                               "day": day, "start": to_hhmm(s - shift), "minutes": e - s, "show_as": "busy",
                               "attendees": [pid]})
                number += 1
            # Some events are shown as free; reading them as busy hides real openings.
            if pid != USER and rng.random() < 0.5:
                minutes = rng.choice((30, 45, 60))
                s = rng.randrange(open_, close - minutes + 1, STEP)
                events.append({"id": f"E{number:02d}", "owner": pid, "title": rng.choice(FREE_TITLES), "day": day,
                               "start": to_hhmm(s - shift), "minutes": minutes, "show_as": "free",
                               "attendees": [pid]})
                number += 1
    return {"user": USER, "week": dict(WEEK), "hours": list(HOURS), "people": dict(PEOPLE),
            "profiles": people, "events": events}


def _request(rng: random.Random, count: int) -> tuple[str, dict[str, Any]]:
    attendees = rng.sample([p for p in PEOPLE if p != USER], rng.choice((2, 2, 3)))
    minutes = rng.choice((30, 45, 60))
    first = rng.randrange(0, 3)
    days = list(DAYS[first:first + rng.choice((2, 3))])
    limit = rng.choice(("earliest", "latest_end"))
    bound = rng.choice(("12:30", "13:00", "14:00")) if limit == "earliest" else rng.choice(("12:00", "12:30", "13:00"))
    truth = {"kind": "booking", "attendees": [PEOPLE[p] for p in attendees], "minutes": minutes,
             "count": count, "days": days, limit: bound}
    if count == 2:
        truth["per_day_max"] = 1
    names = [PEOPLE[p] for p in attendees]
    who = ", ".join(names[:-1]) + f" and {names[-1]}"
    span = f"{DAY_NAMES[days[0]]} to {DAY_NAMES[days[-1]]}" if len(days) > 2 else \
        f"{DAY_NAMES[days[0]]} or {DAY_NAMES[days[1]]}"
    when = f"starting no earlier than {bound}" if limit == "earliest" else f"finished by {bound}"
    topic = rng.choice(TOPICS)
    if count == 1:
        text = f"Book a {minutes}-minute {topic} with {who} this week, {span}, {when}."
    else:
        # "each with both" keeps two meetings from reading as one per person.
        each = f"each with both {who}" if len(names) == 2 else f"each with all of {who}"
        text = f"Book two {minutes}-minute {topic}s, {each}, this week, {span}, on different days, {when}."
    return text, truth


def _window(env: SchedulingEnvironment, truth: dict[str, Any]) -> range:
    lo = max(env.open, to_minutes(truth["earliest"])) if truth.get("earliest") else env.open
    hi = min(env.close, to_minutes(truth["latest_end"])) if truth.get("latest_end") else env.close
    return range(lo, hi - truth["minutes"] + 1, STEP)


def _starts(env: SchedulingEnvironment, truth: dict[str, Any]) -> dict[str, list[int]]:
    """Every start, by day, that the checker would accept for one meeting."""
    attendees = [env.resolve(n) for n in truth["attendees"]]
    return {day: [s for s in _window(env, truth) if env.acceptable(day, s, attendees, truth["minutes"]) is None]
            for day in truth["days"]}


def face_value_starts(task: dict[str, Any]) -> dict[str, list[int]]:
    """The starts a reader who takes the calendars at face value would find:
    every listed time read as Maya's, every event blocking, no buffer, and
    no working hours. This is what a function written blind would compute."""
    env = SchedulingEnvironment(task)
    truth = task["truth"]
    people = [USER, *(env.resolve(n) for n in truth["attendees"])]
    found: dict[str, list[int]] = {}
    for day in truth["days"]:
        busy = []
        for person in people:
            shift = env.shift(person)
            busy += [(to_minutes(e["start"]) + shift, to_minutes(e["start"]) + shift + e["minutes"])
                     for e in env.events_for(person, day)]
        found[day] = [s for s in _window(env, truth) if all(s + truth["minutes"] <= a or b <= s for a, b in busy)]
    return found


def starts_misreading(task: dict[str, Any], convention: str) -> dict[str, list[int]]:
    """The starts found by a reader who gets every convention right but one."""
    env = SchedulingEnvironment(task)
    truth = task["truth"]
    people = [USER, *(env.resolve(n) for n in truth["attendees"])]
    found: dict[str, list[int]] = {}
    for day in truth["days"]:
        ok = []
        for s in _window(env, truth):
            e = s + truth["minutes"]
            fine = True
            for person in people:
                shift = env.shift(person)
                if convention != "hours":
                    first, last = (to_minutes(h) for h in env.profiles[person]["hours"])
                    fine &= first <= s + shift and e + shift <= last
                buffer = BUFFER if env.external(person) and convention != "buffer" else 0
                for ev in env.events_for(person, day):
                    if ev.get("show_as") == "free" and convention != "show_as":
                        continue
                    a = to_minutes(ev["start"]) + (shift if convention == "timezone" else 0)
                    fine &= not (s - buffer < a + ev["minutes"] and a < e + buffer)
            if fine:
                ok.append(s)
        found[day] = ok
    return found


def deciding_conventions(task: dict[str, Any]) -> list[str]:
    """The conventions that, misread alone, make the first start found wrong."""
    true = {d: [to_minutes(x) for x in v] for d, v in solutions(task).items()}
    decided = []
    for convention in CONVENTIONS:
        alt = starts_misreading(task, convention)
        first = next(((d, alt[d][0]) for d in task["truth"]["days"] if alt[d]), None)
        if first is None or first[1] not in true.get(first[0], []):
            decided.append(convention)
    return decided


def solutions(task: dict[str, Any]) -> dict[str, list[str]]:
    """The starts, by day, that the checker would accept for one meeting."""
    return {d: [to_hhmm(s) for s in v] for d, v in _starts(SchedulingEnvironment(task), task["truth"]).items() if v}


def generate(seed: int) -> dict[str, Any]:
    """The task for one seed; the same seed always gives the same task."""
    return copy.deepcopy(_generate(seed))


@functools.lru_cache(maxsize=None)
def _generate(seed: int, max_tries: int = 4000) -> dict[str, Any]:
    # Rejection sampling takes a moment per seed, so each task is drawn once.
    rng = random.Random(seed)
    for _ in range(max_tries):
        world = _world(rng)
        # Every fourth seed asks for two meetings, so each set has some.
        text, truth = _request(rng, 2 if seed % 4 == 0 else 1)
        task = {"id": f"S{seed:03d}", "seed": seed, "class": "multi_attendee", "text": text,
                "world": world, "truth": truth}
        env = SchedulingEnvironment(task)
        shared = _starts(env, truth)
        if sum(1 for v in shared.values() if v) < truth["count"]:
            continue  # no solution
        if sum(len(v) for v in shared.values()) > 8:
            continue  # scarce, so a guess rarely lands
        # The trap: at face value the first start found is wrong, or none is
        # found, so the conventions decide the answer.
        naive = face_value_starts(task)
        first = next(((d, naive[d][0]) for d in truth["days"] if naive[d]), None)
        if first is not None and first[1] in shared[first[0]]:
            continue
        # Seeds take turns requiring each convention to matter, so a memory
        # that learned only one of them cannot pass the whole set.
        task["decided_by"] = deciding_conventions(task)
        if CONVENTIONS[seed % len(CONVENTIONS)] not in task["decided_by"]:
            continue
        return task
    raise RuntimeError(f"seed {seed}: no task met the constraints in {max_tries} tries")


def task_set(split: str) -> list[dict[str, Any]]:
    return [generate(seed) for seed in SPLITS[split]]
