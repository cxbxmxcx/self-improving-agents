"""Superseded 2026-10-31: plain calendars (free intervals given, no conventions).

Kept to replay recorded/reasoning_floor.plain_calendars.jsonl, where the code
skill solved every task at reasoning none. The original summary follows.

Each seed draws a week of busy calendars for the user and the people they
work with, then a request to meet several of them under constraints: a
length, one or two meetings, a range of days, a time limit, and different
days for two meetings. The request's constraints are its truth, and the
checker in `environment.py` scores any booking against them.

A task is kept only when it calls for the procedure the chapter teaches.
It must have a solution, and the user's own first free slot that meets the
stated limits must clash with an attendee, so a run that reads only the
user's calendar fails. Solutions are scarce, so guessing rarely lands.

Every week is new, so no answer recurs for a memory to store. What can be
learned is the method, which chapter 6 keeps as a code skill.
"""

from __future__ import annotations

import random
from typing import Any

from chapters.ch06.assistant.environment_plain_calendars import DAY_NAMES, DAYS, SchedulingEnvironment, to_hhmm, to_minutes

WEEK = {"mon": "2026-10-19", "tue": "2026-10-20", "wed": "2026-10-21", "thu": "2026-10-22", "fri": "2026-10-23"}
HOURS = ("09:00", "17:00")
USER = "maya"
PEOPLE = {
    "maya": "Maya Rivera", "sam": "Sam Ortiz", "dana": "Dana Park", "priya": "Priya Shah",
    "leo": "Leo Chen", "ana": "Ana Silva", "omar": "Omar Haddad",
}
TOPICS = ("project sync", "design review", "planning session", "budget check-in", "launch review",
          "hiring debrief")
# The task sets of section 7 of the plan: learning, selection, and a held-out test.
SPLITS = {"dev": range(1, 17), "validation": range(101, 117), "test": range(201, 217)}
STEP = 15  # starts fall on quarter hours


def _busy_day(rng: random.Random, open_: int, close: int, density: int) -> list[tuple[int, int]]:
    """Non-overlapping busy blocks for one person on one day."""
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
    events, number = [], 1
    for pid in PEOPLE:
        # The user's calendar is lighter than everyone else's, so their own
        # free time looks plentiful and misleads a run that stops there.
        low, high = (1, 3) if pid == USER else (3, 5)
        for day in DAYS:
            for s, e in _busy_day(rng, open_, close, rng.randint(low, high)):
                events.append({"id": f"E{number:02d}", "owner": pid, "title": "Focus time" if pid == USER else "Busy",
                               "day": day, "start": to_hhmm(s), "minutes": e - s, "attendees": [pid]})
                number += 1
    return {"user": USER, "week": dict(WEEK), "hours": list(HOURS), "people": dict(PEOPLE), "events": events}


def _request(rng: random.Random) -> tuple[str, dict[str, Any]]:
    attendees = rng.sample([p for p in PEOPLE if p != USER], rng.choice((2, 2, 3)))
    minutes = rng.choice((30, 45, 60))
    count = rng.choice((1, 2))
    first = rng.randrange(0, 3)
    days = list(DAYS[first:first + rng.choice((2, 3))])
    limit = rng.choice(("earliest", "latest_end"))
    bound = rng.choice(("13:00", "13:30", "14:00")) if limit == "earliest" else rng.choice(("12:00", "12:30"))
    truth = {"kind": "booking", "attendees": [PEOPLE[p] for p in attendees], "minutes": minutes,
             "count": count, "days": days, limit: bound}
    if count == 2:
        truth["per_day_max"] = 1
    names = [PEOPLE[p] for p in attendees]
    who = names[0] if len(names) == 1 else ", ".join(names[:-1]) + f" and {names[-1]}"
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


def _starts(env: SchedulingEnvironment, truth: dict[str, Any], people: list[str]) -> dict[str, list[int]]:
    """Every start, by day, that meets the stated limits and suits everyone in `people`."""
    lo = max(env.open, to_minutes(truth["earliest"])) if truth.get("earliest") else env.open
    hi = min(env.close, to_minutes(truth["latest_end"])) if truth.get("latest_end") else env.close
    found: dict[str, list[int]] = {}
    for day in truth["days"]:
        found[day] = [s for s in range(lo, hi - truth["minutes"] + 1, STEP)
                      if all(env.clash(p, day, s, s + truth["minutes"]) is None for p in people)]
    return found


def solutions(task: dict[str, Any]) -> dict[str, list[str]]:
    """The starts, by day, that the checker would accept for one meeting."""
    env = SchedulingEnvironment(task)
    people = [USER, *(env.resolve(n) for n in task["truth"]["attendees"])]
    return {d: [to_hhmm(s) for s in v] for d, v in _starts(env, task["truth"], people).items() if v}


def generate(seed: int, *, max_tries: int = 500) -> dict[str, Any]:
    """The task for one seed; the same seed always gives the same task."""
    rng = random.Random(seed)
    for _ in range(max_tries):
        world = _world(rng)
        text, truth = _request(rng)
        task = {"id": f"S{seed:03d}", "seed": seed, "class": "multi_attendee", "text": text,
                "world": world, "truth": truth}
        env = SchedulingEnvironment(task)
        attendees = [env.resolve(n) for n in truth["attendees"]]
        shared = _starts(env, truth, [USER, *attendees])
        days_with = [d for d, v in shared.items() if v]
        if len(days_with) < truth["count"]:
            continue  # no solution
        # Scarce: few acceptable starts in all, so a guess rarely lands.
        if sum(len(v) for v in shared.values()) > 6:
            continue
        # The trap: the user's first free start inside the limits clashes with
        # an attendee, so reading only the user's calendar books the wrong time.
        own = _starts(env, truth, [USER])
        first_own = next(((d, v[0]) for d in truth["days"] for v in [own[d]] if v), None)
        if first_own is None or first_own[1] in shared[first_own[0]]:
            continue
        return task
    raise RuntimeError(f"seed {seed}: no task met the constraints in {max_tries} tries")


def task_set(split: str) -> list[dict[str, Any]]:
    return [generate(seed) for seed in SPLITS[split]]
