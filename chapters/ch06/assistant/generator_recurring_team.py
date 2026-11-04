"""Superseded 2026-10-31: a team whose weeks repeat, memory as recalled answers.

Kept as the evidence behind recorded/reasoning_floor.recurring_team.jsonl. On 16
tasks gpt-5.4-mini solved 15 at low without memory, so remembering the team's
windows saved little reasoning. The original summary follows.

Seeded scheduling tasks for a team whose weeks repeat.

Maya Rivera works with the same six colleagues every week, and their
calendars follow a standing pattern: recurring meetings fill most of each
day, and the whole team keeps one shared window free each weekday. Each
seed is a different week. The pattern stays, a few one-off events land on
top of it (sometimes inside a window), and a request asks Maya's assistant
to meet some colleagues under day and time limits.

Without memory, the assistant must intersect everyone's free time from the
calendars, which takes reasoning. With the team's windows remembered, it
only has to check one candidate, as a child recalls 7 x 8 instead of
counting. The windows are never stated by the tools; they can be learned
only from experience, or written down by someone who knows the team.

A task is kept only when it has a solution, every solution falls inside a
team window, and the user's own first free slot within the limits clashes
with an attendee, so reading only the user's calendar fails.
"""

from __future__ import annotations

import random
from datetime import date, timedelta
from typing import Any

from chapters.ch06.assistant.environment_plain_calendars import DAY_NAMES, DAYS, SchedulingEnvironment, to_hhmm, to_minutes

HOURS = ("09:00", "17:00")
USER = "maya"
PEOPLE = {
    "maya": "Maya Rivera", "sam": "Sam Ortiz", "dana": "Dana Park", "priya": "Priya Shah",
    "leo": "Leo Chen", "ana": "Ana Silva", "omar": "Omar Haddad",
}
# The team's standing shared free time, one window a weekday. The bootstrap
# skill states these; learners must recover them from experience.
WINDOWS = {"mon": ("15:00", "16:30"), "tue": ("09:30", "11:00"), "wed": ("14:00", "15:15"),
           "thu": ("10:00", "11:30"), "fri": ("13:30", "14:45")}
TOPICS = ("project sync", "design review", "planning session", "budget check-in", "launch review",
          "hiring debrief")
# The task sets of section 7 of the plan: learning, selection, and a held-out test.
SPLITS = {"dev": range(1, 17), "validation": range(101, 117), "test": range(201, 217)}
TEAM_SEED = 2026  # the standing pattern is the same in every week
FIRST_MONDAY = date(2026, 10, 19)
STEP = 15  # starts fall on quarter hours


def _fill(rng: random.Random, start: int, end: int, *, gap_chance: float) -> list[tuple[int, int]]:
    """Busy blocks covering [start, end), leaving occasional short gaps."""
    blocks, cursor = [], start
    while cursor < end:
        if rng.random() < gap_chance:
            cursor += rng.choice((30, 45))
            continue
        length = min(rng.choice((30, 45, 60, 60, 90)), end - cursor)
        blocks.append((cursor, cursor + length))
        cursor += length
    return [(s, e) for s, e in blocks if e - s >= 15]


def standing_pattern() -> dict[str, dict[str, list[tuple[int, int]]]]:
    """Each person's recurring busy blocks by weekday, the same every week.

    Colleagues are busy outside the team window except for short gaps that
    often line up for all but one person, the near misses that make
    intersecting calendars slow; the user keeps lighter focus blocks, so her own free
    time looks plentiful and misleads a run that reads only her calendar.
    """
    rng = random.Random(TEAM_SEED)
    open_, close = (to_minutes(h) for h in HOURS)
    pattern: dict[str, dict[str, list[tuple[int, int]]]] = {}
    for pid in PEOPLE:
        pattern[pid] = {}
        for day in DAYS:
            w0, w1 = (to_minutes(h) for h in WINDOWS[day])
            chance = 0.6 if pid == USER else 0.45
            pattern[pid][day] = _fill(rng, open_, w0, gap_chance=chance) + _fill(rng, w1, close, gap_chance=chance)
    return pattern


def _week(seed: int) -> dict[str, str]:
    monday = FIRST_MONDAY + timedelta(weeks=seed % 52)
    return {d: (monday + timedelta(days=i)).isoformat() for i, d in enumerate(DAYS)}


def _world(rng: random.Random, seed: int) -> dict[str, Any]:
    open_, close = (to_minutes(h) for h in HOURS)
    events, number = [], 1

    def add(pid: str, day: str, s: int, e: int, title: str) -> None:
        nonlocal number
        events.append({"id": f"E{number:02d}", "owner": pid, "title": title, "day": day,
                       "start": to_hhmm(s), "minutes": e - s, "attendees": [pid]})
        number += 1

    for pid, days in standing_pattern().items():
        for day, blocks in days.items():
            for s, e in blocks:
                add(pid, day, s, e, "Focus time" if pid == USER else "Busy")
    # This week's one-off events, placed anywhere, sometimes inside a window.
    for pid in PEOPLE:
        for _ in range(rng.randint(1, 3)):
            day = rng.choice(DAYS)
            minutes = rng.choice((30, 45, 60))
            s = rng.randrange(open_, close - minutes + 1, STEP)
            add(pid, day, s, s + minutes, "One-off" if pid == USER else "Busy")
    return {"user": USER, "week": _week(seed), "hours": list(HOURS), "people": dict(PEOPLE), "events": events}


def _request(rng: random.Random, count: int) -> tuple[str, dict[str, Any]]:
    attendees = rng.sample([p for p in PEOPLE if p != USER], rng.choice((3, 3, 4)))
    minutes = rng.choice((30, 45, 60))
    length = rng.choice((3, 4))
    first = rng.randrange(0, len(DAYS) - length + 1)
    days = list(DAYS[first:first + length])
    limit = rng.choice(("earliest", "latest_end"))
    bound = rng.choice(("12:30", "13:00", "13:30")) if limit == "earliest" else rng.choice(("12:00", "12:30"))
    truth = {"kind": "booking", "attendees": [PEOPLE[p] for p in attendees], "minutes": minutes,
             "count": count, "days": days, limit: bound}
    if count == 2:
        truth["per_day_max"] = 1
    names = [PEOPLE[p] for p in attendees]
    who = ", ".join(names[:-1]) + f" and {names[-1]}"
    span = f"{DAY_NAMES[days[0]]} to {DAY_NAMES[days[-1]]}"
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
    return {day: [s for s in range(lo, hi - truth["minutes"] + 1, STEP)
                  if all(env.clash(p, day, s, s + truth["minutes"]) is None for p in people)]
            for day in truth["days"]}


def in_window(day: str, start: int, minutes: int) -> bool:
    w0, w1 = (to_minutes(h) for h in WINDOWS[day])
    return w0 <= start and start + minutes <= w1


def solutions(task: dict[str, Any]) -> dict[str, list[str]]:
    """The starts, by day, that the checker would accept for one meeting."""
    env = SchedulingEnvironment(task)
    people = [USER, *(env.resolve(n) for n in task["truth"]["attendees"])]
    return {d: [to_hhmm(s) for s in v] for d, v in _starts(env, task["truth"], people).items() if v}


def generate(seed: int, *, max_tries: int = 2000) -> dict[str, Any]:
    """The task for one seed; the same seed always gives the same week and request."""
    rng = random.Random(seed)
    for _ in range(max_tries):
        world = _world(rng, seed)
        # Every fourth week asks for two meetings, so each set has some;
        # left to chance, the filters below would rarely keep one.
        text, truth = _request(rng, 2 if seed % 4 == 0 else 1)
        task = {"id": f"S{seed:03d}", "seed": seed, "class": "multi_attendee", "text": text,
                "world": world, "truth": truth}
        env = SchedulingEnvironment(task)
        attendees = [env.resolve(n) for n in truth["attendees"]]
        shared = _starts(env, truth, [USER, *attendees])
        if sum(1 for v in shared.values() if v) < truth["count"]:
            continue  # no solution
        if any(not in_window(d, s, truth["minutes"]) for d, v in shared.items() for s in v):
            continue  # the remembered windows must be the whole answer
        # The trap: the user's first free start inside the limits clashes with
        # an attendee, so reading only the user's calendar books the wrong time.
        own = _starts(env, truth, [USER])
        first_own = next(((d, own[d][0]) for d in truth["days"] if own[d]), None)
        if first_own is None or first_own[1] in shared[first_own[0]]:
            continue
        return task
    raise RuntimeError(f"seed {seed}: no task met the constraints in {max_tries} tries")


def task_set(split: str) -> list[dict[str, Any]]:
    return [generate(seed) for seed in SPLITS[split]]
