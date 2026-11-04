"""Generate requests for the same meeting time across several days.

The second class uses the first class's calendars, tools, and conventions.
A request specifies two or three days and requires a common start time.
The reference calculation therefore intersects valid starts across days.

Retain tasks with a common solution, few usable starts, and a naive answer
that the checker rejects. Seeds require different conventions to affect the
answer, while separate ranges distinguish development, validation, and test.
These ranges do not overlap those of the first task class.
"""

from __future__ import annotations

import copy
import functools
import random
from typing import Any

from chapters.ch06.assistant.environment import DAY_NAMES, SchedulingEnvironment, to_hhmm
from chapters.ch06.assistant.generator import (CONVENTIONS, PEOPLE, TOPICS, USER, _starts, _world,
                                               face_value_starts, starts_misreading)

SPLITS = {"dev": range(301, 317), "validation": range(401, 417), "test": range(501, 517)}
# Who must be invited for a convention to be able to decide the answer.
NEEDS = {"timezone": ("omar", "ana", "dana"), "show_as": (), "buffer": ("dana",), "hours": ("ana", "dana")}
DAY_SETS = (("mon", "wed"), ("tue", "thu"), ("wed", "fri"), ("mon", "wed", "fri"), ("tue", "thu", "fri"))


def _request(rng: random.Random, convention: str) -> tuple[str, dict[str, Any]]:
    attendees = rng.sample([p for p in PEOPLE if p != USER], rng.choice((2, 2, 3)))
    if NEEDS[convention] and not set(NEEDS[convention]) & set(attendees):
        attendees[0] = rng.choice(NEEDS[convention])
    minutes = rng.choice((30, 45, 60))
    days = list(rng.choice(DAY_SETS))
    limit = rng.choice(("earliest", "latest_end"))
    bound = rng.choice(("12:30", "13:00", "14:00")) if limit == "earliest" else rng.choice(("12:00", "12:30", "13:00"))
    truth = {"kind": "booking", "attendees": [PEOPLE[p] for p in attendees], "minutes": minutes,
             "count": len(days), "days": days, limit: bound, "per_day_max": 1, "same_start": True}
    names = [PEOPLE[p] for p in attendees]
    who = ", ".join(names[:-1]) + f" and {names[-1]}"
    on = ", ".join(DAY_NAMES[d] for d in days[:-1]) + f" and {DAY_NAMES[days[-1]]}"
    when = f"starting no earlier than {bound}" if limit == "earliest" else f"finished by {bound}"
    text = (f"Book a {minutes}-minute {rng.choice(TOPICS)} with {who} at the same time on {on} "
            f"this week, {when}.")
    return text, truth


def common(per_day: dict[str, list[int]]) -> list[int]:
    """The starts that appear on every day, earliest first."""
    sets = [set(v) for v in per_day.values()]
    return sorted(set.intersection(*sets)) if sets else []


def solutions(task: dict[str, Any]) -> list[str]:
    """The common starts the checker would accept, earliest first."""
    return [to_hhmm(s) for s in common(_starts(SchedulingEnvironment(task), task["truth"]))]


def deciding_conventions(task: dict[str, Any]) -> list[str]:
    true = set(common(_starts(SchedulingEnvironment(task), task["truth"])))
    decided = []
    for convention in CONVENTIONS:
        found = common(starts_misreading(task, convention))
        if not found or found[0] not in true:
            decided.append(convention)
    return decided


def generate(seed: int) -> dict[str, Any]:
    """The task for one seed; the same seed always gives the same task."""
    return copy.deepcopy(_generate(seed))


@functools.lru_cache(maxsize=None)
def _generate(seed: int, max_tries: int = 6000) -> dict[str, Any]:
    rng = random.Random(seed)
    for _ in range(max_tries):
        world = _world(rng)
        text, truth = _request(rng, CONVENTIONS[seed % len(CONVENTIONS)])
        task = {"id": f"T{seed:03d}", "seed": seed, "class": "same_time", "text": text,
                "world": world, "truth": truth}
        shared = common(_starts(SchedulingEnvironment(task), truth))
        if not shared or len(shared) > 4:
            continue  # a common start exists, and few do
        naive = common(face_value_starts(task))
        if naive and naive[0] in shared:
            continue  # at face value, the first answer must be wrong
        task["decided_by"] = deciding_conventions(task)
        if CONVENTIONS[seed % len(CONVENTIONS)] not in task["decided_by"]:
            continue
        return task
    raise RuntimeError(f"seed {seed}: no task met the constraints in {max_tries} tries")


def task_set(split: str) -> list[dict[str, Any]]:
    return [generate(seed) for seed in SPLITS[split]]
