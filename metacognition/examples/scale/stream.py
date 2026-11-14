"""A long stream of calendar requests: its days, what each day runs under, and what is held back from it.

Conditions change the way they do in production. Dana's note alternates between two wordings, so a function that
reads one goes stale under the other. On some days most requests leave out the meeting's length, so the program
hands them off. On some days the calendar cannot be read for a stretch. A long quiet stretch has no change at all,
which is where a scan that grows with the stream would show.

Weeks come from Chapter 6's generators and are reused with a new reference each time, since a long stream needs
many requests and not many calendars. Each day also sets requests aside that are never served: that night's gate, and
what a search may measure on.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from chapters.ch06.assistant import generator, generator_drift

from examples.scale.scripted import TASKS

CACHE = Path(__file__).parent / "runs" / "weeks.json"
GENERATORS = {"ordinary": generator, "changed": generator_drift}
SERVED_WEEKS = 400  # per kind of note; a request takes the next one and a new reference
HELD_BACK_WEEKS = 60
# Six weeks for each note, by the ids Chapter 6's practice function rebuilds them from: a relearned function's gate.
# A generator names a week by a letter and its seed, so no week has to be built to name it.
FUNCTION_WEEKS = {"ordinary": [f"S{seed}" for seed in range(30_001, 30_007)],
                  "changed": [f"D{seed}" for seed in range(30_001, 30_007)]}


@dataclass(frozen=True)
class Day:
    number: int
    conditions: str  # "ordinary" or "changed": the note Dana has that day
    changed: bool = False  # the note is not the one of the day before
    garbled: bool = False  # most requests for one meeting leave out its length
    outage: bool = False  # the calendar cannot be read for a stretch of the day

    @property
    def eventful(self) -> bool:
        return self.changed or self.garbled or self.outage


def plan(days: int, *, change_every: int = 2, quiet: tuple[int, int] = (0, 0), garbled_every: int = 15,
         outage_every: int = 25) -> list[Day]:
    """The stream's calendar: the note changes every `change_every` days, except in the quiet stretch."""
    out, conditions, since_change = [], "ordinary", 0
    for number in range(1, days + 1):
        calm = quiet[0] <= number <= quiet[1]
        changed = False
        if number > 3 and not calm and since_change >= change_every:  # the first days let the search install
            conditions, changed, since_change = ("changed" if conditions == "ordinary" else "ordinary"), True, 0
        since_change += 1
        out.append(Day(number, conditions, changed=changed,
                       garbled=not calm and not changed and number > 3 and number % garbled_every == 0,
                       outage=not calm and not changed and number > 3 and number % outage_every == 7))
    return out


class Weeks:
    """Weeks for each note, generated once and kept on disk."""

    def __init__(self, path: Path = CACHE) -> None:
        self.path = path
        self.pool: dict[str, dict[str, Any]] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        self.fresh = 0

    def get(self, conditions: str, index: int) -> dict[str, Any]:
        key = f"{conditions}:{index}"
        if key not in self.pool:
            self.pool[key] = GENERATORS[conditions].generate(40_000 + index)
            self.fresh += 1
        return self.pool[key]

    def save(self) -> None:
        if self.fresh:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.pool), encoding="utf-8")
            self.fresh = 0


def request(week: dict[str, Any], reference: str, *, near: bool = False, garbled: bool = False,
            outage: bool = False) -> dict[str, Any]:
    """One request on a week. The week's calendars are shared, since a world copies what it changes."""
    truth = week["truth"]
    several = truth["count"] > 1
    if near:
        text = f"Is the meeting with {truth['attendees'][0]} already on the calendar? Do not book anything."
    elif garbled and not several:
        text = week["text"].replace(f"{truth['minutes']}-minute ", "", 1)
    else:
        text = week["text"]
    text = f"{text} (ref {reference})"
    task = {**week, "id": f"{week['id']}~{reference}", "base": week["id"], "text": text, "outage": outage,
            "style": "near_miss" if near else "plain", "kind": "two meetings" if several else "one meeting",
            "expect": "handoff" if near else "act"}
    TASKS[text] = task
    return task


def served(day: Day, per_day: int, weeks: Weeks, *, near_share: float = 0.25, garbled_share: float = 0.7,
           outage_run: int = 40) -> list[dict[str, Any]]:
    """The day's requests, in the order they arrive."""
    rng = random.Random(day.number)
    first = (day.number - 1) * per_day
    out = []
    for n in range(per_day):
        week = weeks.get(day.conditions, (first + n) % SERVED_WEEKS)
        out.append(request(week, f"{day.number}-{n}", near=rng.random() < near_share,
                           garbled=day.garbled and rng.random() < garbled_share,
                           outage=day.outage and per_day // 2 <= n < per_day // 2 + outage_run))
    return out


def held_back(day: Day, weeks: Weeks, *, acting: int = 12, near: int = 4, garbled_share: float = 0.7
              ) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Requests under the day's own conditions that are never served: tonight's gate, and a search's own cases."""
    rng = random.Random(10_000 + day.number)
    sets = []
    for role in ("gate", "search"):
        first = SERVED_WEEKS + (HELD_BACK_WEEKS // 2 if role == "search" else 0)
        rows = [request(weeks.get(day.conditions, first + (day.number * 7 + n) % (HELD_BACK_WEEKS // 2)),
                        f"{role}{day.number}-{n}", near=n >= acting,
                        garbled=day.garbled and rng.random() < garbled_share)
                for n in range(acting + near)]
        sets.append({row["id"]: row for row in rows})
    return sets[0], sets[1]


def forget(before_day: int) -> None:
    """Drop requests from days the store no longer keeps, so the script's own table stays bounded too."""
    for text in [text for text, task in TASKS.items() if _day_of(task) < before_day]:
        del TASKS[text]


def _day_of(task: dict[str, Any]) -> int:
    reference = task["id"].split("~", 1)[1]
    digits = "".join(ch for ch in reference.split("-")[0] if ch.isdigit())
    return int(digits or 0)
