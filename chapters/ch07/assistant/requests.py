"""Word Chapter 6's scheduling requests the way people write them, and split them by who may see them.

Chapter 6's requests come from one template, so a program could pass by matching that sentence. Here each request keeps
its week, its people and its checker, and gets new words in four styles: casual, shorthand, distractor and buried.
Near misses use the words of the work on the same weeks but ask for something else, so nothing should be booked.

The words come from banks.json, which a separate agent wrote from a brief that described only the assistant's job.
Four groups of writers keep the sets apart: development requests are experience, and the test requests divide into
those a search may read, those only the gate sees, and those held out for the final measurement.

`week` lays the same requests out as working days for the online listing. Monday is the experience set, Tuesday brings
weeks the assistant has not served, and from Wednesday the requests come from Chapter 6's weeks after Dana's note
changed.
"""

from __future__ import annotations

import json
import random
from functools import cache
from pathlib import Path
from typing import Any

from chapters.ch06.assistant import generator, generator_drift, generator_same_time
from chapters.ch06.assistant.environment import DAY_NAMES, DAYS

STYLES = ("casual", "shorthand", "distractor", "buried")
# Chapter 6's test seeds: the first four are the search's, the next four the gate's, and the rest are held out.
SEARCH_SEEDS = range(201, 205)
GATE_SEEDS = range(205, 209)
HELD_OUT_SEEDS = range(209, 217)
# Chapter 6's weeks after Dana's note changed: the requests of Wednesday and Thursday, and the cases held back for
# a gate that has to judge the changed world.
CHANGED_SEEDS = generator_drift.SPLITS["stream"]
CHANGED_GATE_SEEDS = generator_drift.SPLITS["gate"]
# Chapter 6's second class, the same meeting at the same time on several days: its development weeks are experience,
# and its test weeks divide into those a component may practice on, the gate's and the held-out ones.
SAME_TIME_DEV = range(301, 309)
SAME_TIME_GATE = range(501, 505)
SAME_TIME_SEARCH = range(505, 509)
SAME_TIME_HELD_OUT = range(509, 517)
SHORT = {"mon": "Mon", "tue": "Tue", "wed": "Wed", "thu": "Thu", "fri": "Fri"}


@cache
def banks() -> dict[str, Any]:
    return json.loads((Path(__file__).parent / "banks.json").read_text(encoding="utf-8"))


def audience(seed: int) -> str:
    """Whose words a request uses: development, search, gate or held-out writers."""
    if seed < 200 or seed in CHANGED_SEEDS or seed in SAME_TIME_DEV:
        return "dev"  # the user's own way of writing, on any day of the week
    if seed in GATE_SEEDS or seed in CHANGED_GATE_SEEDS or seed in SAME_TIME_GATE:
        return "gate"
    return "search" if seed in SEARCH_SEEDS or seed in SAME_TIME_SEARCH else "held"


def _joined(names: list[str]) -> str:
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def _span(days: list[str], names: dict[str, str]) -> str:
    return f"{names[days[0]]} to {names[days[-1]]}" if len(days) > 2 else f"{names[days[0]]} or {names[days[1]]}"


def fields(task: dict[str, Any]) -> dict[str, str]:
    """What a request's words are filled with, read from the task's own truth."""
    truth, group = task["truth"], audience(task["seed"])
    kind = "earliest" if truth.get("earliest") else "latest_end"
    phrases = banks()["limit"][group][kind]
    other = next(day for day in (*DAYS[::-1], *DAYS) if day not in truth["days"])
    return {"minutes": str(truth["minutes"]), "topic": task["topic"], "who": _joined(truth["attendees"]),
            "first_names": _joined([name.split()[0] for name in truth["attendees"]]),
            "days": _span(truth["days"], DAY_NAMES), "short_days": _span(truth["days"], SHORT),
            "each_day": _joined([DAY_NAMES[day] for day in truth["days"]]),
            "short_each_day": _joined([SHORT[day] for day in truth["days"]]),
            "limit": phrases[task["seed"] % len(phrases)].format(time=truth[kind]), "other_day": DAY_NAMES[other],
            "first": truth["attendees"][0].split()[0], "day": DAY_NAMES[truth["days"][0]]}


def _base(seed: int) -> dict[str, Any]:
    """Chapter 6's task for a seed, with the topic its own sentence named."""
    if seed in CHANGED_SEEDS or seed in CHANGED_GATE_SEEDS:
        task = generator_drift.generate(seed)
    elif 300 < seed < 600:
        task = generator_same_time.generate(seed)
    else:
        task = generator.generate(seed)
    topic = next(t for t in generator.TOPICS if t in task["text"])
    several = "same time" if task["truth"].get("same_start") else "two meetings"
    return {**task, "base": task["id"], "kind": several if task["truth"]["count"] > 1 else "one meeting",
            "topic": topic}


def worded(seed: int) -> list[dict[str, Any]]:
    """One request in four styles. A request for two meetings keeps one sentence shape per style."""
    base = _base(seed)
    words, group, filled = banks(), audience(seed), fields(base)
    out = []
    for n, style in enumerate(STYLES):
        if base["kind"] == "same time":
            choices = words["same_time"][group][style]
            text = choices[seed % len(choices)].format(**filled)
        elif base["truth"]["count"] > 1:
            choices = words["double"][group]
            text = choices[(seed + n) % len(choices)].format(**filled)
        else:
            choices = words["single"][group][style]
            text = choices[seed % len(choices)].format(**filled)
        out.append({**base, "id": f"{base['id']}~{style}", "style": style, "expect": "act", "text": text})
    return out


def near_misses(seeds: list[int], group: str, tag: str) -> list[dict[str, Any]]:
    """Each near miss on the weeks of two requests: the words of the work, with nothing to book."""
    out = []
    for seed in seeds:
        base = _base(seed)
        filled = fields(base)
        out += [{**base, "id": f"{base['id']}~{tag}{n}", "style": "near_miss", "expect": "handoff",
                 "text": template.format(**filled)} for n, template in enumerate(banks()["near_miss"][group])]
    return out


def out_of_scope(seeds: list[int], split: str) -> list[dict[str, Any]]:
    bases = [_base(seed) for seed in seeds]
    return [{**bases[n % len(bases)], "id": f"{bases[n % len(bases)]['id']}~out{n}", "style": "out_of_scope",
             "expect": "handoff", "text": template.format(**fields(bases[n % len(bases)]))}
            for n, template in enumerate(banks()["out_of_scope"][split])]


def _all(seeds: Any) -> list[dict[str, Any]]:
    return [request for seed in seeds for request in worded(seed)]


@cache
def _sets() -> dict[str, tuple[dict[str, Any], ...]]:
    dev = list(generator.SPLITS["dev"])
    return {
        # What the agent serves before any search: the development weeks, with near misses mixed in.
        "experience": (*_all(dev), *near_misses(dev[:2], "dev", "neardev"), *out_of_scope(dev[:2], "dev")),
        # What a search may read and practice on.
        "search": (*_all(SEARCH_SEEDS), *near_misses([201], "search", "near"), *out_of_scope([201, 202], "test")),
        # What only the gate sees.
        "gate": (*_all(GATE_SEEDS), *near_misses([205], "gate", "gate")),
        # What only the gate sees once Dana's note has changed.
        "changed_gate": (*_all(CHANGED_GATE_SEEDS), *near_misses([801], "gate", "gate")),
        # What nothing sees until the final measurement.
        "held_out": (*_all(HELD_OUT_SEEDS), *near_misses([209, 210], "held", "fresh")),
    }


def _taking_turns(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The same requests with each kind taking its turn, so the start of a set holds some of every kind."""
    kinds: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        kinds.setdefault("near miss" if row["expect"] == "handoff" else row["kind"], []).append(row)
    mixed = []
    while any(kinds.values()):
        mixed += [waiting.pop(0) for waiting in kinds.values() if waiting]
    return mixed


@cache
def _second_class() -> dict[str, tuple[dict[str, Any], ...]]:
    """The sets of listing 7.5, where requests for the same time on several days join the first class's."""
    first = _sets()
    sets = {
        # A day on which both kinds of request arrive.
        "second_experience": [*_all(SAME_TIME_DEV), *_a_day(generator.SPLITS["validation"], 301, "second")],
        "second_search": [*first["search"], *_all(SAME_TIME_SEARCH)],
        "second_gate": [*first["gate"], *_all(SAME_TIME_GATE)],
        "second_held_out": [*first["held_out"], *_all(SAME_TIME_HELD_OUT)],
    }
    return {name: tuple(_taking_turns(rows)) for name, rows in sets.items()}


def load(name: str) -> list[dict[str, Any]]:
    """The requests of one set: experience, search, gate, changed_gate or held_out, or one of listing 7.5's."""
    sets = _second_class() if name.startswith("second_") else _sets()
    return [dict(request) for request in sets[name]]


def _a_day(seeds: Any, near: int, tag: str) -> list[dict[str, Any]]:
    """Each week once, the four styles taking turns, with the near misses of one of those weeks."""
    seeds = list(seeds)
    return [*[worded(seed)[n % len(STYLES)] for n, seed in enumerate(seeds)], *near_misses([near], "dev", tag)]


def week() -> dict[str, list[dict[str, Any]]]:
    """Four working days of requests, in the order they arrive."""
    changed = list(CHANGED_SEEDS)
    days = {"Monday": load("experience"),
            "Tuesday": _a_day(generator.SPLITS["validation"], 101, "tue"),
            "Wednesday": _a_day(changed[:12], 701, "wed"),
            "Thursday": _a_day(changed[12:], 713, "thu")}
    for n, requests in enumerate(days.values()):
        random.Random(11 + n).shuffle(requests)
    return days
