"""Seeded tasks after a change in conditions: Dana's note now asks for half an hour.

The online loop needs the world to move under a memory that was right.
These tasks are the first class's, with one difference: Dana Park, the
external client, rewords her note to "Please leave half an hour between any
meeting with me and my other meetings" and the checker enforces 30 minutes.
A reasoning agent can interpret the new wording, while a remembered function
that expects a numeric minute count finds no number and applies no buffer.

A task is kept only when Dana is invited, a solution exists, few starts
work, and reading her note as no buffer at all gives a wrong first answer,
so a stale memory shows up as a rejected booking.
"""

from __future__ import annotations

import copy
import functools
import random
from typing import Any

from chapters.ch06.assistant.environment import SchedulingEnvironment, to_hhmm
from chapters.ch06.assistant.generator import _request, _starts, _world

NOTE = "External client. Please leave half an hour between any meeting with me and my other meetings."
BUFFER = 30
# The stream the online loop serves, and the held-back cases its gate checks.
SPLITS = {"stream": range(701, 721), "gate": range(801, 807)}


def changed(world: dict[str, Any], *, buffer: int = BUFFER) -> dict[str, Any]:
    world = copy.deepcopy(world)
    world["profiles"]["dana"].update(note=NOTE, buffer=buffer)
    return world


def solutions(task: dict[str, Any]) -> dict[str, list[str]]:
    shared = _starts(SchedulingEnvironment(task), task["truth"])
    return {d: [to_hhmm(s) for s in v] for d, v in shared.items() if v}


def generate(seed: int) -> dict[str, Any]:
    return copy.deepcopy(_generate(seed))


@functools.lru_cache(maxsize=None)
def _generate(seed: int, max_tries: int = 6000) -> dict[str, Any]:
    rng = random.Random(seed)
    for _ in range(max_tries):
        world = changed(_world(rng))
        text, truth = _request(rng, 2 if seed % 4 == 0 else 1)
        if "Dana Park" not in truth["attendees"]:
            continue
        task = {"id": f"D{seed:03d}", "seed": seed, "class": "drift", "text": text, "world": world, "truth": truth}
        shared = _starts(SchedulingEnvironment(task), truth)
        if sum(1 for v in shared.values() if v) < truth["count"] or sum(len(v) for v in shared.values()) > 8:
            continue
        # Read as no buffer at all, the first start found must be wrong.
        stale = {**task, "world": changed(world, buffer=0)}
        found = _starts(SchedulingEnvironment(stale), truth)
        first = next(((d, found[d][0]) for d in truth["days"] if found[d]), None)
        if first is not None and first[1] in shared[first[0]]:
            continue
        return task
    raise RuntimeError(f"seed {seed}: no task met the constraints in {max_tries} tries")


def task_set(split: str) -> list[dict[str, Any]]:
    return [generate(seed) for seed in SPLITS[split]]
