"""Connect development experience, function contracts, and evaluation.

Collect the assistant's observable tool interactions at medium and high
effort without memory. Store each request and its simulated checker verdict
as an episode, then recall that collection for learning. Each mechanism
produces a function matching the task class's name and return contract.

The act helper tries a function on a development request and returns feedback.
The match signal compares its outputs with the reference on evaluation cases.
The classes share the calendar environment while supplying different
contracts and booking rules; after four colleagues add rules to their notes,
the checker also honors each rule.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from chapters.ch05.personal.encoder import embed
from chapters.ch06.assistant import generator, generator_drift, generator_rules, generator_same_time
from chapters.ch06.assistant.environment import DAY_NAMES, SchedulingEnvironment
from chapters.ch06.assistant.harness import (MEMORIES, STARTING_MEMORY, STARTING_SAME_TIME, ReadOnlyCalendar,
                                             environment, run_task)
from helix.memory.base import MemoryContext, MemoryEntry, MemoryQuery, Scope, ScopeKey
from helix.memory.code_skills import skill_tool
from helix.memory.episodic import EpisodicMemory
from helix.memory.learning_mechanisms import Contract, Episode
from helix.signals.behavior_match import BehaviorMatchSignal

# The reasoning settings whose runs count as experience.
PRACTICE = ("medium", "high")
USER_SCOPE = ScopeKey(Scope.USER, "maya")
API = ("`api.read_calendar(person, day)` returns exactly what the assistant's read_calendar tool returns; "
       "person \"\" means the user, Maya Rivera, and a full name means that person.")
SIGNATURE = ("def {name}(api, attendees: list[str], days: list[str], minutes: int,\n"
             "{pad}earliest: str = \"\", latest_end: str = \"\") -> dict:")


def _signature(name: str) -> str:
    return SIGNATURE.format(name=name, pad=" " * (len(name) + 5))


def _book_each_day(env: SchedulingEnvironment, task: dict[str, Any], starts: Any) -> str | None:
    """First class: the earliest start on as many listed days as the request asks for."""
    if not isinstance(starts, dict):
        return "a mapping of day to starts"
    if any(not isinstance(day, str) or not isinstance(values, list)
           or any(not isinstance(value, str) for value in values)
           for day, values in starts.items()):
        return "a mapping of day to lists of time strings"
    booked = 0
    for day in task["truth"]["days"]:
        found = starts.get(day) or []
        if found and booked < task["truth"]["count"]:
            env.book_time("Meeting", day, str(found[0]), task["truth"]["minutes"], task["truth"]["attendees"])
            booked += 1
    return None


def _book_each_day_checked(env: SchedulingEnvironment, task: dict[str, Any], starts: Any) -> str | None:
    """After the colleagues' rules: the first class's booking, refusing starts that are not lists of times."""
    if isinstance(starts, dict) and any(not isinstance(day, str) or not isinstance(values, list)
                                        or any(not isinstance(value, str) for value in values)
                                        for day, values in starts.items()):
        return "a mapping of day to lists of time strings"
    return _book_each_day(env, task, starts)


def _book_same_time(env: SchedulingEnvironment, task: dict[str, Any], starts: Any) -> str | None:
    """Second class: the earliest common start, on every listed day."""
    if not isinstance(starts, list):
        return "a list of starts"
    if any(not isinstance(value, str) for value in starts):
        return "a list of time strings"
    if starts:
        for day in task["truth"]["days"]:
            env.book_time("Meeting", day, str(starts[0]), task["truth"]["minutes"], task["truth"]["attendees"])
    return None


@dataclass(frozen=True)
class TaskClass:
    name: str
    prefix: str  # task ids begin with it
    entry: dict[str, Any]  # the starting memory
    contract: Contract
    generate: Callable[[int], dict[str, Any]]
    task_set: Callable[[str], list[dict[str, Any]]]
    book: Callable[[SchedulingEnvironment, dict[str, Any], Any], str | None]


CONTRACT = Contract(
    name=STARTING_MEMORY["name"], signature=_signature(STARTING_MEMORY["name"]),
    description=STARTING_MEMORY["description"], api=API,
    returns=("Return {\"starts\": {day: [\"HH:MM\", ...]}} with each day keyed as the calendar keys it, starts "
             "on quarter hours and earliest first, and days with no start left out. Times in the arguments and "
             "the result are on the user's clock."),
)
SAME_TIME_CONTRACT = Contract(
    name=STARTING_SAME_TIME["name"], signature=_signature(STARTING_SAME_TIME["name"]),
    description=STARTING_SAME_TIME["description"], api=API,
    returns=("Return {\"starts\": [\"HH:MM\", ...]} with the starts that work on every listed day, on quarter "
             "hours and earliest first. Times in the arguments and the result are on the user's clock."),
)
CLASSES = {
    "multi_attendee": TaskClass("multi_attendee", "S", STARTING_MEMORY, CONTRACT, generator.generate,
                                generator.task_set, _book_each_day),
    "same_time": TaskClass("same_time", "T", STARTING_SAME_TIME, SAME_TIME_CONTRACT,
                           generator_same_time.generate, generator_same_time.task_set, _book_same_time),
    # The first class after Dana's note changes, served by the online loop.
    "drift": TaskClass("drift", "D", STARTING_MEMORY, CONTRACT, generator_drift.generate,
                       generator_drift.task_set, _book_each_day),
    # The first class after four colleagues add rules to their calendar notes, served by listing 6.3.
    "team_rules": TaskClass("team_rules", generator_rules.PREFIX, STARTING_MEMORY, CONTRACT,
                            generator_rules.generate, generator_rules.task_set, _book_each_day_checked),
}


def class_of(task_or_id: dict[str, Any] | str) -> TaskClass:
    task_id = task_or_id if isinstance(task_or_id, str) else task_or_id["id"]
    return next(c for c in CLASSES.values() if task_id.startswith(c.prefix))


def starting_source(task_class: str = "multi_attendee") -> str:
    return (MEMORIES / CLASSES[task_class].entry["file"]).read_text(encoding="utf-8")


def arguments(task: dict[str, Any], *, day_names: bool = False) -> dict[str, Any]:
    """The request's stated parameters, the inputs every learned function takes.

    With day_names, days are written the way the agent often writes them
    ("Wednesday" rather than "wed"), which the calendar tool also accepts.
    """
    truth = task["truth"]
    days = [DAY_NAMES[d] for d in truth["days"]] if day_names else list(truth["days"])
    return {"attendees": list(truth["attendees"]), "days": days, "minutes": truth["minutes"],
            "earliest": truth.get("earliest", ""), "latest_end": truth.get("latest_end", "")}


async def run_on_case(source: str, task: dict[str, Any], *, day_names: bool = False) -> Any:
    """A function's result on one request's calendars, under the guard."""
    contract = class_of(task).contract
    env = environment(task)
    try:
        tool = skill_tool(contract.name, contract.description, source, bind={"api": ReadOnlyCalendar(env)})
        return await tool.call(**arguments(task, day_names=day_names))
    except Exception as exc:  # refused by the guard or not callable with the contract's arguments
        return {"error": f"{contract.name} could not be called: {type(exc).__name__}: {exc}"}


def calendars_read(task: dict[str, Any]) -> str:
    """The calendars a request touches, as the tool shows them."""
    env = environment(task)
    pages = [env.read_calendar(person, day) for person in ["", *task["truth"]["attendees"]]
             for day in task["truth"]["days"]]
    return "\n".join(json.dumps(page) for page in pages)


async def act(source: str, task_id: str, *, day_names: bool = False) -> tuple[float, str, str]:
    """Use a function on one request as the agent with memory would, and return the verdict."""
    task_class = class_of(task_id)
    task = task_class.generate(int(task_id[1:]))
    env = environment(task)
    result = await run_on_case(source, task, day_names=day_names)
    if isinstance(result, dict) and "error" in result:
        return 0.0, f"the function crashed when run: {result['error']}", calendars_read(task)
    if day_names and isinstance(result, dict) and isinstance(result.get("starts"), dict):
        # Keys may come back as the names they were asked with; book by the day they name.
        named = {}
        for key, value in result["starts"].items():
            try:
                named[env.day(str(key))] = value
            except ValueError:
                named[key] = value
        result = {**result, "starts": named}
    wrong_shape = task_class.book(env, task, result.get("starts") if isinstance(result, dict) else None)
    if wrong_shape:
        return 0.0, f"the function returned {result!r}, not {{\"starts\": {wrong_shape}}}", calendars_read(task)
    verdict = env.check()
    if verdict["success"]:
        return 1.0, "accepted", calendars_read(task)
    return 0.0, f"booking its earliest start was rejected by the user: {verdict['reason']}", calendars_read(task)


async def gather_experience(*, efforts: tuple[str, ...] = PRACTICE, task_class: str = "multi_attendee"
                            ) -> list[Episode]:
    """The agent's runs without memory on a class's development tasks, from the recording."""
    episodes = []
    for task in CLASSES[task_class].task_set("dev"):
        for effort in efforts:
            success, reason, trajectory = await run_task(task, effort)
            episodes.append(Episode.from_trajectory(task["id"], trajectory, success, reason))
    return episodes


async def remember(episodes: list[Episode], path: str | Path = ":memory:") -> EpisodicMemory:
    """Write the episodes to episodic memory, in the order they happened."""
    tier = EpisodicMemory(path, embedding_fn=embed)
    for number, episode in enumerate(episodes):
        await tier.write(MemoryEntry(id=f"{episode.task_id}-{number:03d}", scope_key=USER_SCOPE,
                                     content={**episode.to_content(), "order": number},
                                     metadata={"created_by": "agent", "kind": "episode"}))
    return tier


async def recall_experience(tier: EpisodicMemory) -> list[Episode]:
    """Every episode, as the mechanisms read them, in the order they happened."""
    entries = await tier.read(MemoryQuery(text="", k=10_000), MemoryContext(user_id="maya"))
    entries.sort(key=lambda e: e.content.get("order", 0))
    return [Episode.from_content(e.content) for e in entries]


def match_signal(task_class: str = "multi_attendee") -> BehaviorMatchSignal:
    return BehaviorMatchSignal(starting_source(task_class), run_on_case, case_id=lambda task: task["id"])


def behavior_cases(split: str, task_class: str = "multi_attendee") -> list[dict[str, Any]]:
    return CLASSES[task_class].task_set(split)
