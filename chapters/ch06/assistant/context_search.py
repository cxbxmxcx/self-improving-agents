"""Searching the procedural-memory artifact, the way chapters 2 to 4 searched prompts.

The artifact under search is the function the assistant recalls from
procedural memory. Chapter 4's GEPA proposes rewrites of it; this module
supplies what GEPA needs to search code instead of prose: an evaluator,
a validation hook, and mutation and crossover instructions.

The evaluator is the measure of good from chapter 3 applied to this task.
It runs a candidate on the development requests the way the assistant with
memory would, books its earliest start, and asks the checker. Each request
runs twice: with days as the contract writes them ("wed") and as the agent
often writes them ("Wednesday"), and it counts as accepted only when both
bookings are. Most of its score is the share accepted; the rest is the
share of returned starts the checker would accept, which gives the search a
gradient before any request passes.

Its feedback names the rejected requests with the user's reasons, and for
the first of them shows the calendars, the call, and what the function
returned. The proposer can see what it got wrong without being handed the
answer.
"""

from __future__ import annotations

import json
from typing import Any

from chapters.ch06.assistant.environment import SchedulingEnvironment, to_minutes
from chapters.ch06.assistant.generator import task_set
from chapters.ch06.assistant.learning import CONTRACT, act, arguments, calendars_read, run_on_case
from helix.memory.code_skills import UnsafeSkill, check_source
from helix.memory.learning_mechanisms import contract_text
from helix.signal import GapMeasurement

SHOWN = 6  # rejected requests named in the feedback
# Accepted requests decide the score; the share of acceptable starts breaks
# ties, so a candidate that fixes part of the problem ranks above one that
# fixes nothing.
ACCEPT_WEIGHT, START_WEIGHT = 0.8, 0.2

MUTATION_PROMPT = """You are improving a Python function by reflective mutation in a
population-based search. A calendar assistant calls the function instead of
working the answer out itself. You will see the current function, feedback
from running it on the assistant's development requests (which bookings the
user rejected and why, and the calendars of one rejected request), and
sometimes a sibling function from the same population.

Produce an improved function that fixes what the feedback shows, keeps what
already works, and keeps this contract exactly:

{contract}

Output only the function source."""

CROSSOVER_PROMPT = """You are combining two Python functions into one offspring. A calendar
assistant calls the function instead of working the answer out itself.
Take what each parent does well, drop what either does poorly, and keep
this contract exactly:

{contract}

Output only the function source."""


def mutation_prompt() -> str:
    return MUTATION_PROMPT.format(contract=contract_text(CONTRACT))


def crossover_prompt() -> str:
    return CROSSOVER_PROMPT.format(contract=contract_text(CONTRACT))


def validate(source: str) -> str | None:
    """Why the guard refuses a proposal, or None when it would run."""
    try:
        check_source(source, CONTRACT.name)
    except UnsafeSkill as exc:
        return str(exc)
    return None


async def acceptance(source: str, split: str) -> tuple[int, int, list[tuple[dict[str, Any], str]]]:
    """How many of a split's requests the user accepts, and the rejections.

    A request counts only when it is accepted with the days written either
    way the assistant writes them, since the agent, not the search, decides
    how it calls the function.
    """
    tasks = task_set(split)
    rejected = []
    accepted = 0
    for task in tasks:
        reward, reason, _ = await act(source, task["id"])
        if reward >= 1.0:
            reward, reason, _ = await act(source, task["id"], day_names=True)
            reason = f"with the days written as full names, {reason}"
        if reward >= 1.0:
            accepted += 1
        else:
            rejected.append((task, reason))
    return accepted, len(tasks), rejected


async def start_quality(source: str, task: dict[str, Any]) -> tuple[int, int]:
    """How many of the starts a candidate returns the checker would accept."""
    result = await run_on_case(source, task)
    starts = result.get("starts") if isinstance(result, dict) else None
    if not isinstance(starts, dict):
        return 0, 0
    env = SchedulingEnvironment(task)
    truth = task["truth"]
    attendees = [env.resolve(name) for name in truth["attendees"]]
    lo = to_minutes(truth["earliest"]) if truth.get("earliest") else 0
    hi = to_minutes(truth["latest_end"]) if truth.get("latest_end") else 24 * 60
    good = total = 0
    for day, values in starts.items():
        for value in values if isinstance(values, list) else []:
            total += 1
            try:
                start = to_minutes(str(value))
            except ValueError:
                continue
            if (day in truth["days"] and lo <= start and start + truth["minutes"] <= hi
                    and env.acceptable(day, start, attendees, truth["minutes"]) is None):
                good += 1
    return good, total


async def evaluator(candidate: Any, questions: Any = None) -> GapMeasurement:
    """Chapter 3's measure of good on this task: accepted development requests,
    with the share of acceptable starts as partial credit."""
    source = candidate.content
    accepted, total, rejected = await acceptance(source, "dev")
    good = returned = 0
    for task in task_set("dev"):
        g, r = await start_quality(source, task)
        good, returned = good + g, returned + r
    start_share = good / returned if returned else 0.0
    score = ACCEPT_WEIGHT * accepted / total + START_WEIGHT * start_share
    if rejected:
        lines = [f"- {task['text']} {reason}" for task, reason in rejected[:SHOWN]]
        first_task, first_reason = rejected[0]
        # Show the proposer its own output on that request, the trace chapter 3's reflection reads.
        day_names = first_reason.startswith("with the days written as full names")
        call = arguments(first_task, day_names=day_names)
        output = await run_on_case(source, first_task, day_names=day_names)
        feedback = (f"{accepted} of {total} development requests accepted; {good} of {returned} returned starts "
                    "acceptable. Rejected:\n" + "\n".join(lines)
                    + f"\n\nCalendars read for the first rejected request:\n{calendars_read(first_task)}"
                    + f"\n\nThe function was called with {json.dumps(call)}\nand returned {json.dumps(output)}")
    else:
        feedback = f"all {total} development requests accepted; {good} of {returned} returned starts acceptable"
    return GapMeasurement(score=score, signal_id="calendar_acceptance", signal_version=4,
                          triggered=accepted < total, raw_value=float(accepted), feedback=feedback,
                          metadata={"accepted": accepted, "total": total, "starts_good": good,
                                    "starts_returned": returned})


class NoQuestions:
    """GEPA asks an evaluation source for questions; the evaluator brings its own."""

    label = "development requests, scored by the evaluator"

    async def get_questions(self, n: int | None = None) -> list:
        return []


