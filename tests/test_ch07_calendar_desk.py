"""Chapter 7's calendar desk: the worded requests, and a hand-written program that books with no language model.

Nothing here calls a model. Jev is scripted from each task's own truth, so the tests check the code path: reading the
request, calling Chapter 6's procedure, booking, and the checker's verdict.
"""

from __future__ import annotations

import string
from collections import Counter
from types import SimpleNamespace
from typing import Any

import pytest

from chapters.ch06.assistant.environment import (
    DAY_NAMES,
    SchedulingEnvironment,
    to_hhmm,
)
from chapters.ch06.assistant.generator import _starts
from chapters.ch07.assistant import requests
from chapters.ch07.assistant.components import (
    hand_written_component,
    hand_written_program,
    memories,
)
from chapters.ch07.assistant.desk import INTENT, check, contract, desk, world_for
from helix.deciders import Answer, DecisionUsage
from helix.memory.mechanism_runtime import check_mechanism, check_runnable
from helix.metacognition.decide import check_program, serve

PROGRAM = hand_written_program()
SETS = ("experience", "search", "gate", "held_out")


class Scripted:
    """Answers as a careful reader would, from the task's truth and what kind of request it is."""

    name = "scripted"

    def __init__(self, task: dict[str, Any]) -> None:
        self.task, self.truth = task, task["truth"]

    async def ask_with_usage(self, state: Any, questions: dict[str, Any]) -> tuple[dict[str, Answer], DecisionUsage]:
        answers = {}
        for name in questions:
            if name == "books":
                answers[name] = Answer("probability", 0.95 if self.task["expect"] == "act" else 0.05)
            elif name == "several":
                answers[name] = Answer("probability", 0.95 if self.truth["count"] > 1 else 0.05)
            elif name == "limit":
                answers[name] = Answer("choice", "earliest" if self.truth.get("earliest") else "latest_end", 0.9)
            else:
                answers[name] = Answer("probability", 0.95 if name.removeprefix("on_") in self.truth["days"] else 0.05)
        return answers, DecisionUsage(calls=1, input_tokens=100, dollars=100 * 42 / 1e9, seconds=0.1)


def placeholders(text: str) -> list[str]:
    return [name for _, name, _, _ in string.Formatter().parse(text) if name]


def strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    return [text for child in (value.values() if isinstance(value, dict) else value) for text in strings(child)]


def test_the_hand_written_component_passes_the_guards_and_has_all_three_routes():
    check_program(PROGRAM)
    check_mechanism(hand_written_component())
    check_runnable(hand_written_component())
    assert [(m["route"], m["read_only"]) for m in memories()] == [("program", False), ("none", False), ("none", True)]


def test_the_contract_names_the_job():
    text = contract().description
    assert "book_time" in text and "find_shared_starts" in text and "read-only" in text


def test_a_verdict_on_a_near_miss_is_about_intent_and_one_on_real_work_is_the_checkers():
    near = next(r for r in requests.load("experience") if r["expect"] == "handoff")
    work = next(r for r in requests.load("experience") if r["expect"] == "act")
    left_alone = check(SimpleNamespace(context=world_for(near)), None)
    assert (left_alone.accepted, left_alone.source) == (True, INTENT)
    acted_on = world_for(near)
    truth = near["truth"]
    day, starts = next((d, s) for d, s in _starts(SchedulingEnvironment(near), truth).items() if s)
    acted_on.book_time("Meeting", day, to_hhmm(starts[0]), truth["minutes"], truth["attendees"])
    booked = check(SimpleNamespace(context=acted_on), None)
    assert acted_on.booked and (booked.accepted, booked.source) == (False, INTENT)
    undone = check(SimpleNamespace(context=world_for(work)), None)
    assert (undone.accepted, undone.source) == (False, "verifier")


def test_the_wordings_are_complete_and_use_only_their_placeholders():
    words = requests.banks()
    wanted = {"casual": {"minutes", "topic", "who", "days", "limit"},
              "shorthand": {"minutes", "topic", "first_names", "short_days", "limit"},
              "distractor": {"minutes", "topic", "who", "days", "limit", "other_day"},
              "buried": {"minutes", "topic", "who", "days", "limit"}}
    for group in ("dev", "search", "gate", "held"):
        for style, needs in wanted.items():
            assert len(words["single"][group][style]) == 2
            for text in words["single"][group][style]:
                assert sorted(placeholders(text)) == sorted(needs), text
        assert len(words["double"][group]) == 2
        assert all(sorted(placeholders(t)) == sorted(wanted["casual"]) for t in words["double"][group])
        for kind in ("earliest", "latest_end"):
            assert len(words["limit"][group][kind]) == 2
            assert all(placeholders(t) == ["time"] for t in words["limit"][group][kind])
    assert [len(words["near_miss"][g]) for g in ("dev", "search", "gate", "held")] == [6, 10, 10, 10]
    for text in strings(words["near_miss"]):
        used = set(placeholders(text))
        assert used <= {"topic", "who", "first", "day", "minutes"} and used & {"topic", "who", "first"}, text
    assert all(set(placeholders(t)) <= {"first", "day"} for t in strings(words["out_of_scope"]))


def test_no_sentence_is_shared_between_groups_of_writers():
    words = requests.banks()
    seen: dict[str, str] = {}
    for key in ("single", "double", "limit", "near_miss"):
        for group, content in words[key].items():
            for text in strings(content):
                assert seen.setdefault(text, group) == group, text
    names = [*DAY_NAMES.values(), "Maya", "Ana", "Omar", "Priya", "Dana", "Leo", "Sam"]
    for text in strings(words):
        assert not any(ch.isdigit() for ch in text) and not any(name in text for name in names), text


def test_the_sets_keep_their_requests_apart():
    sets = {name: requests.load(name) for name in SETS}
    bases = {name: {r["base"] for r in rows} for name, rows in sets.items()}
    texts = {name: {r["text"] for r in rows} for name, rows in sets.items()}
    for a in SETS:
        for b in SETS:
            if a < b:
                assert not bases[a] & bases[b] and not texts[a] & texts[b]
    assert {name: len(rows) for name, rows in sets.items()} == {"experience": 80, "search": 30, "gate": 26, "held_out": 52}
    held = Counter((r["kind"], r["expect"]) for r in sets["held_out"])
    assert held == {("one meeting", "act"): 24, ("two meetings", "act"): 8, ("one meeting", "handoff"): 20}


def test_the_second_class_sets_keep_their_weeks_and_wordings_apart():
    sets = {name: requests.load(name) for name in ("second_experience", "second_search", "second_gate",
                                                     "second_held_out")}
    same_time = {name: [r for r in rows if r["kind"] == "same time" and r["expect"] == "act"]
                 for name, rows in sets.items()}
    assert {name: len(rows) for name, rows in same_time.items()} == {
        "second_experience": 32, "second_search": 16, "second_gate": 16, "second_held_out": 32}
    names = list(sets)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert not {r["base"] for r in sets[a]} & {r["base"] for r in sets[b]}
            assert not {r["text"] for r in same_time[a]} & {r["text"] for r in same_time[b]}
    for rows in same_time.values():
        for request in rows:
            assert all(person.split()[0] in request["text"] for person in request["truth"]["attendees"])
            assert "{" not in request["text"] and str(request["truth"]["minutes"]) in request["text"]


def test_every_request_carries_what_its_truth_asks_for():
    for name in SETS:
        for request in requests.load(name):
            if request["expect"] != "act":
                continue
            truth, text = request["truth"], request["text"]
            assert str(truth["minutes"]) in text and (truth.get("earliest") or truth["latest_end"]) in text
            assert all(person.split()[0] in text for person in truth["attendees"])
            assert "{" not in text and "}" not in text


@pytest.mark.parametrize("name", SETS)
async def test_the_hand_written_program_books_single_meetings_with_no_model(name):
    the_desk, outcomes = desk(), Counter()
    for request in requests.load(name):
        row = await serve(PROGRAM, request, the_desk, Scripted(request))
        if request["expect"] == "handoff":
            assert row.right and row.read_only, (request["id"], request["text"], row.reason)
        elif request["kind"] == "two meetings":
            assert not row.handled and row.route == "none", (request["id"], row.reason)
        else:
            outcomes[bool(row.handled and row.right)] += 1
            assert row.decider_calls == 2  # the router, then the program
            if not (row.handled and row.right):
                print(request["id"], request["text"], "|", row.reason)
    assert outcomes[True] >= 0.9 * sum(outcomes.values()), outcomes
