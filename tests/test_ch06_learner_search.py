"""Chapter 6, section 6.4: the new team's rules, the learner monitor, and the world listing 6.3 serves."""

from __future__ import annotations

import asyncio
from collections import Counter

import pytest

from chapters.ch06.assistant import generator_rules
from chapters.ch06.assistant.environment import SchedulingEnvironment
from chapters.ch06.assistant.harness import MEMORIES, environment
from chapters.ch06.assistant.learning import act, class_of, starting_source
from chapters.ch06.assistant.online import LearnerMonitor

TEAM_RULES = (MEMORIES / "find_shared_starts.team_rules.py").read_text(encoding="utf-8")
SPLITS = ("first_contact", "stream", "gate", "test")


def accepted(source: str, split: str) -> list[float]:
    return [asyncio.run(act(source, t["id"]))[0] for t in generator_rules.task_set(split)]


@pytest.mark.parametrize("split", SPLITS)
def test_a_function_that_honors_every_rule_is_accepted_on_every_request(split):
    assert accepted(TEAM_RULES, split) == [1.0] * len(generator_rules.task_set(split))


@pytest.mark.parametrize("split", SPLITS)
def test_the_starting_memory_is_rejected_on_every_request_from_the_new_team(split):
    assert accepted(starting_source(), split) == [0.0] * len(generator_rules.task_set(split))


@pytest.mark.parametrize("split", ("stream", "gate", "test"))
def test_every_colleagues_rule_decides_some_request_in_each_split(split):
    decided = Counter(p for t in generator_rules.task_set(split) for p in t["decided_by"])
    assert set(decided) == set(generator_rules.ORDER)


def test_the_new_teams_requests_run_in_the_world_that_checks_each_rule():
    task = generator_rules.task_set("first_contact")[0]
    assert task["id"] == "R017" and class_of(task).name == "team_rules"
    assert type(environment(task)) is generator_rules.RulesEnvironment
    assert type(environment({**task, "id": "S017"})) is SchedulingEnvironment


def test_a_booking_across_priyas_lunch_is_refused_with_the_reason_she_would_give():
    task = next(t for t in generator_rules.task_set("stream") if "Priya Shah" in t["truth"]["attendees"])
    env = environment(task)
    reason = env.rule_refusal("priya", "tue", 12 * 60 - env.shift("priya") - 15, 30)
    assert reason and "free for lunch" in reason


def test_starts_that_are_not_lists_of_times_are_refused_before_booking():
    source = '''def find_shared_starts(api, attendees: list[str], days: list[str], minutes: int,
                       earliest: str = "", latest_end: str = "") -> dict:
    return {"starts": {days[0]: "10:00"}}
'''
    reward, reason, _ = asyncio.run(act(source, "R001"))
    assert reward == 0.0 and "lists of time strings" in reason


def test_the_learner_monitor_fires_on_refusals_in_a_row_and_a_pass_clears_it():
    monitor = LearnerMonitor(limit=2)
    assert monitor.observe(False) is False
    assert monitor.observe(True) is False
    assert monitor.observe(False) is False
    assert monitor.observe(False) is True
    monitor.reset()
    assert monitor.refusals == 0
