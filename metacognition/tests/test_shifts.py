"""D5, the on-call desk: plans have answers, and the forms a plan's constraints take depend on who sees it."""

from __future__ import annotations

from collections import Counter

import pytest

from examples.decisions.loop import split_near, split_test
from examples.decisions.wordings import load
from examples.shifts.world import (
    DAYS,
    GATE_ONLY,
    HELD_OUT_ONLY,
    PEOPLE,
    SEEN,
    ShiftWorld,
    holds,
    solutions,
    task_set,
    worded,
)


def plans(split: str = "test") -> list[dict]:
    return [t for t in worded(split) if t["kind"] == "plan"]


def number(task: dict) -> int:
    return int(task["id"].split("~")[0][1:])


def test_every_plan_has_an_answer_and_few_others():
    for task in plans("dev") + plans("test"):
        assert task["answer"] in solutions(task["constraints"], task["leave"])
        assert 1 <= task["solutions"] <= 3 and len(task["constraints"]) >= 4


def test_the_checker_accepts_the_answer_and_says_what_a_wrong_rota_breaks():
    task = plans()[0]
    world = ShiftWorld(task)
    world.publish_rota(*[task["answer"][day] for day in DAYS])
    assert world.check() == {"success": True, "reason": "accepted"}
    wrong = next(dict(zip(DAYS, [p] * 5, strict=True)) for p in PEOPLE)
    world = ShiftWorld(task)
    world.publish_rota(*wrong.values())
    verdict = world.check()
    assert not verdict["success"] and "doesn't meet" in verdict["reason"]


def test_a_plan_must_be_published_once():
    task = plans()[0]
    world = ShiftWorld(task)
    assert world.check() == {"success": False, "reason": "nothing was changed"}
    for _ in range(2):
        world.publish_rota(*[task["answer"][day] for day in DAYS])
    assert not world.check()["success"]


def test_setting_each_day_and_then_publishing_is_still_one_plan():
    task = plans()[0]
    world = ShiftWorld(task)
    for day in DAYS:
        world.assign_shift(day, task["answer"][day])
    world.publish_rota(*[task["answer"][day] for day in DAYS])
    assert world.check()["success"]


def test_a_requested_change_never_breaks_a_standing_rule():
    for task in worded("dev") + worded("test"):
        if task["kind"] == "assign":
            assert sum(p == task["person"] for d, p in task["rota"].items() if d != task["day"]) < 2


def test_a_shift_change_is_right_only_for_that_person_and_day():
    task = next(t for t in worded("test") if t["kind"] == "assign" and t["style"] == "casual")
    world = ShiftWorld(task)
    world.assign_shift(task["day"], task["person"])
    assert world.check()["success"]
    world = ShiftWorld(task)
    world.assign_shift(task["day"], task["other"])
    assert not world.check()["success"]
    assert task["person"] in task["text"] and task["day"] in task["text"]


def test_the_search_sees_only_the_forms_used_in_development():
    forms = {c["form"] for t in plans("dev") + [p for p in plans() if number(p) in range(201, 205)] for c in t["constraints"]}
    assert forms <= set(SEEN) and len(forms) == len(SEEN)


def test_the_gate_and_the_held_out_requests_bring_forms_nobody_else_sees():
    gate = [c["form"] for t in plans() if number(t) in range(205, 209) for c in t["constraints"]]
    held = [c["form"] for t in plans() if number(t) >= 209 for c in t["constraints"]]
    assert set(gate) & set(GATE_ONLY) and not set(gate) & set(HELD_OUT_ONLY)
    assert set(held) & set(HELD_OUT_ONLY) and not set(held) & set(GATE_ONLY)
    for task in plans():
        if number(task) >= 205:
            required = GATE_ONLY if number(task) < 209 else HELD_OUT_ONLY
            assert sum(c["form"] in required for c in task["constraints"]) >= 2


def test_the_loop_splits_it_as_it_does_the_other_domains():
    gate, held = split_test("d5")
    assert {t["id"].split("~")[0] for t in gate}.isdisjoint({t["id"].split("~")[0] for t in held})
    assert len(held) == 32 and Counter(t["kind"] for t in held) == {"assign": 16, "plan": 16}
    seen, _ = split_near(load("d5", "stress"))
    _, hidden = split_near(load("d5", "gate"))
    assert len(seen) == 10 and len(hidden) == 10 and len(load("d5", "stress2")) == 20
    assert not {t["text"] for t in seen} & {t["text"] for t in hidden}
    assert len(load("d5", "dev+near")) == 64 + 4 + 12


def test_wordings_differ_between_who_sees_them():
    def texts(group: range) -> set[str]:
        return {t["text"] for t in worded("test") if t["kind"] == "assign" and number(t) in group}

    assert not texts(range(201, 205)) & texts(range(205, 209))
    assert len(task_set("test")) == 16


@pytest.mark.parametrize("form", ["off", "fixed", "same", "either", "skill", "not_both", "exactly_two"])
def test_each_form_can_say_no(form):
    rota = dict(zip(DAYS, ["Priya", "Omar", "Lena", "Sam", "Noor"], strict=True))
    wrong = {"off": {"a": "Priya", "d": "Monday"}, "fixed": {"a": "Omar", "d": "Monday"}, "same": {"d": "Monday", "e": "Tuesday"},
             "either": {"d": "Monday", "a": "Omar", "b": "Lena"}, "skill": {"d": "Thursday", "s": "databases"},
             "not_both": {"a": "Priya", "b": "Omar"}, "exactly_two": {"a": "Priya"}}[form]
    assert not holds({"form": form, **wrong}, rota)


def test_the_small_model_serves_and_the_strong_model_writes_the_code():
    from examples.decisions import continuous, llm, loop, mixed

    tier = loop.tier_for("d5", None, loop.parse(["--domain", "d5"]))
    assert tier.writer_model == tier.search.designer_model == llm.BUILDER == "claude-fable-5-2"
    assert llm.MODEL == "gpt-5.4-mini"
    for example in (continuous, mixed):
        args = example.parse([])
        assert args.writer_model == args.designer_model == llm.BUILDER
