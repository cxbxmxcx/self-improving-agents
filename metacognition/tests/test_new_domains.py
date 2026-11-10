"""D6 and D7, the domains the frozen method is tested on: their worlds, and the text a separate agent wrote for them."""

from __future__ import annotations

import string

import pytest

from examples.billing.world import BillingWorld, banks as billing_banks, credit_for, task_set
from examples.decisions.loop import split_near, split_test
from examples.decisions.wordings import load
from examples.triage.world import OWNERS, TriageWorld, banks as triage_banks, severity_of, team_of


def account(**changes: object) -> dict:
    base = {"account_id": "A-1", "plan": "team", "term": "monthly", "seats": 10, "billing_day": 1, "promo_percent": 0,
            "change": {"type": "cancellation", "date": "2026-03-11", "seats_removed": 10}}
    return {**base, **changes}


@pytest.mark.parametrize(("changes", "credit"), [
    ({}, 154.83),  # 240 over 31 days, 20 days after the change
    ({"plan": "enterprise"}, 372.90),  # 680, with three days fewer
    ({"term": "annual"}, 129.03),  # ten twelfths
    ({"promo_percent": 10}, 139.35),
    ({"change": {"type": "seat_reduction", "date": "2026-03-11", "seats_removed": 3}}, 46.45),
    ({"change": {"type": "cancellation", "date": "2026-03-02", "seats_removed": 10}}, 240.00),  # the first two days
    ({"billing_day": 15, "change": {"type": "cancellation", "date": "2026-03-20", "seats_removed": 10}}, 193.54),  # 25 of 31
])
def test_a_credit_follows_the_billing_policy(changes, credit):
    assert credit_for(account(**changes)) == credit


def test_the_billing_checker_accepts_the_credit_and_names_the_convention_missed():
    task = task_set("dev")[0]
    world = BillingWorld(task)
    world.issue_credit(task["account"]["account_id"], task["expected_usd"])
    assert world.check()["success"]
    wrong = BillingWorld(task)
    wrong.issue_credit(task["account"]["account_id"], task["expected_usd"] + 50)
    assert not wrong.check()["success"]


@pytest.mark.parametrize(("facts", "plan", "severity"), [
    ({"environment": "production", "scope": "all", "workaround": "no", "data_loss": "no", "security": "no"}, "standard", "P1"),
    ({"environment": "production", "scope": "some", "workaround": "no", "data_loss": "no", "security": "no"}, "standard", "P2"),
    ({"environment": "production", "scope": "some", "workaround": "no", "data_loss": "no", "security": "no"}, "enterprise", "P1"),
    ({"environment": "production", "scope": "all", "workaround": "yes", "data_loss": "no", "security": "no"}, "standard", "P3"),
    ({"environment": "production", "scope": "one", "workaround": "yes", "data_loss": "no", "security": "no"}, "enterprise", "P2"),
    ({"environment": "staging", "scope": "all", "workaround": "no", "data_loss": "no", "security": "no"}, "standard", "P4"),
    ({"environment": "staging", "scope": "all", "workaround": "no", "data_loss": "no", "security": "no"}, "enterprise", "P3"),
    ({"environment": "staging", "scope": "one", "workaround": "no", "data_loss": "yes", "security": "no"}, "standard", "P4"),
    ({"environment": "staging", "scope": "one", "workaround": "yes", "data_loss": "no", "security": "yes"}, "standard", "P1"),
    ({"environment": "production", "scope": "one", "workaround": "yes", "data_loss": "yes", "security": "no"}, "standard", "P1"),
])
def test_severity_follows_the_triage_policy(facts, plan, severity):
    assert severity_of(facts, plan) == severity


def test_a_security_exposure_goes_to_secops_whatever_the_area():
    assert team_of({"security": "yes", "area": "billing"}) == "secops"
    assert team_of({"security": "no", "area": "billing"}) == "payments"


def fields(text: str) -> set[str]:
    return {name for _, name, _, _ in string.Formatter().parse(text) if name}


def strings(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    children = value.values() if isinstance(value, dict) else value
    return [text for child in children for text in strings(child)]


def test_the_billing_text_is_complete_and_uses_only_its_placeholders():
    words = billing_banks()
    for audience in ("dev", "search", "gate", "held"):
        for style in ("casual", "shorthand", "distractor", "buried"):
            assert len(words["requests"][audience][style]) == 2
            for text in words["requests"][audience][style]:
                assert fields(text) <= {"account", "number", "other"}
                assert ("number" in fields(text)) == (style == "shorthand") and ("account" in fields(text)) == (style != "shorthand")
                assert ("other" in fields(text)) == (style == "distractor")
    assert [len(words["near_miss"][a]) for a in ("dev", "search", "gate", "held")] == [6, 10, 10, 10]
    assert all(fields(text) == {"account"} for text in strings(words["near_miss"]) + strings(words["out_of_scope"]))


def test_the_triage_text_is_complete_and_uses_only_its_placeholders():
    words = triage_banks()
    for group, extra in (("triage_requests", set()), ("assign_requests", {"owner"})):
        for audience in ("dev", "search", "gate", "held"):
            for style in ("casual", "shorthand", "distractor", "buried"):
                assert len(words[group][audience][style]) == 2
                for text in words[group][audience][style]:
                    assert fields(text) <= {"ticket", "number", "other"} | extra and extra <= fields(text)
                    assert ("number" in fields(text)) == (style == "shorthand") and ("ticket" in fields(text)) == (style != "shorthand")
                    assert ("other" in fields(text)) == (style == "distractor")
    assert [len(words["near_miss"][a]) for a in ("dev", "search", "gate", "held")] == [6, 10, 10, 10]
    assert all(fields(text) <= {"ticket", "owner"} and "ticket" in fields(text) for text in strings(words["near_miss"]))
    for audience in ("dev", "gate", "held"):
        thread = words["thread"][audience]
        assert len(thread["openers"]) == len(thread["acks"]) == 3
        for voice in ("customer", "engineer"):
            assert all(len(texts) == 2 for values in thread[voice].values() for texts in values.values())
            assert sorted(thread[voice]) == sorted(["environment", "scope", "workaround", "data_loss", "security", "area"])
            assert all(not fields(text) for text in strings(thread[voice]))


@pytest.mark.parametrize("words", [billing_banks, triage_banks])
def test_no_sentence_is_shared_between_audiences(words):
    bank = words()
    seen: dict[str, str] = {}
    for key in ("requests", "triage_requests", "assign_requests", "near_miss"):
        for audience, content in bank.get(key, {}).items():
            for text in strings(content):
                assert seen.setdefault(text, audience) == audience, text


@pytest.mark.parametrize("domain", ["d6", "d7"])
def test_the_loop_splits_it_as_it_does_the_other_domains(domain):
    gate, held = split_test(domain)
    assert {t["id"].split("~")[0] for t in gate}.isdisjoint({t["id"].split("~")[0] for t in held})
    assert len(held) == 32 and len(load(domain, "stress2")) == 20 and len(load(domain, "dev+near")) == 80
    seen, _ = split_near(load(domain, "stress"))
    _, hidden = split_near(load(domain, "gate"))
    assert len(seen) == 10 and len(hidden) == 10


def test_a_triage_request_is_right_only_with_the_policy_answer():
    task = next(t for t in load("d7", "test") if t["kind"] == "triage")
    ticket = task["ticket"]
    right = TriageWorld(task)
    right.triage_ticket(ticket["ticket_id"], ticket["severity"], ticket["team"])
    assert right.check()["success"]
    wrong = TriageWorld(task)
    wrong.triage_ticket(ticket["ticket_id"], "P4" if ticket["severity"] != "P4" else "P1", ticket["team"])
    assert not wrong.check()["success"]
    assert ticket["thread"][0]["from"] == "customer" and ticket["thread"][-1]["from"] == "engineer"


def test_an_assignment_is_right_only_for_that_person():
    task = next(t for t in load("d7", "test") if t["kind"] == "assign" and t["style"] == "casual")
    right = TriageWorld(task)
    right.assign_ticket(task["ticket"]["ticket_id"], task["owner"])
    assert right.check()["success"]
    other = next(o for o in OWNERS if o != task["owner"])
    wrong = TriageWorld(task)
    wrong.assign_ticket(task["ticket"]["ticket_id"], other)
    assert not wrong.check()["success"] and task["owner"] in task["text"]
