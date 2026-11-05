"""D2: the returns desk world (DOMAINS.md), its hand-written procedure, its audit, and the online loop's
mechanics with scripted replies (no API key)."""

from __future__ import annotations

import json
import re

import pytest

from examples.refunds import study
from examples.refunds.agent import REFERENCE, REGULAR, practice, refund_decision
from examples.refunds.world import (
    DIAGNOSES,
    ReturnsWorld,
    conventions_that_matter,
    outcome,
    rules_that_matter,
    task_by_id,
    task_set,
)
from metacog import Recording
from metacog.testing import RespondingModel, call, text

# The procedure as written before the holiday policy: it never looks for the holiday note.
STALE = REGULAR


@pytest.mark.parametrize("split", ["dev", "validation", "test", "stream", "gate"])
async def test_the_hand_written_procedure_meets_the_policy_on_every_return(split):
    tasks = task_set(split)
    assert [(await practice(REFERENCE, t["id"]))[0] for t in tasks] == [1.0] * len(tasks)


def test_returns_cover_every_outcome_and_convention():
    regular = [t for split in ("dev", "validation", "test") for t in task_set(split)]
    assert {t["expected"][0] for t in regular} == {"refund", "credit", "decline"}
    seen = set().union(*(conventions_that_matter(t) for t in regular))
    assert seen == {"damaged", "final_sale", "beauty", "restocking", "accessories", "window", "members", "late"}
    assert all("holiday" in conventions_that_matter(t) for t in task_set("stream"))


def test_the_audit_names_the_broken_rule_without_the_answer():
    for task in task_set("dev")[:8]:
        right = outcome(task)
        for rule in rules_that_matter(task):
            wrong = outcome(task, ignore=rule)
            world = ReturnsWorld(task)
            if wrong[0] == "decline":
                world.decline_return(task["return"]["return_id"], "no")
            else:
                world.issue_refund(task["return"]["return_id"], wrong[1], "card" if wrong[0] == "refund" else
                                   "store_credit")
            verdict = world.check()
            # Two mistakes can give the same action (a decline from counting the window from the order date, or
            # from refusing the credit band); the audit names one that explains what was done.
            explains = [r for r in DIAGNOSES if outcome(task, ignore=r) == wrong]
            assert not verdict["success"] and any(DIAGNOSES[r] in verdict["reason"] for r in explains), (
                task["id"], rule)
            if right[0] != "decline" and abs(right[1] - wrong[1]) > 0.01:  # never an amount it didn't give
                assert f"{right[1]:.2f}" not in verdict["reason"]


async def test_after_the_holiday_policy_a_stale_procedure_is_rejected_for_it():
    before = [(await practice(STALE, t["id"]))[0] for t in task_set("test")]
    after = [await practice(STALE, t["id"]) for t in task_set("gate")]
    assert before == [1.0] * len(before)
    assert all(reward == 0.0 and "regular windows and fees to a holiday delivery" in reason
               for reward, reason, _ in after)


def test_the_audit_wants_exactly_one_action_on_the_return():
    task = task_set("dev")[0]
    world = ReturnsWorld(task)
    assert world.check()["reason"] == "nothing was done with the return"
    world.decline_return(task["return"]["return_id"], "no")
    world.decline_return(task["return"]["return_id"], "still no")
    assert world.check()["reason"] == "2 actions were taken on one return"
    assert ReturnsWorld(task).read_return("RET-999")["error"].startswith("no return")


def test_the_procedure_contract_comes_from_the_stub():
    contract = refund_decision.contract
    assert contract.signature == "def refund_decision(api, return_id: str) -> dict:"
    assert contract.returns.startswith('{"action": "refund" | "credit" | "decline", "amount_usd": float}')
    assert all(f"`api.{name}(" in contract.api for name in ("read_return", "read_order", "read_customer",
                                                             "read_policy"))


# ------------------------------------------------------------------ the online loop, scripted

def returned(number_text: str) -> dict:
    """The task a return id such as RET-007 or RET-712H belongs to."""
    found = re.fullmatch(r"RET-(\d+)(H?)", number_text)
    return task_by_id(f"{'H' if found.group(2) else 'R'}{found.group(1)}")


def act(decision: tuple[str, float], return_id: str):
    kind, amount = decision
    if kind == "decline":
        return call("decline_return", {"return_id": return_id, "reason": "under the policy"})
    return call("issue_refund", {"return_id": return_id, "amount_usd": amount,
                                 "method": "card" if kind == "refund" else "store_credit"})


def desk(request: dict):
    """A scripted model for the whole loop: the writer always writes the hand-written procedure, the judge
    is always right, and the agent calls the procedure when offered (doing what it says) or else reads the
    return, order, customer and policy and decides correctly."""
    system = request["messages"][0]["content"]
    if system.startswith("You are writing a Python function"):
        return text(f"```python\n{REFERENCE}```")
    if system.startswith("You judge whether an assistant handled a request correctly"):
        shown = request["messages"][1]["content"]
        case = returned(re.search(r"return (RET-\d+H?)", shown).group(1))
        acted = re.search(r"called (issue_refund|decline_return)\((\{.*?\})\)", shown)
        args = json.loads(acted.group(2)) if acted else {}
        kind = ("decline" if acted.group(1) == "decline_return" else
                "refund" if args["method"] == "card" else "credit") if acted else None
        right = outcome(case)
        ok = kind == right[0] and (kind == "decline" or abs(args["amount_usd"] - right[1]) <= 0.01)
        return text(json.dumps({"accepted": ok, "reason": "matches the policy" if ok else "does not"}))
    return_id = re.search(r"return (RET-\d+H?)", request["messages"][1]["content"]).group(1)
    case = returned(return_id)
    tools = [m for m in request["messages"] if m["role"] == "tool"]
    names = {t["function"]["name"] for t in request.get("tools") or []}
    if tools and json.loads(tools[-1]["content"]).get("status") in ("issued", "declined"):
        return text("Done.")
    if "refund_decision" in names:
        if not tools:
            return call("refund_decision", {"return_id": return_id})
        decision = json.loads(tools[-1]["content"])
        return act((decision["action"], decision["amount_usd"]), return_id)
    reads = [("read_return", {"return_id": return_id}), ("read_order", {"order_id": case["order"]["order_id"]}),
             ("read_customer", {"customer_id": case["customer"]["customer_id"]}), ("read_policy", {})]
    if len(tools) < len(reads):
        return call(*reads[len(tools)])
    return act(outcome(case), return_id)


@pytest.mark.parametrize("arm", ["checker", "reviewers", "judge"])
async def test_the_online_loop_recovers_from_the_holiday_policy(arm, tmp_path):
    args = study.parse(["--phases", "online", "--arms", arm, "--regular", "4", "--holiday", "30", "--review-rate",
                        "0.5", "--review-delay", "3", "--approve-after", "2", "--output", str(tmp_path),
                        "--recording", str(tmp_path / "unused.jsonl")])
    with RespondingModel(desk).active():
        result = await study.run_arm(arm, args, Recording(tmp_path / "unused.jsonl"))
    # The stale procedure serves holiday returns wrongly until it's quarantined; the agent then decides
    # alone (rightly), the learner writes the procedure again, and it's trusted once more.
    assert result["wrong"] == result["wrong_after_change"] > 0
    assert result["first_holiday_run"] == 5 < result["quarantined_at"] < result["relearned_at"]
    assert result["relearned_at"] <= result["trusted_again_at"] < 34
    assert [state for _, state, _ in result["procedures"]][:2] == ["quarantined", "trusted"]
    if arm == "checker":
        assert result["quarantined_at"] == 6  # judged at once, two wrong runs are enough
    if arm == "judge":
        assert result["judge_agreed"] == result["judge_runs"] == 34
