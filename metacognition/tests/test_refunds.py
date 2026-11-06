"""D2: the returns desk world (DOMAINS.md), its hand-written procedure, its audit, and the online loop's
mechanics with scripted replies (no API key)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from examples.refunds import study
from examples.refunds.agent import (
    CONTRACTS,
    REFERENCE,
    REGULAR,
    in_tool_terms,
    method_of,
    practice,
    refund_decision,
    refund_decision_in_tool_terms,
    refund_decision_to_act_on,
)
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
from metacog.offline import Floors
from metacog.testing import RespondingModel, call, text


@pytest.mark.parametrize("contract", list(CONTRACTS))
@pytest.mark.parametrize("split", ["dev", "validation", "test", "stream", "gate"])
async def test_the_hand_written_procedure_meets_the_policy_on_every_return(split, contract):
    reference = CONTRACTS[contract][1]
    tasks = task_set(split)
    assert [(await practice(reference, t["id"]))[0] for t in tasks] == [1.0] * len(tasks)


def test_the_second_pass_procedure_is_the_same_code_in_the_tools_terms():
    for words in (REFERENCE, REGULAR):
        tool = in_tool_terms(words)
        changed = [(a, b) for a, b in zip(words.splitlines(), tool.splitlines(), strict=True) if a != b]
        assert changed and all(a.strip().startswith("return {\"action\"") and b.strip().startswith(
            "return {\"method\"") for a, b in changed)
    assert [method_of(d) for d in ({"action": "refund"}, {"action": "credit"}, {"method": "store_credit"},
                                   {"method": "decline"}, {"action": "card"}, None)] == [
        "card", "store_credit", "store_credit", "decline", None, None]
    contract = refund_decision_in_tool_terms.contract
    assert contract.name == "refund_decision"
    assert contract.signature == "def refund_decision(api, return_id: str) -> dict:"
    assert contract.returns.startswith('{"method": "card" | "store_credit" | "decline", "amount_usd": float}')


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


@pytest.mark.parametrize("contract", list(CONTRACTS))
async def test_after_the_holiday_policy_a_stale_procedure_is_rejected_for_it(contract):
    stale = CONTRACTS[contract][2]
    before = [(await practice(stale, t["id"]))[0] for t in task_set("test")]
    after = [await practice(stale, t["id"]) for t in task_set("gate")]
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
        asked = json.dumps(request["messages"])
        return text(f"```python\n{in_tool_terms(REFERENCE) if 'method' in asked else REFERENCE}```")
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
        kind = {"card": "refund", "store_credit": "credit", "decline": "decline"}[method_of(decision)]
        return act((kind, decision["amount_usd"]), return_id)
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


@pytest.mark.parametrize("arm", ["checker", "judge"])
async def test_the_second_pass_measures_its_settings_and_gates_on_labeled_returns(arm, tmp_path):
    args = study.parse(["--contract", "tool", "--controller", "measured", "--gate", "cases", "--arms", arm,
                        "--tasks", "2", "--efforts", "none", "high", "--regular", "4", "--holiday", "30",
                        "--review-rate", "0.5", "--review-delay", "3", "--approve-after", "2",
                        "--output", str(tmp_path), "--recording", str(tmp_path / "unused.jsonl")])
    unused = Recording(tmp_path / "unused.jsonl")
    with RespondingModel(desk).active():
        await study.phase_floors(args, unused)
        await study.phase_memory(args, unused)
        result = await study.run_arm(arm, args, unused)
    assert (tmp_path / "floors_memory_tool.json").exists()
    # The scripted desk is right at every setting, so the measured floors put every run at `none`.
    assert all(r["effort"] == "none" for r in json.loads((tmp_path / f"online_{arm}.json").read_text("utf-8")))
    gated = [a["gate"] for a in result["attempts"] if a["gate"]]
    assert gated and all(g["total"] == 6 for g in gated)  # the labeled holiday returns, never replays
    assert result["relearned_at"] <= result["trusted_again_at"] < 34


STUDY =Path(__file__).parents[1] / "examples" / "refunds" / "recorded" / "study.jsonl"


async def test_the_recorded_d2_study_replays(tmp_path):
    """D2 (DOMAINS.md): the returns desk study, live once on 2026-11-05, replayed from its recording."""
    if not STUDY.exists():
        pytest.skip("no recorded D2 study")
    await study.main(study.parse(["--recording", str(STUDY), "--output", str(tmp_path)]))
    efforts = ("none", "low", "medium", "high")
    without = Floors.load(tmp_path / "floors_no_memory.json").sweeps["no_memory"]
    assert [without.cells[e].solved for e in efforts] == [6, 15, 16, 16]
    with_procedure = Floors.load(tmp_path / "floors_memory_words.json").sweeps["memory"]
    assert [with_procedure.cells[e].solved for e in efforts] == [13, 15, 16, 16]
    arms = {a["arm"]: a for a in json.loads((tmp_path / "online_summary.json").read_text(encoding="utf-8"))}
    measured = {name: (a["wrong"], a["quarantined_at"], a["trusted_again_at"], a["under_refunded_usd"])
                for name, a in arms.items()}
    assert measured == {"checker": (13, 12, 20, 31.95), "reviewers": (19, 22, None, 111.6),
                        "judge": (11, 8, 21, 9.0)}
    judge = arms["judge"]
    assert (judge["judge_agreed"], judge["judge_runs"], judge["judge_false_accepts"]) == (56, 60, 1)


async def test_the_recorded_d2_second_pass_replays(tmp_path):
    """D2's second pass (DOMAINS.md): the procedure answers in the tools' terms, the controller is measured, and
    the gate checks 6 labeled holiday returns. Live once on 2026-11-05; the no-procedure floors replay from the
    first pass."""
    if not STUDY.exists():
        pytest.skip("no recorded D2 study")
    await study.main(study.parse(["--recording", str(STUDY), "--output", str(tmp_path), "--contract", "tool",
                                  "--controller", "measured", "--gate", "cases"]))
    efforts = ("none", "low", "medium", "high")
    with_procedure = Floors.load(tmp_path / "floors_memory_tool.json").sweeps["memory"]
    assert [with_procedure.cells[e].solved for e in efforts] == [15, 16, 16, 16]
    arms = {a["arm"]: a for a in json.loads((tmp_path / "online_summary.json").read_text(encoding="utf-8"))}
    measured = {name: (a["wrong"], a["quarantined_at"], a["trusted_again_at"], a["over_refunded_usd"],
                       a["under_refunded_usd"]) for name, a in arms.items()}
    assert measured == {"checker": (3, 11, 15, 85.99, 281.94), "reviewers": (27, 33, None, 85.99, 310.29),
                        "judge": (4, 11, 20, 85.99, 281.94)}
    assert all([a["gate"]["accepted"] for a in arm["attempts"]] == [6] for arm in arms.values())
    judge = arms["judge"]
    assert (judge["judge_agreed"], judge["judge_runs"], judge["judge_false_accepts"]) == (57, 60, 0)
    # The reasoning settings came from the measured floors: none with the procedure, low without.
    rows = json.loads((tmp_path / "online_checker.json").read_text(encoding="utf-8"))
    assert {(bool(r["offered"]), r["effort"]) for r in rows} == {(True, "none"), (False, "low")}


def test_the_act_contract_differs_only_in_what_it_tells_the_agent():
    tool, act = refund_decision_in_tool_terms.contract, refund_decision_to_act_on.contract
    assert (act.name, act.signature, act.returns) == (tool.name, tool.signature, tool.returns)
    assert "act on its answer" in act.description and "act on its answer" not in tool.description


async def test_the_recorded_d2_follow_ups_replay(tmp_path):
    """D2's follow-ups (DOMAINS.md), live once on 2026-11-06: an offered procedure acted on, and a person's one
    rejection enough to quarantine."""
    if not STUDY.exists():
        pytest.skip("no recorded D2 study")
    measured = {}
    for contract in ("tool", "act"):
        out = tmp_path / contract
        await study.main(study.parse(["--recording", str(STUDY), "--output", str(out), "--phases", "memory",
                                      "--contract", contract, "--efforts", "none", "--memory-splits", "test",
                                      "stream", "--tasks", "60"]))
        runs = Floors.load(out / f"floors_memory_{contract}.json").sweeps["memory"].cells["none"].runs
        calls = sum(r.model_calls for r in runs) / len(runs)
        measured[contract] = (sum(r.solved for r in runs), len(runs), round(calls, 2))
    assert measured == {"tool": (74, 76, 4.17), "act": (76, 76, 3.0)}  # fewer reads, no overruled answers
    out = tmp_path / "person"
    await study.main(study.parse(["--recording", str(STUDY), "--output", str(out), "--arms", "reviewers",
                                  "--contract", "tool", "--controller", "measured", "--gate", "cases",
                                  "--person-limit", "1"]))
    arm = json.loads((out / "online_summary.json").read_text(encoding="utf-8"))[0]
    assert (arm["wrong"], arm["quarantined_at"], arm["trusted_again_at"], arm["under_refunded_usd"]) == (15, 22, 38,
                                                                                                     192.45)
