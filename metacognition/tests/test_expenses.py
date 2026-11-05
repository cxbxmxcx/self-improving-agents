"""M5: the expense world (metacog's second task class) and its recorded live smoke study."""

from __future__ import annotations

from pathlib import Path

import pytest

from examples.expenses import study
from examples.expenses.agent import REFERENCE, practice, reimbursable_total
from examples.expenses.world import DIAGNOSES, ExpenseWorld, _rules_that_matter, reimbursable, task_set
from metacog.offline import Floors

SMOKE = Path(__file__).parents[1] / "examples" / "expenses" / "recorded" / "smoke.jsonl"
# Trip-start conversion always: the procedure as it stood before the policy change.
STALE = REFERENCE.replace('any("month on the receipt" in note for note in policy["notes"])', "False")


@pytest.mark.parametrize("split", ["dev", "validation", "test", "stream", "gate"])
async def test_the_hand_written_procedure_meets_the_policy_on_every_case(split):
    tasks = task_set(split)
    rewards = [(await practice(REFERENCE, t["id"]))[0] for t in tasks]
    assert rewards == [1.0] * len(tasks)
    assert all(len(_rules_that_matter(t["trip"], t["drift"])) >= 3 for t in tasks)


def test_the_checker_names_the_broken_rule_without_the_answer():
    for task in task_set("dev")[:6]:
        for rule in _rules_that_matter(task["trip"], task["drift"]):
            world = ExpenseWorld(task)
            world.file_claim(task["trip"]["trip_id"], reimbursable(task["trip"], ignore=rule))
            verdict = world.check()
            assert not verdict["success"] and DIAGNOSES[rule][0] in verdict["reason"], (task["id"], rule)
            assert f"{task['expected_usd']:.2f}" not in verdict["reason"]


async def test_after_the_policy_change_a_stale_procedure_is_rejected_for_its_rates():
    before = [(await practice(STALE, t["id"]))[0] for t in task_set("test")[:6]]
    after = [await practice(STALE, t["id"]) for t in task_set("gate")]
    assert before == [1.0] * 6
    assert all(reward == 0.0 and "month the trip started" not in reason and "starting month" in reason
               for reward, reason, _ in after)


def test_the_procedure_contract_comes_from_the_stub():
    contract = reimbursable_total.contract
    assert contract.signature == "def reimbursable_total(api, trip_id: str) -> dict:"
    assert contract.returns == '{"total_usd": float}, the reimbursable total rounded to the cent.'
    assert "`api.read_trip(trip_id)`" in contract.api and "`api.read_policy()`" in contract.api


async def test_the_recorded_smoke_study_replays(tmp_path):
    if not SMOKE.exists():
        pytest.skip("no recorded smoke study")
    args = study.parse(["--phases", "floors", "memory", "--tasks", "4", "--efforts", "none", "high", "--memory",
                        "reference", "--recording", str(SMOKE), "--output", str(tmp_path)])
    await study.main(args)
    without = Floors.load(tmp_path / "floors_no_memory.json")
    with_procedure = Floors.load(tmp_path / "floors_memory_reference.json")
    assert [without.sweeps["no_memory"].cells[e].solved for e in ("none", "high")] == [0, 2]
    assert [with_procedure.sweeps["memory"].cells[e].solved for e in ("none", "high")] == [4, 4]
    assert with_procedure.floor("memory") == "none" and without.floor("no_memory") is None
