"""M5 and D1: the expense world (metacog's second task class), its smoke study, and the full D1 study."""

from __future__ import annotations

import json
import re
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


STUDY = Path(__file__).parents[1] / "examples" / "expenses" / "recorded" / "study.jsonl"
TRACE = re.compile(r"^\s*(\d+) (\S+)\s+(yes|no)\s+(\S+)\s+(accepted|rejected)", re.MULTILINE)
# D1's online loop: (case, procedure offered, effort, accepted). The policy changes at F701.
D1_TRACE = ([(f"E20{i}", "yes", "none", "accepted") for i in range(1, 7)]
            + [("F701", "yes", "none", "rejected"), ("F702", "yes", "none", "rejected"),
               ("F703", "no", "high", "accepted"), ("F704", "no", "high", "accepted"),
               ("F705", "no", "high", "rejected"), ("F706", "no", "high", "accepted")]
            + [(f"F7{i:02d}", "yes", "none", "accepted") for i in range(7, 15)])


async def test_the_recorded_d1_study_replays(tmp_path, capsys):
    """D1 (DOMAINS.md): the whole expense study, live once on 2026-11-05, replayed from its recording."""
    if not STUDY.exists():
        pytest.skip("no recorded D1 study")
    await study.main(study.parse(["--recording", str(STUDY), "--output", str(tmp_path)]))
    out = capsys.readouterr().out
    efforts = ("none", "low", "medium", "high")
    without = Floors.load(tmp_path / "floors_no_memory.json").sweeps["no_memory"]
    assert [without.cells[e].solved for e in efforts] == [0, 6, 9, 12]
    assert "validation 16/16 accepted, 16/16 match the hand-written reference" in out
    learned = Floors.load(tmp_path / "floors_memory_learned.json").sweeps["memory"]
    assert [learned.cells[e].solved for e in efforts] == [16, 16, 16, 16]
    assert [m.groups()[1:] for m in TRACE.finditer(out)] == D1_TRACE
    assert "detector: quarantined reimbursable_total@1" in out
    assert "value_learning wrote reimbursable_total from 4 runs; gate 6/6 passed, installed" in out
    summary = json.loads((tmp_path / "online_summary.json").read_text(encoding="utf-8"))
    assert (summary["learning"]["writer_calls"], round(summary["learning"]["dollars"], 4)) == (3, 0.0511)
    assert round(summary["break_even_requests"], 1) == 8.5
    assert summary["procedures"]["reimbursable_total"] == [[1, "quarantined", "value_learning"],
                                                           [2, "trusted", "value_learning"]]

    # D1's repeats: both measurement phases three times over, with the same learned procedure.
    await study.main(study.parse(["--phases", "floors", "memory", "--repeats", "3", "--recording", str(STUDY),
                                  "--output", str(tmp_path)]))
    without = Floors.load(tmp_path / "floors_no_memory.json").sweeps["no_memory"]
    assert [without.cells[e].solved for e in efforts] == [0, 15, 29, 35]  # of 48
    learned = Floors.load(tmp_path / "floors_memory_learned.json").sweeps["memory"]
    assert [learned.cells[e].solved for e in efforts] == [48, 48, 48, 48]


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
