"""D3: the accounts-payable desk (DOMAINS.md), its hand-written procedures, its audit, routing, discovery,
and the online loop with several procedures and several clients, all with scripted replies (no API key)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from examples.invoices import study
from examples.invoices.agent import (
    PARSER,
    PARSER_BEFORE,
    PAYABLE,
    before_layout_change,
    labeled_practice,
    practice,
)
from examples.invoices.world import (
    CLIENTS,
    PAYMENT_DIAGNOSES,
    TACIT,
    VENDORS,
    InvoiceWorld,
    payment,
    record_of,
    rules_that_matter,
    task,
    task_set,
)
from metacog import Recording
from metacog.meta import MetaStore
from metacog.offline import Floors
from metacog.testing import RespondingModel, call, text

WORK = ("intake", "payment")


@pytest.mark.parametrize("split", ["dev", "validation", "test", "stream", "gate", "payable_gate"])
async def test_the_hand_written_procedures_are_right_on_every_case(split):
    for case in [t for t in task_set(split) if t["kind"] in WORK]:
        source = PARSER if case["kind"] == "intake" else PAYABLE
        assert (await practice(source, case["id"]))[0] == 1.0, case["id"]
        assert (await labeled_practice(source, case["id"]))[0] == 1.0, case["id"]


async def test_the_parser_written_before_the_change_misreads_only_kestrels_new_layout():
    assert PARSER_BEFORE == before_layout_change(PARSER) != PARSER
    for split in ("dev", "validation", "test", "stream", "gate"):
        for case in [t for t in task_set(split) if t["kind"] == "intake"]:
            reward, reason, _ = await practice(PARSER_BEFORE, case["id"])
            new_layout = case["drift"] and case["vendor"] == "KES"
            assert reward == (0.0 if new_layout else 1.0), case["id"]
            if new_layout:  # wrong without failing: it returned a record, and the audit names the misreading
                assert reason.startswith("it read the document's amounts in the wrong number format"), reason
    assert sum(t["kind"] == "intake" and t["vendor"] == "KES" for t in task_set("gate")) == 3


def test_every_split_mixes_the_work_and_the_clients():
    for split in ("dev", "validation", "test"):
        tasks = task_set(split)
        assert {t["kind"] for t in tasks} == {"intake", "payment", "question"}
        assert {t["client"] for t in tasks} == set(CLIENTS)
    stream = task_set("stream")
    assert [t["drift"] for t in stream] == [False] * 20 + [True] * 40
    assert sum(t["kind"] == "intake" and t["vendor"] == "KES" and t["drift"] for t in stream) >= 8
    assert task(207) == task(207) and task(207)["id"] == "V207"  # requests come from their numbers alone


def test_the_unwritten_conventions_change_payments_and_show_only_in_verdicts():
    payments = [t for s in ("dev", "validation", "test", "stream") for t in task_set(s) if t["kind"] == "payment"]
    for client, rule in TACIT.items():
        mine = [t for t in payments if t["client"] == client]
        hinging = [t for t in mine if rule in rules_that_matter(t)]
        assert 0 < len(mine) - len(hinging) < len(hinging)  # most payments turn on it; some don't
        for case in hinging:
            amount, pay_on = payment(case, ignore=rule)
            world = InvoiceWorld(case)
            world.schedule_payment(case["invoice"]["invoice_id"], amount, pay_on)
            assert world.check()["reason"].endswith(PAYMENT_DIAGNOSES[rule]), case["id"]
    terms = json.dumps([InvoiceWorld(payments[0]).read_client_terms(c) for c in CLIENTS]).lower()
    assert not any(word in terms for word in ("carrier", "whole dollar", "retention", "rounding down"))


def test_the_audit_names_the_broken_rule_without_the_answer():
    for case in [t for t in task_set("dev") + task_set("test") if t["kind"] == "payment"]:
        right = payment(case)
        for rule in rules_that_matter(case):
            amount, pay_on = payment(case, ignore=rule)
            world = InvoiceWorld(case)
            world.schedule_payment(case["invoice"]["invoice_id"], amount, pay_on)
            verdict = world.check()
            # Two mistakes can give the same payment; the audit names one that explains it.
            explains = [r for r in PAYMENT_DIAGNOSES if payment(case, ignore=r) == (amount, pay_on)]
            assert not verdict["success"] and any(PAYMENT_DIAGNOSES[r] in verdict["reason"] for r in explains)
            if abs(amount - right[0]) > 0.01:
                assert f"{right[0]:.2f}" not in verdict["reason"]
    intake = next(t for t in task_set("dev") if t["kind"] == "intake" and t["invoice"]["freight_usd"])
    record = record_of(intake)
    swapped = {**record, "invoice_date": record["invoice_date"][:5] + record["invoice_date"][8:] + "-" +
               record["invoice_date"][5:7]}
    padded = {**record, "subtotal_usd": record["subtotal_usd"] + record["freight_usd"]}
    for wrong, said in ((swapped, "day and month the wrong way round"), (padded, "the subtotal includes the freight")):
        world = InvoiceWorld(intake)
        world.record_invoice(**wrong)
        assert said in world.check()["reason"]
    world = InvoiceWorld(intake)
    assert world.check()["reason"] == "nothing was recorded"
    question = next(t for t in task_set("dev") if t["kind"] == "question")
    assert InvoiceWorld(question).check(f"Use {VENDORS[question['vendor']]['remittance_email']}.")["success"]
    assert not InvoiceWorld(question).check("I don't know.")["success"]


def test_routing_calibration_separates_the_three_kinds():
    calibration = study.routing_calibration()
    assert calibration.accuracy == 1.0
    assert study.routed_right(calibration.router, "test") == (16, 16)
    assert study.routed_right(study.Router(k=2), "test") == (0, 16)  # without thresholds both are always offered


# ------------------------------------------------------------------ the online loop, scripted

def requested(request: str) -> dict:
    """The task a request names: by its document or invoice number, or a question about a vendor."""
    found = re.search(r"(DOC|INV)-(\d+)", request)
    if found:
        return task(int(found.group(2)))
    vendor = next(v for v, info in VENDORS.items() if info["name"] in request)
    return {"kind": "question", "vendor": vendor}


def desk(request: dict):
    """A scripted model for the whole loop. The writer writes the hand-written procedure the contract asks for.
    The agent calls an offered procedure for its kind of work and passes the answer to the action tool;
    without one it reads what it needs and acts correctly. Intake spends more reasoning than payment."""
    system = request["messages"][0]["content"]
    if system.startswith("You are writing a Python function"):
        asked = json.dumps(request["messages"])
        return text(f"```python\n{PARSER if 'def parse_invoice' in asked else PAYABLE}```")
    case = requested(request["messages"][1]["content"])
    tools = [m for m in request["messages"] if m["role"] == "tool"]
    names = {t["function"]["name"] for t in request.get("tools") or []}
    if tools and json.loads(tools[-1]["content"]).get("status") in ("recorded", "scheduled"):
        return text("Done.")
    if case["kind"] == "question":
        if not tools:
            return call("read_vendor", {"vendor": VENDORS[case["vendor"]]["name"]})
        return text(f"Send it to {json.loads(tools[-1]['content'])['remittance_email']}.")
    effort = 400 if case["kind"] == "intake" else 100
    procedure, action = (("parse_invoice", "record_invoice") if case["kind"] == "intake" else
                         ("amount_payable", "schedule_payment"))
    argument = ({"doc_id": case["doc_id"]} if case["kind"] == "intake" else
                {"invoice_id": case["invoice"]["invoice_id"]})
    if procedure in names:
        if not tools:
            return call(procedure, argument)
        return call(action, json.loads(tools[-1]["content"]))  # the procedure's answer, copied across
    reads = ([("read_document", argument)] if case["kind"] == "intake" else
             [("read_invoice", argument), ("read_purchase_order", {"po_number": case["po"]["po_number"]}),
              ("read_client_terms", {"client_id": case["client"]})])
    if len(tools) < len(reads):
        return call(*reads[len(tools)], reasoning_tokens=effort)
    return call(action, case["expected"], reasoning_tokens=effort)


def online(tmp_path, *extra: str):
    return study.parse(["--phases", "online", "--executor", "guarded", "--output", str(tmp_path),
                        "--recording", str(tmp_path / "unused.jsonl"), *extra])


async def test_the_layout_change_quarantines_only_the_parser_and_a_labeled_gate_readmits_it(tmp_path):
    args = online(tmp_path)
    with RespondingModel(desk).active():
        result = await study.run_arm("routed", args, Recording(tmp_path / "unused.jsonl"))
    rows = json.loads((tmp_path / "online_routed.json").read_text(encoding="utf-8"))
    assert result["routed_to_own"] == 60  # each request offered its own procedure, questions none
    assert result["wrong"] == result["wrong_after_drift_kestrel_intake"] > 0  # only the stale parser's misreadings
    assert result["parser_quarantines"] == 1
    assert 20 < result["parser_quarantined_at"] < result["parser_trusted_again_at"]
    assert result["payable_quarantines"] == 0  # blame went to the procedure the failing runs called
    assert all(r["audit"] for r in rows if r["n"] >= result["parser_trusted_again_at"])
    parser = [a for a in result["attempts"] if a["procedure"] == "parse_invoice"]
    assert [a["gate"]["passed"] for a in parser] == [True] and parser[0]["gate"]["total"] == 6 + 18
    assert [(v, state) for v, state, _ in result["procedures"]["shared"]["parse_invoice"]] == [
        (1, "quarantined"), (2, "trusted")]


async def test_without_routing_offered_blame_quarantines_the_payable_too(tmp_path):
    args = online(tmp_path)
    with RespondingModel(desk).active():
        result = await study.run_arm("unrouted", args, Recording(tmp_path / "unused.jsonl"))
    assert result["routed_to_own"] == 0
    # Every run offered both procedures, so the parser's failures were counted against the payable as well.
    assert result["payable_quarantines"] >= 1 and result["payable_quarantined_at"] == result["parser_quarantined_at"]
    assert any(a["procedure"] == "amount_payable" for a in result["attempts"])  # learning spent on a sound procedure


async def test_per_client_scopes_learn_from_their_own_clients_and_relearn_the_shared_work_apart(tmp_path):
    args = online(tmp_path, "--payable", "learn")
    with RespondingModel(desk).active():
        result = await study.run_arm("per_client", args, Recording(tmp_path / "unused.jsonl"))
    learned = {(a["scope"], a["procedure"]) for a in result["attempts"] if a["outcome"].endswith("installed")}
    # Each client learned its own payable, from its own runs, and checked it on its own labeled payments.
    assert {("ALDER", "amount_payable"), ("BRAMBLE", "amount_payable"), ("COPPER", "amount_payable")} <= learned
    gates = [a["gate"] for a in result["attempts"] if a["procedure"] == "amount_payable" and a["gate"]]
    assert gates and all(g["total"] == 6 + 6 for g in gates)
    rows = json.loads((tmp_path / "online_per_client.json").read_text(encoding="utf-8"))
    client_of = {r["case"]: r["client"] for r in rows}
    for client in CLIENTS:
        store = MetaStore(str(tmp_path / "per_client" / f"{client}.meta"))
        assert {client_of[run["case_id"]] for run in store.runs()} == {client}
    # The parser is shared work, but each client's scope met the layout change and relearned it apart.
    assert result["parser_quarantines"] >= 2


async def test_discover_finds_the_parsing_work_before_anyone_declares_it(tmp_path):
    args = study.parse(["--phases", "discover", "--output", str(tmp_path), "--recording",
                        str(tmp_path / "unused.jsonl")])
    with RespondingModel(desk).active():
        found = await study.phase_discover(args, Recording(tmp_path / "unused.jsonl"))
    assert found["parsing_rank"] == 1
    tools = [g["tools"] for g in found["groups"]]
    assert tools[:2] == [["read_document", "record_invoice"],
                         ["read_client_terms", "read_invoice", "read_purchase_order", "schedule_payment"]]


D3_STUDY = Path(__file__).parents[1] / "examples" / "invoices" / "recorded" / "study.jsonl"
DESIGNED = Path(__file__).parents[1] / "examples" / "learning" / "designed"


async def test_the_recorded_d3_study_replays_as_far_as_it_ran(tmp_path):
    """D3 live on 2026-11-06 (DOMAINS.md), with the payable learned from scratch, learner tournaments, and the two
    designs that transferred approved as candidates. It stopped at its $3 budget, 37 requests into the routed
    arm; those 37 replay."""
    if not D3_STUDY.exists():
        pytest.skip("no recorded D3 study")
    designs = [str(DESIGNED / f"{name}.py") for name in ("designed_ee25cd476662", "designed_e48a9701c5e3")]
    await study.main(study.parse(["--recording", str(D3_STUDY), "--output", str(tmp_path), "--payable", "learn",
                                  "--selection", "--designed", *designs, "--arms", "routed", "--requests", "37"]))
    without = Floors.load(tmp_path / "floors_no_memory.json").sweeps["no_memory"]
    assert [without.cells[e].solved for e in ("none", "low", "medium", "high")] == [4, 8, 9, 8]
    with_procedures = Floors.load(tmp_path / "floors_memory.json").sweeps["memory"]
    assert [with_procedures.cells[e].solved for e in ("none", "low", "medium", "high")] == [16, 16, 16, 16]
    assert json.loads((tmp_path / "discover.json").read_text(encoding="utf-8"))["parsing_rank"] == 2
    rows = json.loads((tmp_path / "online_routed.json").read_text(encoding="utf-8"))
    tournament = next(e for r in rows for e in r["events"] if e.startswith("learner: tournament for amount_payable"))
    # Every seed, and the second design, gets none of the 18 labeled payments (6 of 24 are the parser's cases);
    # the design searched on D1 and D2 gets all of them, and its version serves from then on.
    assert "designed_ee25cd476662 24/24" in tournament and tournament.endswith("chose designed_ee25cd476662")
    assert all(f"{seed} 6/24" in tournament for seed in ("value_learning", "reflection", "success_distillation",
                                                         "transition_distillation", "skill_synthesis"))
    arm = json.loads((tmp_path / "online_summary.json").read_text(encoding="utf-8"))[0]
    assert arm["payable_learned_at"] == 16
