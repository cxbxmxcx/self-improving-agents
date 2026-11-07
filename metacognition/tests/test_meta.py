"""The metacognition runtime on the scripted basket domain (see basket.py)."""

from __future__ import annotations

import asyncio
import threading

import pytest
from basket import CASES, V1, V2, WRITTEN, agent, basket_total, make_meta, practice, serve, tax, uses, verifier
from basket import price as price_tool

from metacog import RunConfig, RunContextWrapper, Runner, UserError, function_tool
from metacog.meta import (
    CALENDAR,
    GENERIC,
    AgentPractice,
    Controller,
    Detector,
    Gate,
    Metacognition,
    MetaStore,
    SkillAPI,
    Verdict,
    procedure,
)
from metacog.meta.mechanisms import contract_text
from metacog.testing import ScriptedModel, text


async def test_the_online_loop_quarantines_relearns_and_reinstalls():
    meta = make_meta()
    meta.install("basket_total", V1)
    shop = agent(meta)
    model = ScriptedModel([*uses(["apple", "pear"], 5), *uses(["apple", "pear"], 5), *uses(["pear", "pear"], 6),
                           text("7"), text("8"), WRITTEN, WRITTEN, *uses(["apple", "apple"], 6)])
    reports = []
    with model.active():
        for case_id in ("S1", "D1", "D2", "A", "B", "D3"):
            reports.append((await serve(shop, case_id)).metacognition)
    assert [(r.effort, bool(r.offered), r.verdict.accepted) for r in reports] == [
        ("none", True, True), ("none", True, False), ("none", True, False),
        ("high", False, True), ("high", False, True), ("none", True, True)]
    assert reports[2].events[0].startswith("detector: quarantined basket_total@1")
    assert reports[4].events == ["learner: value_learning wrote basket_total from 2 runs; gate 2/2 passed, installed"]
    # The agent saw the procedure as a tool with one line of context, and the controller's setting.
    first = model.requests[0]
    assert [t["function"]["name"] for t in first["tools"]] == ["price", "tax", "basket_total"]
    assert "Use the tool `basket_total`. Add up what the listed items cost." in first["messages"][0]["content"]
    assert first["reasoning_effort"] == "none" and model.requests[6]["reasoning_effort"] == "high"
    # The writer saw only the domain-neutral pack.
    assert "calendar" not in model.requests[8]["messages"][0]["content"].lower()
    assert meta.summary()["procedures"]["basket_total"] == [(1, "quarantined", "human"),
                                                            (2, "trusted", "value_learning")]
    assert model.replies == []


async def test_human_approval_holds_a_passing_version_until_approved():
    meta = make_meta(approval="human")
    meta.install("basket_total", V1)
    shop = agent(meta)
    model = ScriptedModel([*uses(["apple", "pear"], 5), *uses(["pear", "pear"], 6), text("7"), text("8"),
                           WRITTEN, WRITTEN, text("6"), *uses(["apple", "apple"], 6)])
    with model.active():
        for case_id in ("D1", "D2", "A", "B"):
            last = await serve(shop, case_id)
        assert last.metacognition.events[-1].endswith("proposed for approval")
        assert [p["version"] for p in meta.proposals()] == [2]
        waiting = await serve(shop, "D3")
        meta.approve("basket_total", 2)
        approved = await serve(shop, "D3")
    assert (waiting.metacognition.effort, approved.metacognition.offered) == ("high", ["basket_total@2"])


async def test_deferred_feedback_drives_the_detector():
    meta = Metacognition([basket_total], recall_k=1, practice=practice, learning="inline")
    assert meta.approval == "human"  # no verifier: a person approves installs
    meta.install("basket_total", V1)
    shop = agent(meta)
    with ScriptedModel([*uses(["apple", "pear"], 5)] * 3).active():
        results = [await serve(shop, "D1") for _ in range(3)]
    assert all(r.metacognition.verdict is None for r in results)
    await meta.feedback(results[0].metacognition.run_id, True)
    await meta.feedback(results[1].metacognition.run_id, False, "tax was missing")
    assert meta.store.trusted("basket_total")["version"] == 1
    await meta.feedback(results[2].metacognition.run_id, False, "tax was missing")
    assert meta.store.trusted("basket_total") is None
    with pytest.raises(KeyError):
        await meta.feedback("run_unknown", True)


async def test_background_learning_and_a_holdout_gate():
    meta = make_meta(learning="background", gate=Gate(holdout=1, bar=1.0))
    meta.install("basket_total", V1)
    shop = agent(meta)
    model = ScriptedModel([*uses(["apple", "pear"], 5), *uses(["pear", "pear"], 6), text("7"), text("8"), WRITTEN])
    with model.active():
        for case_id in ("D1", "D2", "A", "B"):
            await serve(shop, case_id)
        await meta.wait_for_learning()
    # The last accepted run (B) was held back as the gate's case, so the learner wrote from A alone.
    attempt = meta.store.learning()[-1]
    assert (attempt["episodes"], attempt["writer_calls"], attempt["gate"]["total"]) == (1, 1, 1)
    assert meta.store.trusted("basket_total")["version"] == 2


async def test_the_gate_can_try_its_cases_its_own_way():
    # As D2's second pass does: the learner practices one way (there, by replaying recorded runs) and the
    # gate checks its labeled cases another (there, with a function and the audit).
    tried: dict[str, list[str]] = {"learner": [], "gate": []}

    def counted(who: str):
        async def wrapped(source: str, case_id: str):
            tried[who].append(case_id)
            return await practice(source, case_id)
        return wrapped

    meta = make_meta(practice=counted("learner"), gate=Gate(cases=["G1", "G2"], bar=1.0, practice=counted("gate")))
    meta.install("basket_total", V1)
    shop = agent(meta)
    model = ScriptedModel([*uses(["apple", "pear"], 5), *uses(["apple", "pear"], 5), *uses(["pear", "pear"], 6),
                           text("7"), text("8"), WRITTEN, WRITTEN])
    with model.active():
        for case_id in ("S1", "D1", "D2", "A", "B"):
            await serve(shop, case_id)
    assert tried["gate"] == ["G1", "G2"]
    assert tried["learner"] and not {"G1", "G2"} & set(tried["learner"])
    assert meta.store.trusted("basket_total")["version"] == 2


async def test_cold_start_learns_a_procedure_it_never_had():
    meta = make_meta()
    shop = agent(meta)
    model = ScriptedModel([text("7"), text("8"), WRITTEN, WRITTEN])
    with model.active():
        await serve(shop, "A")
        last = await serve(shop, "B")
    assert last.metacognition.effort == "high" and last.metacognition.events[-1].endswith("installed")
    assert meta.store.trusted("basket_total")["created_by"] == "value_learning"


async def test_frozen_runs_recall_without_recording_or_control():
    meta = make_meta()
    meta.install("basket_total", V1)
    model = ScriptedModel(uses(["apple", "pear"], 5))
    with model.active():
        result = await Runner.run(agent(meta), "What do apple, pear cost?", context=CASES["S1"],
                                  run_config=RunConfig(metacognition_mode="frozen"))
    assert result.metacognition.offered == ["basket_total@1"] and meta.store.count() == 0
    assert "reasoning_effort" not in model.requests[0]


async def test_agent_practice_runs_the_candidate_through_the_agent():
    meta = make_meta(practice=AgentPractice(world=lambda case: case))
    shop = agent(meta)
    model = ScriptedModel([text("7"), *uses(["apple", "pear"], 7)])
    with model.active():
        await serve(shop, "A")
        reward, reason, _ = await meta._practice_for("basket_total")(V2, "A")
    assert (reward, reason, meta.store.count()) == (1.0, "the total is 7", 1)
    practice_request = model.requests[1]
    assert practice_request["tools"][-1]["function"]["name"] == "basket_total"
    assert practice_request["reasoning_effort"] == "none"


async def test_skill_api_calls_sync_and_async_tools_from_the_worker_thread():
    ctx = RunContextWrapper(context=CASES["D1"])
    api = SkillAPI([price_tool, tax], ctx, asyncio.get_running_loop())
    seen = []

    def work():
        seen.append((api.price("apple"), api.price(item="pear"), api.tax(), threading.current_thread().name))

    await asyncio.to_thread(work)
    assert seen[0][:3] == (2, 3, 1)
    with pytest.raises(AttributeError):
        api.cancel_order  # noqa: B018 - only listed read-only tools exist
    with pytest.raises(RuntimeError):
        api.tax()  # an async tool on the loop's own thread would deadlock


def test_procedure_stubs_become_contracts():
    contract = basket_total.contract
    assert contract.signature == "def basket_total(api, items: list[str]) -> int:"
    assert contract.description == "Add up what the listed items cost."
    assert contract.returns == "the total as an integer."
    assert "`api.price(item)` calls the price tool" in contract.api and "`api.tax()`" in contract.api
    assert basket_total.reference is None

    @procedure
    def double(api, x: int) -> int:
        """Double a number."""
        return x + x

    assert double.reference.startswith("def double(api, x: int) -> int:")
    meta = Metacognition([double], verifier=verifier)
    assert meta.store.trusted("double")["created_by"] == "human"

    with pytest.raises(UserError):
        @procedure
        def no_api(x: int) -> int:
            """Needs api first."""
            return x

    with pytest.raises(UserError):
        procedure(api=[function_tool(lambda: 1, name_override="writes")])(basket_total_stub)


def basket_total_stub(items: list[str]) -> int:
    """A stub."""
    ...


def test_policy_parts():
    detector = Detector(window=4, limit=2)
    recent, quarantine = detector.observe([True, True, True], False)
    assert (recent, quarantine) == ([True, True, True, False], False)
    recent, quarantine = detector.observe(recent, False)
    assert (recent, quarantine) == ([True, True, False, False], True)
    assert Controller().effort("p", trusted=True) == "none" and Controller().effort("p", trusted=False) is None
    controller = Controller.from_floors({"p": {"memory": "low", "no_memory": "medium"}})
    assert (controller.effort("p", True), controller.effort("p", False)) == ("low", "medium")
    assert Verdict.coerce({"success": False, "reason": "late"}) == Verdict(False, "late", "verifier")
    assert Verdict.coerce((True, "ok"), "human") == Verdict(True, "ok", "human")
    assert Verdict.coerce(None) is None


async def test_gate_bar_and_minimum():
    gate = Gate(bar=7 / 8, min_cases=2)
    assert not (await gate.check(V2, practice, ["G1"])).passed
    assert (await gate.check(V2, practice, ["G1", "G2"])).passed
    assert not (await gate.check(V1, practice, ["G1", "G2"])).passed


def test_store_versions_on_disk(tmp_path):
    store = MetaStore(tmp_path / "store")
    assert store.add_version("p", V1, state="trusted", created_by="human") == 1
    assert store.add_version("p", V2, state="candidate", created_by="value_learning", parent=1) == 2
    assert store.trusted("p")["version"] == 1
    assert (tmp_path / "store" / "meta.sqlite").exists()


def test_prompt_packs():
    assert contract_text(basket_total.contract, CALENDAR).startswith(
        "You are writing a Python function that a calendar assistant will call")
    assert "calendar" not in contract_text(basket_total.contract, GENERIC).lower()


def test_a_later_verdict_on_a_run_replaces_its_earlier_one():
    detector = Detector(window=4, limit=2)
    window, quarantine = detector.record([], "r1", False, source="judge")
    assert (window, quarantine) == ([["r1", False, "judge"]], False)
    assert detector.record(window, "r1", False, source="judge") == (window, False)  # the same run: replaced, not added
    assert detector.record(window, "r1", True, source="human") == ([["r1", True, "human"]], False)  # overturned
    assert detector.record(window, "r2", False)[1]
    full = [["r3", True], ["r4", True], ["r5", True], ["r6", True]]  # a window stored before sources were kept
    assert detector.record(full, "r1", False, counted=["r1"]) == ([[*e, None] for e in full], False)


def test_a_source_trusted_more_can_quarantine_sooner():
    detector = Detector(window=4, limit=2, limits={"human": 1})
    assert not detector.record([], "r1", False, source="judge")[1]  # a judge still needs two
    assert detector.record([], "r1", False, source="human")[1]  # one rejection from a person is enough
    window, _ = detector.record([], "r1", False, source="judge")
    assert detector.record(window, "r1", False, source="human")[1]  # a person confirms the judge's rejection


async def test_one_rejection_from_a_person_can_be_enough():
    meta = make_meta(verifier=None, detector=Detector(limits={"human": 1}))
    meta.install("basket_total", V1)
    with ScriptedModel([*uses(["apple", "pear"], 5)]).active():
        run_id = (await serve(agent(meta), "D1")).metacognition.run_id
    assert await meta.feedback(run_id, False, "the tax is missing") == [
        "detector: quarantined basket_total@1 (the tax is missing)"]


async def test_feedback_counts_each_run_once_and_says_what_it_set_off():
    meta = make_meta(verifier=None)
    meta.install("basket_total", V1)
    with ScriptedModel([*uses(["apple", "pear"], 5), *uses(["pear", "pear"], 6)]).active():
        first = (await serve(agent(meta), "D1")).metacognition.run_id
        second = (await serve(agent(meta), "D2")).metacognition.run_id
    assert await meta.feedback(first, False, "the tax is missing") == []
    assert await meta.feedback(first, False, "the tax is still missing") == []  # one run, one count
    assert meta.store.trusted("basket_total")["version"] == 1
    assert await meta.feedback(second, False, "the tax is missing") == [
        "detector: quarantined basket_total@1 (the tax is missing)"]
