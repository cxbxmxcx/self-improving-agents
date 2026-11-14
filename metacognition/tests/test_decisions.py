"""The decision tier: signal, gap, gate and per-class switch, on a scripted model and a fake Jev (no key)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import helix
import pytest
from helix.deciders import Answer, DecisionUsage
from helix.memory.learning_mechanisms import Contract
from helix.memory.mechanism_runtime import UnsafeMechanism, check_runnable, pack_memories, unpack_memories
from helix.metacognition import Controller, Gate, Metacognition, procedure
from helix.metacognition.decide import KIT_GUIDE, RETURNS, Desk, Kit, frozen_change, run_program
from helix.metacognition.gap import Attempt, Good, gap_of, outcome_of, report
from helix.metacognition.decisions import DECISION, RUNG, ProgramSearch, ProgramTier, ServiceGap
from helix.signal import Signal
from helix.search.methods import METHODS

from metacog import Agent, RunConfig, RunContextWrapper, Runner, function_tool
from metacog.testing import RespondingModel, ScriptedModel, call, text


class World:
    def __init__(self, task: dict[str, Any]) -> None:
        self.task = task
        self.shipped: list[str] = []

    def check(self) -> dict[str, Any]:
        if self.task.get("expect") == "handoff":
            return {"success": not self.shipped, "reason": "acted" if self.shipped else "no action"}
        return {"success": self.shipped == [self.task["order"]] and self.task["order"] != "O-13",
                "reason": "accepted" if self.task["order"] != "O-13" else "that order can't ship"}


@function_tool(read_only=True)
def read_order(ctx: RunContextWrapper[World], order_id: str) -> dict:
    """Read an order."""
    return {"order_id": order_id}


@function_tool
def ship(ctx: RunContextWrapper[World], order_id: str) -> dict:
    """Ship an order."""
    ctx.context.shipped.append(order_id)
    return {"status": "shipped"}


@procedure(api=[read_order])
def order_status(order_id: str) -> dict:
    """Work out an order's status.

    Returns: {"status": str}.
    """
    ...


@procedure(api=[read_order])
def order_weight(order_id: str) -> dict:
    """Work out what an order weighs.

    Returns: {"grams": int}.
    """
    ...


STATUS = 'def order_status(api, order_id: str) -> dict:\n    return {"status": "ok"}\n'
WEIGHT = 'def order_weight(api, order_id: str) -> dict:\n    return {"grams": 100}\n'
GOOD = '''async def decide(request, kit):
    ids = kit.find("O-[0-9]+", request)
    answers = await kit.ask({"request": request}, {"go": {"type": "noul", "instructions": "Ship it now?"}})
    if answers["go"]["noul"] < 0.7 or len(ids) != 1:
        return {"handoff": "not sure"}
    return {"action": "ship", "arguments": {"order_id": ids[0]}, "reply": "Shipped."}
'''
RECKLESS = '''async def decide(request, kit):
    ids = kit.find("O-[0-9]+", request)
    return {"action": "ship", "arguments": {"order_id": ids[0]}, "reply": "Shipped."}
'''


def mechanism(program: str) -> str:
    return f"PROGRAM = {program!r}\n\n\nasync def learn(lab):\n    return PROGRAM\n"


class FakeJev:
    name = "fake"

    async def ask_with_usage(self, state: Any, questions: dict[str, Any]) -> tuple[dict[str, Any], DecisionUsage]:
        said = state["request"].lower()
        answer = Answer("probability", 0.1 if "don't" in said else 0.9)
        return {"go": answer}, DecisionUsage(calls=1, input_tokens=100, dollars=100 * 42 / 1e9, seconds=0.05)


def task(n: int, text_: str | None = None, expect: str = "act") -> dict[str, Any]:
    return {"id": f"c{n}", "order": f"O-{n}", "text": text_ or f"Ship O-{n}.", "expect": expect}


LABELED = {t["id"]: t for t in [task(101), task(102), task(103), task(104, "Don't ship O-104.", "handoff")]}


def tier(*programs: str, **overrides: Any) -> ProgramTier:
    desk = Desk(world=World, reads=lambda world: {}, procedures=lambda world: {},
                actions={"ship": lambda world, order_id: world.shipped.append(order_id)},
                verdict=lambda world, reply: world.check())
    contract = Contract(name="decide", signature="async def decide(request, kit) -> dict", description="Ship orders.",
                        api=KIT_GUIDE, returns=RETURNS)
    settings = {"min_runs": 3, "retry_after": 2, **overrides}
    mechanisms = settings.pop("mechanisms", [(f"m{i}", mechanism(p)) for i, p in enumerate(programs)])
    return ProgramTier(desk=desk, decider=FakeJev(), contract=contract, labeled=LABELED, mechanisms=mechanisms,
                       writer_model="gpt-5.4-mini", **settings)


def build(*programs: str, trusted: str | None = None, procedures: Any = (order_status,),
          **overrides: Any) -> tuple[Agent, Metacognition]:
    def verify(ctx: RunContextWrapper[World], result: Any) -> dict[str, Any]:
        return ctx.context.check() if ctx.context.task.get("judge", True) else {"success": True, "reason": "agent"}

    meta = Metacognition(list(procedures), verifier=verify, learning="inline", relearn_after=10**9,
                         programs=tier(*programs, **overrides), executor="guarded")
    meta.install("order_status", STATUS)
    if trusted:
        meta.store.add_version(DECISION, trusted, state="trusted", created_by="test")
    agent = Agent(name="Shop", instructions="Ship orders.", tools=[read_order, ship], model="gpt-5.4-mini",
                  metacognition=meta)
    return agent, meta


@pytest.fixture
def model() -> RespondingModel:
    return RespondingModel(lambda request: text("done", prompt_tokens=2000, completion_tokens=100))


async def run(agent: Agent, item: dict[str, Any]) -> tuple[Any, World]:
    world = World(item)
    result = await Runner.run(agent, item["text"], context=world,
                              run_config=RunConfig(case_id=item["id"], case_data=item))
    return result, world


async def test_the_report_says_what_the_program_did_and_a_gap_counts_it(model):
    agent, _ = build(trusted=GOOD)
    with model.active():
        served, _ = await run(agent, task(1))
        handed, world = await run(agent, task(2, "Don't ship O-2.", "handoff"))
    assert served.metacognition.program == "served" and handed.metacognition.program == "handoff"
    assert handed.metacognition.verdict.accepted and world.shipped == []
    # The same words a gap is measured in, for a request served online.
    assert outcome_of(True, program="served") == "program" and outcome_of(False, program="served") == "wrong"
    assert outcome_of(True, program="read_only", should_act=False) == "read_only"
    assert outcome_of(True, program="read_only") == "lost"  # nothing could act, so nothing was done
    assert outcome_of(True, program="handoff") == "llm" and outcome_of(False) == "llm_wrong"
    assert gap_of([Attempt("c1", "Ship O-1.", True, outcome_of(True, program="served"), 1.0, "accepted"),
                   Attempt("c2", "Ship O-2.", True, outcome_of(True), 1.0, "accepted")], Good()) == pytest.approx(0.2)


async def test_a_run_with_no_decision_tier_reports_no_program(model):
    meta = Metacognition([order_status], verifier=lambda ctx, result: {"success": True, "reason": "agent"},
                         learning="inline", relearn_after=10**9, executor="guarded")
    meta.install("order_status", STATUS)
    agent = Agent(name="Shop", instructions="Ship orders.", tools=[read_order, ship], model="gpt-5.4-mini",
                  metacognition=meta)
    with model.active():
        result, _ = await run(agent, task(1))
    assert result.metacognition.program is None and result.metacognition.memory is None


async def test_a_trusted_program_serves_the_request_with_no_model(model):
    agent, meta = build(trusted=GOOD)
    with model.active():
        result, world = await run(agent, task(1))
    assert model.requests == [] and world.shipped == ["O-1"]
    report = result.metacognition
    assert report.effort == RUNG and report.verdict.accepted and report.offered == [f"{DECISION}@1"]
    assert result.cost.dollars == pytest.approx(100 * 0.042 / 1e6)
    assert meta.store.runs()[0]["effort"] == RUNG


ROUTING = '''NOTE = "Orders ship from the north dock."


async def decide(request, kit):
    return {"handoff": "this one needs the model", "route": "low", "memory": NOTE}
'''


async def test_a_program_can_send_a_request_to_the_llm_at_a_reasoning_level_with_a_memory(model):
    agent, meta = build(trusted=ROUTING)
    with model.active():
        result, _ = await run(agent, {**task(3), "judge": False})
    assert len(model.requests) == 1 and model.requests[0]["reasoning_effort"] == "low"
    assert "north dock" in model.requests[0]["messages"][0]["content"]
    assert result.metacognition.effort == "low"
    assert any(e.startswith("program: routed to the LLM at low") for e in result.metacognition.events)
    assert meta.programs.tier is not None and meta.store.runs()[0]["effort"] == "low"


async def test_an_unknown_route_is_dropped_and_the_request_is_a_plain_handoff(model):
    program = 'async def decide(request, kit):\n    return {"handoff": "x", "route": "maximum", "memory": "NOTE-9"}\n'
    agent, _ = build(trusted=program)
    with model.active():
        result, _ = await run(agent, {**task(4), "judge": False})
    assert model.requests[0].get("reasoning_effort") != "maximum"
    assert "NOTE-9" not in model.requests[0]["messages"][0]["content"]
    assert any(e.startswith("program: handed off") for e in result.metacognition.events)


async def test_each_route_answers_for_itself_at_the_gate():
    async def llm_right(task_, effort, memory):
        return True, "fine"

    async def llm_wrong(task_, effort, memory):
        return False, "the model got it wrong"

    for attempt, passes in ((llm_right, True), (llm_wrong, False)):
        agent, meta = build(trusted=ROUTING, bar=0.5, routed_bar=0.7, route_practice=attempt)
        partly = '''NOTE = "n"


async def decide(request, kit):
    if "104" in request:
        return {"handoff": "not the job"}
    if "101" in request:
        return {"handoff": "hard", "route": "medium", "memory": NOTE}
    ids = kit.find("O-[0-9]+", request)
    return {"action": "ship", "arguments": {"order_id": ids[0]}, "reply": "Shipped."}
'''
        judged = await meta.programs.judge(partly)
        assert (judged.routed, judged.routed_right) == (1, 1 if passes else 0) and judged.wrong == 0
        assert meta.programs._gate(judged).passed is passes
    assert judged.score(2.0) < 1.0  # a routed request is never worth one served with no model


async def test_a_handoff_continues_into_the_agent_in_the_same_run(model):
    agent, _ = build(trusted=GOOD)
    item = {**task(2, "Don't ship O-2."), "judge": False}
    with model.active():
        result, world = await run(agent, item)
    assert len(model.requests) == 1 and world.shipped == []
    assert any(e.startswith("program: handed off") for e in result.metacognition.events)
    assert result.cost.by_source["decider"]["dollars"] == pytest.approx(100 * 0.042 / 1e6)
    assert result.cost.by_source["agent"]["dollars"] > 0


async def test_a_failing_class_is_switched_off_and_goes_back_to_the_llm(model):
    agent, meta = build(trusted=GOOD, min_runs=10**6)
    bad = [{**task(13), "id": f"bad{i}"} for i in range(2)]
    with model.active():
        await run(agent, bad[0])
        last, _ = await run(agent, bad[1])
        assert any(e.startswith("program: switched off ship") for e in last.metacognition.events)
        assert model.requests == []
        result, world = await run(agent, {**task(3), "judge": False})
    assert len(model.requests) == 1 and world.shipped == []  # the scripted LLM just replied
    assert "ship is switched off" in " ".join(result.metacognition.events)
    assert meta.programs.signal()["off"] == ["ship"]


async def test_the_gap_opens_on_llm_spend_and_a_program_that_passes_the_gate_takes_over(model):
    agent, meta = build(GOOD)
    seen = []
    with model.active():
        for n in range(1, 4):
            result, _ = await run(agent, {**task(n), "judge": False})
            seen += result.metacognition.events
        assert any(e.startswith("gap: the LLM spends") for e in seen)
        assert any("wrote decision_program@1; gate 4/4 passed" in e for e in seen)
        calls = len(model.requests)
        served, world = await run(agent, task(9))
    assert len(model.requests) == calls and served.metacognition.effort == RUNG and world.shipped == ["O-9"]
    assert meta.store.trusted(DECISION)["created_by"] == "m0"


async def test_the_gate_refuses_a_program_that_acts_wrongly_and_tries_the_next_mechanism(model):
    agent, meta = build(RECKLESS, GOOD)
    seen = []
    with model.active():
        for n in range(1, 4):
            result, _ = await run(agent, {**task(n), "judge": False})
            seen += result.metacognition.events
    assert any("m0 wrote a program the gate refused: 3/4, 1 wrong" in e for e in seen)
    assert meta.store.trusted(DECISION)["created_by"] == "m1"


async def test_no_gap_without_a_trusted_procedure_or_worthwhile_spend(model):
    agent, meta = build(GOOD, min_gap_usd=1.0)
    with model.active():
        for n in range(1, 5):
            await run(agent, {**task(n), "judge": False})
    assert meta.store.trusted(DECISION) is None
    assert meta.programs.gap(meta.programs.signal()) is None


@pytest.mark.parametrize("method", METHODS)
async def test_either_search_method_finds_a_program_the_gate_admits(method):
    designer = RespondingModel(lambda request: text(mechanism(GOOD)))
    agent, meta = build(RECKLESS, search=ProgramSearch(method=method, population=1, generations=1, seed=3))
    meta.agent = agent
    events: list[str] = []
    with designer.active():
        await meta.programs.improve(events)
    assert any("wrote a program the gate admits" in e for e in events), events
    assert meta.programs.current()["source"] == GOOD


async def test_a_search_is_shown_its_own_cases_and_never_the_gates():
    designer = RespondingModel(lambda request: text(mechanism(RECKLESS)))
    seen = {"c201": task(201, "Don't ship O-201.", "handoff")}
    agent, meta = build(RECKLESS, search=ProgramSearch(population=1, generations=1, seed=3), search_cases=seen)
    meta.agent = agent
    with designer.active():
        await meta.programs.improve([])
    shown = designer.requests[0]["messages"][1]["content"]
    assert "Don't ship O-201." in shown and "O-104" not in shown


SLOPPY = GOOD.replace("< 0.7", "< 0.05")
HELPED = """TOP = 3


def first(items):
    return items[0]


async def decide(request, kit):
    ids = kit.find("O-[0-9]+", request)
    answers = await kit.ask({"request": request}, {"go": {"type": "noul", "instructions": "Ship it?"}})
    if answers["go"]["noul"] < 0.5:
        return {"handoff": "no"}
    return {"action": "ship", "arguments": {"order_id": first(ids)}, "reply": "Shipped."}
"""


@pytest.mark.parametrize("edited", [
    HELPED.replace("< 0.5", "< 0.9"),  # a threshold
    HELPED.replace("Ship it?", "Is shipping this order what the request asks for now?"),  # a question
    HELPED.replace('    return {"action"',  # a route
                   '    if len(ids) > 1:\n        return {"handoff": "which?", "route": "low"}\n    return {"action"'),
    HELPED.replace("async def decide", "NEW = 1\n\n\nasync def decide"),  # a constant
    HELPED.replace('{"handoff": "no"}', '{"handoff": "no", "read_only": True}'),  # a hand-off that can't act
])
def test_an_edit_may_change_how_a_program_judges(edited):
    assert frozen_change(HELPED, edited) is None


@pytest.mark.parametrize("edited", [
    HELPED.replace('"reply": "Shipped."', '"reply": "Sent."'),  # an action
    HELPED.replace('    return {"action"',  # a new action
                   '    if not ids:\n        return {"action": "refund", "arguments": {}}\n    return {"action"'),
    HELPED.replace("return items[0]", "return items[-1]"),  # a helper
    HELPED.replace("TOP = 3", "TOP = 4"),  # a constant
    HELPED.replace('kit.find("O-[0-9]+", request)', 'kit.find("O-[0-9]", request)'),  # how ids are found
    HELPED.replace("first(ids)", "ids[0]"),  # a call to a helper
])
def test_an_edit_may_not_change_what_already_works(edited):
    assert "frozen" in frozen_change(HELPED, edited) or "added an action" in frozen_change(HELPED, edited)


def edit_search(method: str = "dgm") -> ProgramSearch:
    return ProgramSearch(method=method, target="program", population=1, generations=1, seed=3)


@pytest.mark.parametrize("method", METHODS)
async def test_either_method_can_edit_a_written_program_into_one_the_gate_admits(method):
    designer = RespondingModel(lambda request: text(GOOD))
    agent, meta = build(SLOPPY, search=edit_search(method))
    meta.agent = agent
    events: list[str] = []
    with designer.active():
        await meta.programs.improve(events)
    assert any("wrote a program the gate admits" in e for e in events), events
    assert meta.programs.current()["source"].strip() == GOOD.strip()


async def test_a_proposal_that_changes_a_frozen_part_is_sent_back_and_dropped():
    changed = GOOD.replace('"reply": "Shipped."', '"reply": "Sent."')
    designer = RespondingModel(lambda request: text(changed))
    agent, meta = build(SLOPPY, search=edit_search())
    meta.agent = agent
    events: list[str] = []
    with designer.active():
        await meta.programs.improve(events)
    assert "keep this exactly" in designer.requests[1]["messages"][1]["content"]
    assert not any("gate admits" in e and "no design" not in e for e in events), events
    assert any(row["outcome"] == "proposal failed validation" for row in meta.programs.search_log)
    assert meta.programs.current() is None


async def test_an_edit_sees_its_own_failed_requests_and_what_jev_said_and_never_the_gates():
    designer = RespondingModel(lambda request: text(SLOPPY.replace("< 0.05", "< 0.06")))
    seen = {"c201": task(201, "Don't ship O-201.", "handoff")}
    agent, meta = build(SLOPPY, search=edit_search(), search_cases=seen)
    meta.agent = agent
    with designer.active():
        await meta.programs.improve([])
    shown = designer.requests[0]["messages"][1]["content"]
    assert "Don't ship O-201." in shown and '"noul": 0.1' in shown
    assert "O-104" not in shown


async def test_a_switched_off_class_comes_back_when_the_program_it_has_passes_again(model):
    agent, meta = build(RECKLESS, trusted=GOOD)
    meta.agent = agent
    meta.store.put_state("program:ship", {"off": True, "reason": "a tool it calls had broken"})
    events: list[str] = []
    with model.active():
        await meta.programs.improve(events)
    assert any("switched back on: ship" in e for e in events), events
    assert not meta.programs.off("ship")
    assert meta.programs.current()["source"] == GOOD
    assert not any("wrote" in e for e in events), events


async def test_a_program_that_still_fails_stays_off_and_a_new_one_is_written(model):
    agent, meta = build(GOOD, trusted=RECKLESS)
    meta.agent = agent
    meta.store.put_state("program:ship", {"off": True})
    events: list[str] = []
    with model.active():
        await meta.programs.improve(events)
    assert any("still refused" in e for e in events), events
    assert any("m0 wrote" in e for e in events), events
    assert meta.programs.current()["source"] == GOOD
    assert not meta.programs.off("ship")


async def test_nothing_is_retested_when_no_class_is_off(model):
    agent, meta = build(GOOD, trusted=RECKLESS)
    meta.agent = agent
    events: list[str] = []
    with model.active():
        await meta.programs.improve(events)
    assert not any("re-tested" in e for e in events), events


READ_ONLY = GOOD.replace('{"handoff": "not sure"}', '{"handoff": "not sure", "read_only": True}')
NEAR_MISS = task(104, "Don't ship O-104.", "handoff")


async def test_a_read_only_handoff_leaves_the_agent_no_way_to_act():
    tries = ScriptedModel([call("ship", {"order_id": "O-104"}), text("A person needs to handle that.")])
    agent, _ = build(trusted=READ_ONLY)
    with tries.active():
        result, world = await run(agent, NEAR_MISS)
    offered = {t["function"]["name"] for t in tries.requests[0]["tools"]}
    assert "ship" not in offered and "read_order" in offered
    assert "can't take any action" in tries.requests[0]["messages"][0]["content"]
    assert world.shipped == []
    assert any(e.startswith("program: handed off read-only") for e in result.metacognition.events)


@pytest.mark.parametrize(("program", "share"), [(READ_ONLY, 0.0), (GOOD, 1.0)])
async def test_only_a_handoff_of_work_counts_as_handing_off_too_much(model, program, share):
    agent, meta = build(trusted=program)
    with model.active():
        await run(agent, NEAR_MISS)
    assert meta.programs.signal()["handoff_share"] == share


async def test_with_read_only_handoffs_a_near_miss_must_be_handed_off_read_only():
    _, meta = build(read_only_handoffs=True)
    plain, safe = await meta.programs.judge(GOOD), await meta.programs.judge(READ_ONLY)
    assert not meta.programs._gate(plain).passed and "must be handed off read-only" in plain.failures[0]
    assert meta.programs._gate(safe).passed and (safe.accepted, safe.wrong) == (4, 0)


async def test_real_work_sent_read_only_is_lost_but_is_not_a_wrong_action():
    everything = 'async def decide(request, kit):\n    return {"handoff": "no", "read_only": True}\n'
    _, meta = build()
    judged = await meta.programs.judge(everything)
    assert (judged.accepted, judged.wrong, judged.handoffs) == (1, 0, 3)
    assert "nothing could serve it" in judged.failures[0]


@pytest.mark.parametrize(("long_running", "left"), [(False, 1), (True, 0)])
async def test_a_gap_that_closes_takes_its_attempts_with_it_in_a_long_running_tier(model, long_running, left):
    agent, meta = build(GOOD, long_running=long_running)
    with model.active():
        for n in range(1, 4):  # three runs by the model open the gap, and the mechanism's program is installed
            await run(agent, task(n))
    assert meta.programs.current() is not None
    assert meta.store.state("decision:loop").get("attempts", 0) == left


@pytest.mark.parametrize(("long_running", "back_on"), [(False, False), (True, True)])
async def test_a_switched_off_action_can_be_tested_again_after_learning_used_its_attempts(long_running, back_on):
    """A re-test asks no builder. Blocked by a lifetime of attempts, an action stayed off after its program was
    right again."""
    agent, meta = build(trusted=GOOD, long_running=long_running, max_attempts=2)
    meta.agent = agent
    meta.store.put_state("decision:loop", {"attempts": 2, "next_attempt": 0})
    meta.store.put_state("program:ship", {"off": True, "reason": "two wrong actions"})
    events: list[str] = []
    await meta.programs.maybe_improve(events)
    assert (not meta.programs.off("ship")) is back_on
    assert any("switched back on" in event for event in events) is back_on


@pytest.mark.parametrize(("long_running", "kept"), [(False, True), (True, False)])
async def test_a_wrong_action_from_before_a_procedure_was_relearned_is_not_a_gate_case(model, long_running, kept):
    """Its world is gone. Kept as a case, it failed every later re-test of a program that was right again."""
    agent, meta = build(trusted=RECKLESS, long_running=long_running)
    with model.active():
        result, _ = await run(agent, task(13))  # the reckless program ships an order that can't ship
    assert result.metacognition.program == "served" and not result.metacognition.verdict.accepted
    assert "c13" in meta.programs._cases()
    meta.install("order_status", STATUS + "# relearned\n")
    assert ("c13" in meta.programs._cases()) is kept


@pytest.mark.parametrize(("long_running", "function_holds", "blamed"),
                         [(True, False, True), (True, True, False), (False, False, False)])
async def test_a_failed_retest_checks_the_procedures_before_a_builder_rewrites_the_program(long_running,
                                                                                          function_holds, blamed):
    """A program stands on its procedures. Rewritten while one of them was stale, it cost builder calls that could
    fix nothing, night after night, and the stale procedure went on serving."""
    agent, meta = build(GOOD, trusted=RECKLESS, long_running=long_running)
    meta.agent = agent

    async def practice(source: str, case_id: str) -> tuple[float, str, str]:
        return (1.0 if function_holds else 0.0), "tried", ""

    meta.practice, meta.gate = practice, Gate(cases=["g1", "g2"])
    meta.store.put_state("program:ship", {"off": True, "reason": "two wrong actions"})
    events: list[str] = []
    await meta.programs.maybe_improve(events)
    assert any("still refused" in event for event in events)  # the reckless program ships what it should not
    assert (meta.store.trusted("order_status") is None) is blamed
    assert any("quarantined order_status@1 (its own gate now passes 0/2)" in event for event in events) is blamed
    assert any(event.startswith("program: m0 wrote") for event in events) is not blamed
    assert meta.store.state("decision:loop").get("attempts", 0) == (0 if long_running else 1)


@pytest.mark.parametrize(("long_running", "waits"), [(True, True), (False, False)])
async def test_a_long_running_tier_waits_while_a_procedure_it_stands_on_is_relearned(long_running, waits):
    agent, meta = build(GOOD, trusted=RECKLESS, long_running=long_running, procedures=(order_status, order_weight))
    meta.agent = agent
    meta.install("order_weight", WEIGHT)
    meta.store.put_state("program:ship", {"off": True, "reason": "two wrong actions"})
    meta.quarantine("order_status", "its own gate now passes 0/2")
    events: list[str] = []
    await meta.programs.maybe_improve(events)
    assert (events == []) is waits
    meta.install("order_status", STATUS + "# relearned\n")
    await meta.programs.maybe_improve(events)
    assert any(event.startswith("program: m0 wrote") for event in events)  # relearned, so it is the program's turn


@pytest.mark.parametrize(("long_running", "by_day", "back_on"),
                         [(True, True, True), (True, False, False), (False, True, False)])
async def test_a_relearned_procedure_has_a_switched_off_action_tested_again_at_once(long_running, by_day, back_on):
    """Waiting for the night left the fast path off for the rest of every day on which something changed."""
    agent, meta = build(trusted=GOOD, long_running=long_running, retest_by_day=by_day)
    meta.agent = agent
    meta.store.put_state("program:ship", {"off": True, "reason": "two wrong actions"})
    events: list[str] = []
    await meta.programs.relearned("order_status", events)
    assert (not meta.programs.off("ship")) is back_on
    assert any("switched back on" in event for event in events) is back_on
    assert meta.store.state("decision:loop").get("attempts", 0) == 0  # a re-test uses no attempt


@pytest.mark.parametrize(("long_running", "keep", "quarantined", "effort"),
                         [(True, None, True, "high"), (False, None, True, "low"), (True, None, False, "low"),
                          (False, True, True, "high"), (True, False, True, "low"), (False, True, False, "low")])
async def test_a_route_does_not_lower_the_reasoning_raised_for_a_quarantined_procedure(model, long_running, keep,
                                                                                      quarantined, effort):
    """In the recorded week three of a day's six failures came from a route that kept the agent at low reasoning
    while the function it relied on was quarantined. `keep_raised_effort` sets this alone; unset, it follows
    `long_running`."""
    agent, meta = build(trusted=ROUTING, long_running=long_running, keep_raised_effort=keep)
    meta.controller = Controller({"memory": "none", "no_memory": "high"})
    if quarantined:
        meta.quarantine("order_status")
    with model.active():
        result, _ = await run(agent, {**task(3), "judge": False})
    assert result.metacognition.effort == effort and model.requests[0]["reasoning_effort"] == effort


async def test_a_read_only_run_that_fails_says_nothing_about_a_procedure():
    """Real work a program sent on read-only can't be done, and that isn't the fault of a procedure the agent called."""
    everything = 'async def decide(request, kit):\n    return {"handoff": "no", "read_only": True}\n'
    agent, meta = build(trusted=everything)
    for n in (1, 2, 3):  # two rejections in the last four judged uses would quarantine the procedure
        tries = ScriptedModel([call("order_status", {"order_id": f"O-{n}"}), text("I can't ship anything from here.")])
        with tries.active():
            result, world = await run(agent, task(n))
        report = result.metacognition
        assert report.program == "read_only" and not report.verdict.accepted and world.shipped == []
        assert report.called == ["order_status@1"]
        assert not any(event.startswith("detector: quarantined") for event in report.events)
    assert meta.store.trusted("order_status") is not None


async def test_read_only_means_nothing_on_an_action():
    source = 'async def decide(request, kit):\n    return {"action": "ship", "read_only": True}\n'
    assert "read_only" not in await run_program(source, "x", Kit({}, {}))


CAREFUL = """async def decide(request, kit):
    ids = kit.find("O-[0-9]+", request)
    found = await kit.procedure("order_status", order_id=ids[0])
    if found["status"] != "ok":
        return {"handoff": "held"}
    return {"action": "ship", "arguments": {"order_id": ids[0]}, "reply": "Shipped."}
"""


async def test_the_gate_calls_the_procedures_the_agent_has_now():
    _, meta = build()
    assert (await meta.programs.judge(CAREFUL, ["c101"])).accepted == 1
    meta.install("order_status", 'def order_status(api, order_id: str) -> dict:\n    return {"status": "held"}\n')
    after = await meta.programs.judge(CAREFUL, ["c101"])
    assert (after.accepted, after.handoffs) == (0, 1)


SHIP_IT = """async def decide(request, kit):
    ids = kit.find("O-[0-9]+", request)
    return {"action": "ship", "arguments": {"order_id": ids[0]}, "reply": "Shipped."}
"""
PICK = """async def decide(request, kit):
    return {"serve": "hold" if "Don't" in request else "ship_it"}
"""


def component(*memories: tuple[Any, ...], router: str = PICK) -> str:
    """A learning component that stores `memories` (name, description, body, route, read_only) and returns `router`."""
    return (f"ROUTER = {router!r}\nSTORED = {list(memories)!r}\n\n\nasync def learn(lab):\n"
            "    for name, description, body, route, read_only in STORED:\n"
            "        lab.remember(name, description, body, route=route, read_only=read_only)\n"
            "    return ROUTER\n")


SHIPS = ("ship_it", "Requests to ship an order.", SHIP_IT, "program", False)
HOLDS = ("hold", "Requests not to ship something.", "Say a person will look at it.", "none", True)
ALL_LLM = ("ship_it", "Requests to ship an order.", "Ids look like O-1.", "none", False)


async def routed_right(task_, effort, memory):
    return True, "fine"


async def installed(*memories: tuple[Any, ...], router: str = PICK,
                    **overrides: Any) -> tuple[Agent, Metacognition, list[str]]:
    seed = [("seed", component(*memories, router=router))]
    agent, meta = build(mechanisms=seed, route_practice=routed_right, **overrides)
    meta.agent = agent
    events: list[str] = []
    model = RespondingModel(lambda request: text("unused"))
    with model.active():
        await meta.programs.improve(events)
    return agent, meta, events


async def test_an_admitted_component_stores_each_memory_with_its_route():
    _, meta, events = await installed(SHIPS, HOLDS)
    assert any("memories ship_it (program), hold (none, read-only)" in e for e in events), events
    router = meta.programs.current()["source"]
    assert "MEMORIES" not in router and '"serve"' in router
    assert meta.store.trusted("memory:ship_it")["source"] == SHIP_IT
    assert meta.store.trusted("memory:hold")["info"] == {"description": HOLDS[1], "route": "none", "read_only": True}


async def test_a_decision_memory_serves_with_no_model_and_a_read_only_one_goes_to_the_llm(model):
    agent, _, _ = await installed(SHIPS, HOLDS)
    with model.active():
        result, world = await run(agent, task(1))
    assert model.requests == [] and world.shipped == ["O-1"]
    assert "program: used memory ship_it" in result.metacognition.events
    with model.active():
        result, world = await run(agent, NEAR_MISS)
    assert world.shipped == [] and len(model.requests) == 1
    assert "ship" not in {t["function"]["name"] for t in model.requests[0]["tools"]}
    assert "Say a person will look at it." in model.requests[0]["messages"][0]["content"]
    assert "program: used memory hold" in result.metacognition.events


async def test_a_memory_routed_to_the_llm_runs_at_its_reasoning_level(model):
    notes = ("ship_it", "Requests to ship an order.", "Ids look like O-1.", "medium", False)
    served = """async def decide(request, kit):
    if "Don't" in request:
        return {"handoff": "no", "read_only": True}
    if "101" in request:
        return {"serve": "direct"}
    return {"serve": "ship_it"}
"""
    direct = ("direct", "Ship directly.", SHIP_IT, "program", False)
    agent, meta, _ = await installed(notes, direct, router=served, bar=0.5)
    with model.active():
        result, _ = await run(agent, {**task(2), "judge": False})
    assert len(model.requests) == 1 and model.requests[0]["reasoning_effort"] == "medium"
    assert "Ids look like O-1." in model.requests[0]["messages"][0]["content"]
    assert result.metacognition.effort == "medium"


async def test_recall_finds_the_closest_memory_in_the_tier():
    router = """async def decide(request, kit):
    found = await kit.recall(request, k=1)
    return {"serve": found[0]["name"]}
"""
    packed = pack_memories(router, [
        {"name": "ship_it", "description": "ship an order out", "route": "program", "read_only": False, "body": SHIP_IT},
        {"name": "other", "description": "questions about invoices and payments", "route": "none", "read_only": True,
         "body": "n"}])
    _, meta = build()
    judged = await meta.programs.judge(packed, ["c101"])
    assert judged.accepted == 1 and judged.acted == 1


async def test_a_component_that_serves_nothing_itself_is_not_admitted():
    _, meta, events = await installed(ALL_LLM, HOLDS)
    assert meta.programs.current() is None
    assert any("the gate refused" in e for e in events), events


@pytest.mark.parametrize("method", METHODS)
async def test_a_search_can_rewrite_a_component_from_the_llm_to_a_decision_memory(method):
    variant = component(SHIPS, HOLDS)
    designer = RespondingModel(lambda request: text(variant))
    agent, meta = build(mechanisms=[("seed", component(ALL_LLM, HOLDS))], route_practice=routed_right,
                        search=ProgramSearch(method=method, population=1, generations=1, seed=3))
    meta.agent = agent
    events: list[str] = []
    with designer.active():
        await meta.programs.improve(events)
    assert any("wrote a program the gate admits" in e for e in events), events
    assert meta.store.trusted("memory:ship_it")["info"]["route"] == "program"
    assert meta.store.trusted("memory:ship_it")["source"] == SHIP_IT


async def test_a_variant_starts_from_what_its_parent_stored():
    moves = (f"ROUTER = {PICK!r}\n\n\nasync def learn(lab):\n    for memory in lab.memories:\n"
             f"        lab.remember(**memory)\n"
             f"    lab.remember('ship_it', 'Requests to ship an order.', {SHIP_IT!r}, route='program')\n"
             "    return ROUTER\n")
    designer = RespondingModel(lambda request: text(moves))
    agent, meta = build(mechanisms=[("seed", component(ALL_LLM, HOLDS))], route_practice=routed_right,
                        search=ProgramSearch(population=1, generations=1, seed=3))
    meta.agent = agent
    events: list[str] = []
    with designer.active():
        await meta.programs.improve(events)
    assert any("gate admits" in e and "no design" not in e for e in events), events
    assert meta.store.trusted("memory:hold")["source"] == HOLDS[2]  # inherited, never written by the variant
    assert meta.store.trusted("memory:ship_it")["info"]["route"] == "program"


async def test_a_variant_reaches_the_gate_only_if_it_clears_the_bar_on_its_own_cases():
    designer = RespondingModel(lambda request: text(mechanism(RECKLESS)))
    seen = {"c201": task(201, "Don't ship O-201.", "handoff")}
    agent, meta = build(RECKLESS, search=ProgramSearch(population=1, generations=1, seed=3), search_cases=seen)
    meta.agent = agent
    with designer.active():
        await meta.programs.improve([])
    rows = [row for row in meta.programs.search_log if row.get("outcome", "").startswith("refused")]
    assert rows and all(row["outcome"] == "refused on its own cases" and "right" not in row for row in rows)


def test_an_edit_may_rewrite_how_a_memory_is_retrieved_but_not_what_it_does():
    def packed(body: str = SHIP_IT, description: str = "ship", route: str = "program") -> str:
        return pack_memories(PICK, [{"name": "ship_it", "description": description, "route": route, "read_only": False,
                                     "body": body}])

    assert frozen_change(packed(), packed(description="Only a new shipment.")) is None
    assert frozen_change(packed(), packed(route="none", body="Notes.")) is None
    assert "in the memory ship_it" in frozen_change(packed(), packed(body=SHIP_IT.replace("Shipped.", "Sent.")))
    other = pack_memories(PICK, [{"name": "z", "description": "d", "route": "none", "body": "b"}])
    assert "removed the memory" in frozen_change(packed(), other)


def test_memories_survive_being_packed_into_a_program():
    memories = [{"name": "a", "description": "d", "route": "program", "read_only": False, "body": SHIP_IT},
                {"name": "b", "description": "has 'quotes' and a \\ slash", "route": "low", "read_only": True,
                 "body": 'ends in "'}]
    router, unpacked = unpack_memories(pack_memories(PICK, memories))
    assert router == PICK and unpacked == memories
    assert unpack_memories(PICK) == (PICK, [])


SEEDS = Path(helix.__file__).parent / "metacognition" / "programs"
KINDS = json.dumps([
    {"name": "ship_it", "description": "Requests to ship an order.", "guidance": "Ids look like O-1.", "acts": True},
    {"name": "hold", "description": "Requests not to ship something.", "guidance": "Answer only.", "acts": False}])


class KindJev(FakeJev):
    async def ask_with_usage(self, state: Any, questions: dict[str, Any]) -> tuple[dict[str, Any], DecisionUsage]:
        if "kind" not in questions:
            return await super().ask_with_usage(state, questions)
        kind = "hold" if "Don't" in state["request"] else "ship_it"
        answer = Answer("choice", kind, 0.9, {kind: 0.9})
        return {"kind": answer}, DecisionUsage(calls=1, input_tokens=100, dollars=0.0, seconds=0.01)


def replying(*replies: str) -> ScriptedModel:
    return ScriptedModel([text(reply) for reply in replies])


async def test_the_llm_seed_stores_guidance_and_nothing_that_serves_without_the_model():
    source = (SEEDS / "component_llm.py").read_text(encoding="utf-8")
    assert "kit.ask" not in source and '"program"' not in source
    agent, meta = build(mechanisms=[("component_llm", source)], route_practice=routed_right)
    meta.agent = agent
    events: list[str] = []
    writer = replying(KINDS)
    with writer.active():
        await meta.programs.improve(events)
    assert meta.programs.current() is None and any("the gate refused" in e for e in events), events
    assert len(writer.requests) == 1


async def test_the_control_seed_writes_a_decision_memory_for_each_kind_that_acts():
    source = (SEEDS / "component_judgment.py").read_text(encoding="utf-8")
    desk_tier = tier(mechanisms=[("component_judgment", source)])
    desk_tier.decider = KindJev()
    meta = Metacognition([order_status], verifier=lambda ctx, result: ctx.context.check(), learning="inline",
                         relearn_after=10**9, programs=desk_tier, executor="guarded")
    meta.install("order_status", STATUS)
    agent = Agent(name="Shop", instructions="Ship orders.", tools=[read_order, ship], model="gpt-5.4-mini",
                  metacognition=meta)
    meta.agent = agent
    events: list[str] = []
    with replying(KINDS, SHIP_IT).active():
        await meta.programs.improve(events)
    assert any("memories ship_it (program), hold (none, read-only)" in e for e in events), events
    assert meta.programs.current() is not None
    assert meta.store.trusted("memory:ship_it")["source"].strip() == SHIP_IT.strip()


async def test_a_search_learns_only_how_many_gate_requests_went_wrong():
    designer = RespondingModel(lambda request: text(mechanism(RECKLESS)))
    agent, meta = build(RECKLESS, search=ProgramSearch(population=1, generations=1, seed=3), search_cases={"c201": task(201)})
    meta.agent = agent
    with designer.active():
        await meta.programs.improve([])
    shown = designer.requests[0]["messages"][1]["content"]
    assert "On 4 other requests" in shown and "3 right, with 1 wrong actions" in shown
    assert "O-104" not in shown
    seed_row = meta.programs.search_log[0]
    assert seed_row["outcome"] == "refused by gate" and seed_row["search_score"] < 1.0  # 1.0 on what it can see


async def test_a_later_search_starts_from_the_memories_kept_in_the_archive(tmp_path):
    counts = (f"ROUTER = {PICK!r}\n\n\nasync def learn(lab):\n"
              "    lab.remember('count', str(len(lab.memories)), 'x', route='none', read_only=True)\n"
              "    return ROUTER\n")
    designer = RespondingModel(lambda request: text(counts))
    spec = ProgramSearch(archive=str(tmp_path / "a.sqlite"), population=1, generations=1, seed=3, max_searches=2)
    agent, meta = build(mechanisms=[("seed", component(ALL_LLM, HOLDS))], route_practice=routed_right, search=spec)
    meta.agent = agent
    with designer.active():
        await meta.programs.improve([])
        await meta.programs.improve([])
    last = unpack_memories(meta.programs.search_log[-1]["program"])[1][0]
    assert last["name"] == "count" and last["description"] in ("1", "2")


def attempt(outcome: str, memory: str | None = None, should_act: bool = True) -> Attempt:
    return Attempt("c1", "Ship O-1.", should_act, outcome, 0.5, "because", memory, dollars=0.002, seconds=2.0)


def test_the_gap_is_the_mean_distance_from_good():
    good = Good(wrong_action=3.0)
    assert gap_of([attempt("program"), attempt("read_only", should_act=False)], good) == 0.0
    assert gap_of([attempt("program"), attempt("llm")], good) == pytest.approx(0.2)
    assert gap_of([attempt("wrong"), attempt("program")], good) == pytest.approx(1.5)
    assert gap_of([attempt("handoff"), attempt("lost"), attempt("llm_wrong")], good) == pytest.approx(1.0)


def test_the_gap_report_says_where_the_gap_is_and_shows_every_request():
    tried = [attempt("llm", "ship_it"), attempt("llm", "ship_it"), attempt("program", "other"), attempt("wrong", "hold")]
    written = report(tried, Good(wrong_action=3.0))
    assert "Gap from good: 0.95 over 4 requests" in written
    assert "- wrong actions: 1 requests, 0.75 of the gap" in written
    assert "- right, but served by the language model: 2 requests, 0.20 of the gap" in written
    assert written.index("- hold: 0.75") < written.index("- ship_it: 0.20")
    assert written.count("'Ship O-1.'") == 4


async def test_sharing_every_test_run_shows_the_rewriter_the_gates_requests_too():
    designer = RespondingModel(lambda request: text(mechanism(RECKLESS)))
    spec = ProgramSearch(population=1, generations=1, seed=3, share_all=True)
    agent, meta = build(RECKLESS, search=spec, search_cases={"c201": task(201)})
    meta.agent = agent
    with designer.active():
        await meta.programs.improve([])
    shown = designer.requests[0]["messages"][1]["content"]
    assert "Gap from good" in shown and "Don't ship O-104." in shown and "O-201" in shown


async def test_the_gap_measurement_is_a_helix_signal():
    agent, meta = build(RECKLESS)
    meta.agent = agent
    signal = ServiceGap(meta.programs, Good())
    assert isinstance(signal, Signal)
    measured = await signal.measure(GOOD)
    assert measured.signal_id == "service_gap" and measured.score == pytest.approx(1.0) and measured.raw_value == 0.0
    assert not measured.triggered


async def test_a_search_can_stop_at_the_first_variant_the_gate_admits():
    designer = RespondingModel(lambda request: text(mechanism(GOOD)))
    spec = ProgramSearch(population=4, generations=1, seed=3, stop_when_admitted=True)
    agent, meta = build(RECKLESS, search=spec)
    meta.agent = agent
    events: list[str] = []
    with designer.active():
        await meta.programs.improve(events)
    assert any("wrote a program the gate admits" in e for e in events), events
    assert len(designer.requests) == 1


PRACTICES_ON_SHARED = (f"ROUTER = {PICK!r}\n\n\nasync def learn(lab):\n"
                       "    lab.remember('hold', 'x', 'x', route='none', read_only=True)\n"
                       "    lab.remember('ship_it', 'x', 'x', route='none')\n"
                       "    tried = []\n"
                       "    for test in lab.tests:\n"
                       "        outcome = await lab.practice(ROUTER, test['id'])\n"
                       "        tried.append([test['id'], lab.request(test['id']), test['should_act'], outcome[0]])\n"
                       "    lab.remember('ship_it', str(tried), 'x', route='none')\n"
                       "    return ROUTER\n")


@pytest.mark.parametrize("share_all", [True, False])
async def test_a_variant_can_practice_on_the_requests_a_search_shares(share_all):
    designer = RespondingModel(lambda request: text(PRACTICES_ON_SHARED))
    spec = ProgramSearch(population=1, generations=1, seed=3, share_all=share_all)
    agent, meta = build(mechanisms=[("seed", component(ALL_LLM, HOLDS))], route_practice=routed_right, search=spec,
                        search_cases={"c201": task(201)})
    meta.agent = agent
    with designer.active():
        await meta.programs.improve([])
    stored = {m["name"]: m["description"] for m in unpack_memories(meta.programs.search_log[-1]["program"])[1]}
    assert "['c201', 'Ship O-201.', True, 0.6]" in stored["ship_it"]
    assert ("['c104', \"Don't ship O-104.\", False, 1.0]" in stored["ship_it"]) == share_all


async def test_a_component_run_again_gets_the_requests_a_search_shares_and_never_the_gates():
    """The first paid search found a component that practices on `lab.tests`. Run again without a search, it had
    none."""
    agent, meta = build(mechanisms=[("found", PRACTICES_ON_SHARED)], route_practice=routed_right,
                        search_cases={"c201": task(201)})
    meta.agent = agent
    trial, _ = await meta.programs._learn(meta.programs._problem(), "found", PRACTICES_ON_SHARED)
    stored = {m["name"]: m["description"] for m in unpack_memories(trial.source)[1]}
    assert "['c201', 'Ship O-201.', True, 0.6]" in stored["ship_it"]
    assert "c104" not in stored["ship_it"]  # a gate request
    agent, meta = build(mechanisms=[("found", PRACTICES_ON_SHARED)], route_practice=routed_right)
    meta.agent = agent
    trial, _ = await meta.programs._learn(meta.programs._problem(), "found", PRACTICES_ON_SHARED)
    stored = {m["name"]: m["description"] for m in unpack_memories(trial.source)[1]}
    assert stored["ship_it"] == "[]"  # with no search requests there is nothing to share but the gate's, so nothing


def test_a_variant_practices_on_what_its_gap_report_shows():
    _, meta = build(RECKLESS, search_cases={"c201": task(201)})
    assert [test["id"] for test in meta.programs._shared_tests(False)] == ["c201"]
    assert {"c201", "c104"} <= {test["id"] for test in meta.programs._shared_tests(True)}
    _, meta = build(RECKLESS)  # with no search cases of its own, the report shows the gate's
    assert {test["id"] for test in meta.programs._shared_tests(False)} == set(LABELED)


async def test_a_program_the_guard_refuses_never_ran_so_it_took_no_wrong_action():
    agent, meta = build(RECKLESS)
    meta.agent = agent
    judged = await meta.programs.judge("async def decide(request, kit:\n")
    assert judged.wrong == 0 and judged.acted == 0 and not meta.programs._gate(judged).passed
    assert {a.outcome for a in judged.attempts} == {"refused"}
    assert gap_of(judged.attempts, Good()) == 1.0
    written = report(judged.attempts, Good())
    assert "No request ran" in written and "does not parse" in written and "Ship O-" not in written


@pytest.mark.parametrize(("source", "refusal"), [
    ("async def learn(lab):\n    return repr(lab)\n", "'repr' is not defined"),
    ("PROMPT = 'Return {\"serve\": name}.'\n\n\nasync def learn(lab):\n    return lab.fill(PROMPT, name='x')\n",
     "PROMPT can't be filled"),
    ("async def learn(lab):\n    return lab.fill('a {b}')\n", "leaves out b"),
])
def test_a_search_refuses_what_would_stop_a_mechanism_when_it_ran(source, refusal):
    with pytest.raises(UnsafeMechanism, match=refusal):
        check_runnable(source)


@pytest.mark.parametrize("source", [
    "PROMPT = 'Return {{\"serve\": {name}}}.'\n\n\nasync def learn(lab):\n    return lab.fill(PROMPT, name='x')\n",
    "async def learn(lab):\n    values = {'b': 1}\n    return lab.fill('a {b}', **values)\n",
    "async def learn(lab):\n    text = 'a {\"b\"}'\n    return text.replace('a', lab.contract)\n",
    "async def learn(lab):\n    items = [x for x in lab.stream()]\n    return str(len(items))\n",
])
def test_a_search_passes_what_can_run(source):
    check_runnable(source)


async def test_a_rewrite_that_could_not_run_goes_back_to_the_rewriter_with_the_reason():
    replies = iter([text("async def learn(lab):\n    return repr(lab)\n"), text(mechanism(GOOD))])
    designer = RespondingModel(lambda request: next(replies))
    spec = ProgramSearch(population=1, generations=1, seed=3, stop_when_admitted=True)
    agent, meta = build(RECKLESS, search=spec)
    meta.agent = agent
    events: list[str] = []
    with designer.active():
        await meta.programs.improve(events)
    assert len(designer.requests) == 2 and "'repr' is not defined" in designer.requests[1]["messages"][1]["content"]
    assert any("wrote a program the gate admits" in e for e in events), events


async def test_a_hidden_gate_says_which_memory_made_the_wrong_actions():
    always = 'async def decide(request, kit):\n    return {"serve": "ship_it"}\n'
    designer = RespondingModel(lambda request: text(component(SHIPS, HOLDS, router=always)))
    spec = ProgramSearch(population=1, generations=1, seed=3)
    agent, meta = build(mechanisms=[("seed", component(SHIPS, HOLDS, router=always))], route_practice=routed_right, search=spec,
                        search_cases={"c201": task(201)})
    meta.agent = agent
    with designer.active():
        await meta.programs.improve([])
    shown = designer.requests[0]["messages"][1]["content"]
    assert "The wrong actions came from: 1 from ship_it." in shown and "O-104" not in shown


def priced(costs: dict[str, float], right_at: set[str] | None = None) -> Any:
    """The LLM on a request: what a run costs at each reasoning level, and the levels that get it right (all, if None)."""

    async def attempt(task_: dict[str, Any], effort: str, memory: str | None) -> tuple[bool, str, float, float]:
        accepted = right_at is None or effort in right_at
        return accepted, "fine" if accepted else "wrong", costs[effort], 0.1

    return attempt


async def searched_on_the_llm(route_practice: Any, *, lower_reasoning: bool = True) -> tuple[Metacognition, RespondingModel, list[str]]:
    designer = RespondingModel(lambda request: text(component(ALL_LLM, HOLDS)))
    spec = ProgramSearch(population=1, generations=1, seed=3, lower_reasoning=lower_reasoning)
    agent, meta = build(mechanisms=[("seed", component(ALL_LLM, HOLDS))], route_practice=route_practice, search=spec,
                        search_cases={"c201": task(201)})
    meta.agent = agent
    meta.controller = Controller({"memory": "high", "no_memory": "high"})
    events: list[str] = []
    with designer.active():
        await meta.programs.improve(events)
    return meta, designer, events


@pytest.mark.parametrize(("costs", "right_at", "admitted"), [
    ({"high": 0.01, "none": 0.001}, None, True),  # as accurate, and cheaper
    ({"high": 0.001, "none": 0.001}, None, False),  # as accurate, but no cheaper
    ({"high": 0.01, "none": 0.001}, {"high"}, False),  # cheaper, but less accurate
])
async def test_a_variant_on_the_llm_must_match_what_is_running_and_cost_less(costs, right_at, admitted):
    meta, designer, events = await searched_on_the_llm(priced(costs, right_at))
    assert (meta.programs.current() is not None) == admitted, events
    assert any("what's running now got" in e for e in events), events
    assert "as what is running now and costs less" in designer.requests[0]["messages"][0]["content"]


async def test_without_the_setting_a_variant_on_the_llm_is_still_refused():
    meta, designer, _ = await searched_on_the_llm(priced({"high": 0.01, "none": 0.001}), lower_reasoning=False)
    assert meta.programs.current() is None
    assert "can't be admitted" in designer.requests[0]["messages"][0]["content"]


async def test_with_no_model_named_a_search_builds_code_with_the_strong_model():
    from helix.metacognition.model_settings import BUILDER_MODEL

    writes = f"PROGRAM = {RECKLESS!r}\n\n\nasync def learn(lab):\n    await lab.ask('hello')\n    return PROGRAM\n"
    designer = RespondingModel(lambda request: text(writes))
    spec = ProgramSearch(population=1, generations=1, seed=3)
    agent, meta = build(mechanisms=[("seed", writes)], search=spec)
    meta.agent = agent
    meta.programs.tier.writer_model = None
    with designer.active():
        await meta.programs.improve([])
    assert len(designer.requests) >= 3 and {request["model"] for request in designer.requests} == {BUILDER_MODEL}
