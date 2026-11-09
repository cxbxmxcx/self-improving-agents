"""The decision tier: signal, gap, gate and per-class switch, on a scripted model and a fake Jev (no key)."""

from __future__ import annotations

from typing import Any

import pytest
from helix.deciders import Answer, DecisionUsage
from helix.memory.learning_mechanisms import Contract
from helix.metacognition import Metacognition, procedure
from helix.metacognition.decide import KIT_GUIDE, RETURNS, Desk, frozen_change
from helix.metacognition.decisions import DECISION, RUNG, ProgramSearch, ProgramTier
from helix.search.methods import METHODS

from metacog import Agent, RunConfig, RunContextWrapper, Runner, function_tool
from metacog.testing import RespondingModel, text


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


STATUS = 'def order_status(api, order_id: str) -> dict:\n    return {"status": "ok"}\n'
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
    return ProgramTier(desk=desk, decider=FakeJev(), contract=contract, labeled=LABELED,
                       mechanisms=[(f"m{i}", mechanism(p)) for i, p in enumerate(programs)],
                       writer_model="gpt-5.4-mini", **settings)


def build(*programs: str, trusted: str | None = None, **overrides: Any) -> tuple[Agent, Metacognition]:
    def verify(ctx: RunContextWrapper[World], result: Any) -> dict[str, Any]:
        return ctx.context.check() if ctx.context.task.get("judge", True) else {"success": True, "reason": "agent"}

    meta = Metacognition([order_status], verifier=verify, learning="inline", relearn_after=10**9,
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
    assert "c201" in shown and "c104" not in shown


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
    HELPED.replace('    return {"action"',
                   '    if len(ids) > 1:\n        return {"handoff": "which?", "route": "low"}\n    return {"action"'),  # a route
    HELPED.replace("async def decide", "NEW = 1\n\n\nasync def decide"),  # a constant
])
def test_an_edit_may_change_how_a_program_judges(edited):
    assert frozen_change(HELPED, edited) is None


@pytest.mark.parametrize("edited", [
    HELPED.replace('"reply": "Shipped."', '"reply": "Sent."'),  # an action
    HELPED.replace('    return {"action"',
                   '    if not ids:\n        return {"action": "refund", "arguments": {}}\n    return {"action"'),  # a new action
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
