"""What a long stream and a loaded server need from the harness (scripted models, no key).

Each test here answers a fault that a long scripted stream exposed: a limit that lasted a lifetime, a store that only
grew, a gate case from a world that had changed, an outage taken for a stale function, and a learner whose cost grew
with its backlog.
"""

from __future__ import annotations

from typing import Any

import pytest
from helix.memory.learning_mechanisms import Episode
from helix.metacognition import Gate, LearningBudget, Metacognition, MetaStore, Verdict, procedure
from helix.metacognition.exceptions import Unavailable

from metacog import Agent, RunConfig, RunContextWrapper, Runner, function_tool
from metacog.testing import RespondingModel, ScriptedModel, call, text


def record(store: MetaStore, n: int, *, right: bool = True, effort: str = "none", judged: bool = True,
           matched: tuple[str, ...] = ()) -> None:
    steps = [{"tool": "read", "arguments": {}, "result": "x" * 4000}]
    store.record(run_id=f"r{n}", case_id=f"c{n}", case_data={"order": n, "week": "y" * 4000}, request=f"request {n}",
                 episode=Episode(f"c{n}", f"request {n}", steps, right, "ok" if right else "no"),
                 verdict=Verdict(right, "ok" if right else "no") if judged else None, matched=list(matched),
                 offered=[], effort=effort, dollars=0.001, reasoning_tokens=0, seconds=1.0)


# ------------------------------------------------------------ a store that stays bounded

def test_a_bounded_store_keeps_counting_runs_and_drops_only_old_episodes():
    bounded, whole = MetaStore(keep=300), MetaStore()
    for n in range(1500):
        record(bounded, n)
        record(whole, n)
    assert bounded.count() == whole.count() == 1500  # every run keeps its number
    assert len(bounded.judged()) == 1500  # and its verdict, setting, cost and time
    assert bounded.case("c5")["case_data"] is None and bounded.case("c5")["verdict"] == "accepted"
    assert bounded.case("c1490")["case_data"]["order"] == 1490
    assert [e.task_id for e in bounded.episodes(last=3)] == ["c1497", "c1498", "c1499"]
    assert bounded.size() < whole.size() / 2


def test_a_bounded_store_stops_growing_with_the_stream():
    store = MetaStore(keep=200)
    for n in range(1000):
        record(store, n)
    early = store.size()
    for n in range(1000, 3000):
        record(store, n)
    assert store.size() < 1.6 * early  # twice as many runs again, and only their small rows were added


def test_a_bounded_store_gives_back_the_space_of_the_episodes_it_dropped():
    """Dropping an episode shrinks its row and leaves the row's page as it was. Over a long stream each old run
    went on holding a few kilobytes, and a store said to be bounded grew by them."""
    store, runs = MetaStore(keep=200), 3 * 4096 + 1
    for n in range(runs):
        record(store, n)
    window = 200 * 8500  # what the latest runs hold whole: an episode and a world of about four kilobytes each
    assert (store.size() - window) / runs < 700  # and an old run holds a few hundred bytes
    assert store.count() == runs and len(store.judged()) == runs  # with nothing lost: every run and its verdict
    assert store.case(f"c{runs - 1}")["case_data"]["order"] == runs - 1
    assert store.repack()  # and it can be asked for at a quiet hour


def test_the_latest_episodes_are_the_ones_a_slice_of_all_of_them_gives():
    store = MetaStore()
    for n in range(200):
        record(store, n, right=n % 3 != 0, judged=n % 7 != 0, matched=("order_status",) if n % 4 else ("other",))
    assert [e.task_id for e in store.episodes(last=80)] == [e.task_id for e in store.episodes()[-80:]]
    for since in (0, 150):  # and of one procedure's runs, since a run number
        whole = [e.task_id for e in store.episodes(since=since, procedure="order_status")]
        assert [e.task_id for e in store.episodes(since=since, procedure="order_status", last=30)] == whole[-30:]


def test_what_a_signal_and_a_gate_read_matches_a_scan_of_every_run():
    store = MetaStore()
    for n in range(120):
        record(store, n, right=n % 5 != 0, effort="program" if n % 2 else "none", judged=n % 11 != 0)
    runs = store.runs(40)
    assert store.judged(40) == [(r["effort"], r["dollars"], r["seconds"]) for r in runs if r["verdict"] is not None]
    assert store.rejected_cases(40, "program") == [r["case_id"] for r in runs
                                                   if r["effort"] == "program" and r["verdict"] == "rejected"]


# ------------------------------------------------------------ limits that do not last a lifetime

def test_a_learning_budget_can_count_a_day_in_runs():
    store = MetaStore()
    for n in range(40):
        record(store, n)
        if n == 5:
            store.log_learning(name="p", mechanism="m", version=1, episodes=4, writer_calls=1, dollars=0.1,
                               gate=None, outcome="refused")
    budget = LearningBudget(attempts_per_day=1, runs_per_day=20)
    assert budget.refusal(store.learning("p", since_run=store.count() - 100)) == (
        "1 attempts in the last 20 runs, the budget's limit")
    assert budget.refusal(store.learning("p", since_run=store.count() - budget.runs_per_day)) is None
    by_the_clock = LearningBudget(attempts_per_day=1).refusal(store.learning("p"))
    assert by_the_clock.endswith("in the last 24 hours, the budget's limit")


# ------------------------------------------------------------ a tool that is down

class Shop:
    def __init__(self, down: bool = False) -> None:
        self.down = down


@function_tool(read_only=True)
def read_order(ctx: RunContextWrapper[Shop], order_id: str) -> dict:
    """Read an order."""
    if ctx.context.down:
        raise Unavailable("the order service did not answer")
    return {"order_id": order_id, "ok": True}


@function_tool
def ship(ctx: RunContextWrapper[Shop], order_id: str) -> dict:
    """Ship an order."""
    return {"status": "shipped"}


@procedure(api=[read_order])
def order_status(order_id: str) -> dict:
    """Work out an order's status.

    Returns: {"status": str}.
    """
    ...


STATUS = 'def order_status(api, order_id: str) -> dict:\n    return {"status": str(api.read_order(order_id)["ok"])}\n'


def shop() -> tuple[Agent, Metacognition]:
    meta = Metacognition([order_status], verifier=lambda ctx, result: {"success": False, "reason": "not shipped"},
                         learning="inline", relearn_after=10**9, executor="guarded")
    meta.install("order_status", STATUS)
    return Agent(name="Shop", instructions="Ship orders.", tools=[read_order, ship], model="gpt-5.4-mini",
                 metacognition=meta), meta


@pytest.mark.parametrize(("down", "still_trusted"), [(True, True), (False, False)])
async def test_a_run_in_which_a_tool_was_unavailable_is_not_held_against_a_procedure(down, still_trusted):
    agent, meta = shop()
    for n in range(3):  # two rejections in the last four judged uses would quarantine the procedure
        offered = meta.store.trusted("order_status") is not None
        tries = ScriptedModel([call("order_status", {"order_id": f"O-{n}"}), text("I could not ship it.")])
        with tries.active():
            result = await Runner.run(agent, f"Ship O-{n}.", context=Shop(down),
                                      run_config=RunConfig(case_id=f"c{n}"))
        assert not result.metacognition.verdict.accepted
        assert result.metacognition.called == (["order_status@1"] if offered else [])
        assert result.context_wrapper.unavailable is (down and offered)
    assert (meta.store.trusted("order_status") is not None) is still_trusted


async def test_a_call_in_flight_when_its_procedure_is_quarantined_gets_an_error_and_not_a_stale_answer():
    """Under load many requests are in flight when a procedure is quarantined. Each had been offered it already,
    and went on calling the stale version."""
    agent, meta = shop()
    replies = iter([call("order_status", {"order_id": "O-1"}), text("I could not ship it.")])

    def respond(request: dict[str, Any]) -> Any:
        meta.quarantine("order_status", "its own gate now passes 0/2")  # while this request is being served
        return next(replies)

    model = RespondingModel(respond)
    with model.active():
        result = await Runner.run(agent, "Ship O-1.", context=Shop(), run_config=RunConfig(case_id="c1"))
    assert result.metacognition.offered == ["order_status@1"]  # it was trusted when the request began
    returned = model.requests[1]["messages"][-1]
    assert returned["role"] == "tool" and "order_status was withdrawn" in returned["content"]
    assert "ok" not in returned["content"]  # what the withdrawn version would have said


# ------------------------------------------------------------ a trusted function checked again

async def test_a_trusted_function_can_be_put_through_its_own_gate_again():
    async def practice(source: str, case_id: str) -> tuple[float, str, str]:
        return (1.0 if case_id == "holds" else 0.0), "tried", ""

    meta = Metacognition([order_status], verifier=lambda ctx, result: True, learning="inline", practice=practice)
    meta.install("order_status", STATUS)
    assert await meta.recheck("order_status") is None  # its gate holds no cases, so there is nothing to check it on
    meta.gate = Gate(cases=["holds", "fails", "fails too"])
    again = await meta.recheck("order_status")
    assert (again.accepted, again.total, again.passed) == (1, 3, False)
    meta.gate = Gate(cases=["holds"])
    assert (await meta.recheck("order_status")).passed
    meta.quarantine("order_status")
    assert await meta.recheck("order_status") is None  # and nothing is trusted now


# ------------------------------------------------------------ a learner asked late

def test_a_learner_is_handed_only_the_latest_runs_when_its_experience_is_limited():
    for limit, expected in ((None, 30), (8, 8)):
        meta = Metacognition([order_status], verifier=lambda ctx, result: True, learning="inline",
                             experience_limit=limit)
        meta.install("order_status", STATUS)
        meta.quarantine("order_status")
        for n in range(30):
            record(meta.store, n, matched=("order_status",))
        experience: Any = meta._experience("order_status")
        assert len(experience) == expected and experience[-1].task_id == "c29"
