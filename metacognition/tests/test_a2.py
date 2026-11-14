"""A2: deployment. A learning worker beside serving, a store several processes share, model-written code
in a child process by default, a learning budget, a judge, and floors kept in the store."""

from __future__ import annotations

import asyncio
import json
import re
import threading
import time

import pytest
from basket import CASES, PRICES, V1, V2, WRITTEN, agent, make_meta, serve, uses
from helix.eval.effort import EffortCell, EffortSweep, RunRecord
from helix.memory.base import MemoryContext
from helix.memory.learning_mechanisms import Episode
from helix.metacognition import Controller, Judge, LearningBudget, MetaStore, Verdict, sandbox
from helix.metacognition.offline import Floors
from helix.metacognition.report import summarize
from helix.metacognition.worker import load_agent, work

from metacog import RunConfig, RunContextWrapper, cli
from metacog.testing import RespondingModel, ScriptedModel, call, text

# Two rejected runs quarantine v1 (relearn_after=2), then two accepted runs make an attempt due.
SERVE = ("D1", "D2", "A", "B")
SERVING_REPLIES = [*uses(["apple", "pear"], 5), *uses(["pear", "pear"], 6), text("7"), text("8")]


async def serve_until_due(meta):
    shop = agent(meta)
    model = ScriptedModel(list(SERVING_REPLIES))
    with model.active():
        for case_id in SERVE:
            await serve(shop, case_id)
    return model


# ------------------------------------------------------------------ the worker


async def test_serving_only_records_and_the_worker_learns():
    meta = make_meta(learning="worker")
    meta.install("basket_total", V1)
    serving = await serve_until_due(meta)
    assert serving.replies == [] and meta.store.learning() == []  # no writer call on the serving path
    assert meta.store.trusted("basket_total") is None

    with ScriptedModel([WRITTEN, WRITTEN]).active():
        events = await meta.learn_due(holder="w1")
    assert events == ["learner: value_learning wrote basket_total from 2 runs; gate 2/2 passed, installed"]
    assert meta.store.trusted("basket_total")["version"] == 2
    state = meta.store.state("basket_total")
    assert "attempt" not in state and state["next_attempt"] == 6
    assert await meta.learn_due(holder="w1") == []  # nothing left to do


async def test_an_attempt_cut_short_is_retried_by_the_next_worker(tmp_path, monkeypatch):
    path = str(tmp_path / "shop.meta")
    meta = make_meta(learning="worker", store=path)
    meta.install("basket_total", V1)
    await serve_until_due(meta)

    async def killed(*args, **kwargs):
        raise asyncio.CancelledError  # the worker process dies mid-attempt

    monkeypatch.setattr(meta.learner, "learn", killed)
    with pytest.raises(asyncio.CancelledError):
        await meta.learn_due(holder="w1")
    state = meta.store.state("basket_total")
    assert state["attempt"]["holder"] == "w1" and state["next_attempt"] == 4  # still due
    assert meta.store.learning() == []

    restarted = make_meta(learning="worker", store=path)  # a new process on the same store
    with ScriptedModel([WRITTEN, WRITTEN]).active():
        events = await restarted.learn_due(holder="w2")
    assert events[0].startswith("learner: retrying basket_total; the attempt w1 started at")
    assert events[1].endswith("installed") and restarted.store.trusted("basket_total")["version"] == 2


def test_the_lease_admits_one_worker_at_a_time(tmp_path):
    store = MetaStore(tmp_path / "s.meta")
    assert store.acquire_lease("worker", "w1", 60, now=0)
    assert not store.acquire_lease("worker", "w2", 60, now=30)
    assert store.acquire_lease("worker", "w1", 60, now=30)  # renewed until 90
    assert not store.acquire_lease("worker", "w2", 60, now=89)
    assert store.acquire_lease("worker", "w2", 60, now=91)  # w1 stopped renewing
    store.release_lease("worker", "w1")  # only the holder can release
    assert store.lease("worker")["holder"] == "w2"
    store.release_lease("worker", "w2")
    assert store.lease("worker") is None


async def test_the_worker_command_learns_for_an_agent_named_by_a_target(tmp_path, monkeypatch):
    app = tmp_path / "worker_app_a2.py"
    app.write_text("from basket import agent, make_meta\n\n"
                   f"shop = agent(make_meta(learning='worker', store={str(tmp_path / 'app.meta')!r}))\n",
                   encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    shop = load_agent("worker_app_a2:shop")
    shop.metacognition.install("basket_total", V1)
    await serve_until_due(shop.metacognition)
    assert cli._open("worker_app_a2:shop").count() == 4

    lines: list[str] = []
    shop.metacognition.store.acquire_lease("worker", "someone else", 60)
    assert await work(shop, once=True, holder="w1", out=lines.append) == 1
    assert lines == ["waiting: another worker (someone else) holds this store's lease"]
    shop.metacognition.store.release_lease("worker", "someone else")

    with ScriptedModel([WRITTEN, WRITTEN]).active():
        await work(shop, once=True, holder="w1", out=lines.append)
    assert lines[-1].endswith("installed") and shop.metacognition.store.lease("worker") is None


# ------------------------------------------------------------------ one store, several processes


def test_serving_worker_and_cli_share_one_store(tmp_path):
    path = tmp_path / "shared.meta"
    MetaStore(path).close()
    errors: list[BaseException] = []

    def serving(tag: str) -> None:
        store = MetaStore(path)  # each role has its own connection, as separate processes would
        for i in range(40):
            store.record(run_id=f"{tag}{i}", case_id=f"{tag}{i}", case_data=None, request="r",
                         episode=Episode(f"{tag}{i}", "r", [], True, ""), verdict=Verdict(True, "ok"), matched=[],
                         offered=[], effort="none", dollars=0.001, reasoning_tokens=0, seconds=0.1)

    def worker() -> None:
        store = MetaStore(path)
        for i in range(40):
            store.acquire_lease("worker", "w", 60)
            store.put_state("p", {"next_attempt": i})
            store.log_learning(name="p", mechanism="value_learning", version=None, episodes=0, writer_calls=1,
                               dollars=0.01, gate=None, outcome="refused")
            store.add_version("p", V1, state="rejected", created_by="value_learning")

    def operator() -> None:
        store = MetaStore(path)
        for _ in range(40):
            summarize(store)
            store.put_floors("gpt-5.4-mini", {"memory": "none"})

    def guarded(fn, *args):
        def run() -> None:
            try:
                fn(*args)
            except BaseException as exc:  # noqa: BLE001 - collected and asserted below
                errors.append(exc)
        return threading.Thread(target=run)

    threads = [guarded(serving, "a"), guarded(serving, "b"), guarded(worker), guarded(operator)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
    store = MetaStore(path)
    assert sorted(r["ord"] for r in store.runs()) == list(range(80))  # no run number given out twice
    assert [v["version"] for v in store.versions("p")] == list(range(1, 41))
    assert len(store.learning()) == 40


# ------------------------------------------------------------------ the executor

HOSTILE = "def basket_total(api, items: list[str]) -> int:\n    return 10 ** (len(items) * 10 ** 8) % 7\n"


async def test_model_written_code_runs_in_a_child_process_by_default(monkeypatch):
    meta = make_meta(skill_timeout=0.5)
    meta.install("basket_total", V2, created_by="value_learning")
    ctx = RunContextWrapper(context=CASES["A"])
    plan = await meta.before_run(agent(meta), ctx, "What do apple, pear cost?", RunConfig(), MemoryContext())
    assert "sandboxed_tool" in plan.tools[0].fn.__qualname__
    people = make_meta()
    people.install("basket_total", V2)  # written by a person: in-process, under the guard
    plan = await people.before_run(agent(people), ctx, "What do apple, pear cost?", RunConfig(), MemoryContext())
    assert "sandboxed_tool" not in plan.tools[0].fn.__qualname__

    # A computation the in-process guard can't interrupt (one long C-level operation) is killed in its child.
    monkeypatch.setattr(sandbox, "STARTUP_ALLOWANCE", 2.0)
    meta.install("basket_total", HOSTILE, created_by="value_learning")
    model = ScriptedModel(uses(["apple", "pear"], 7))
    started = time.perf_counter()
    with model.active():
        await serve(agent(meta), "A")
    assert json.loads(model.requests[1]["messages"][-1]["content"]) == {
        "error": "basket_total was stopped after 0.5 seconds"}
    assert time.perf_counter() - started < 30


# ------------------------------------------------------------------ the budget


def shop_answers(request: dict):
    """The agent works out totals itself, calls the procedure when offered, and the writer keeps writing v1."""
    if request["messages"][0]["content"].startswith("You are writing a Python function"):
        return text(f"```python\n{V1}```")  # a stale version: the gate refuses it
    last = request["messages"][-1]
    if last["role"] == "tool":
        return text(last["content"])
    items = re.search(r"What do (.*) cost\?", last["content"]).group(1).split(", ")
    if any(t["function"]["name"] == "basket_total" for t in request.get("tools") or []):
        return call("basket_total", {"items": items})
    return text(str(sum(PRICES[item] + 1 for item in items)))


async def test_the_budget_stops_learning_and_says_why():
    meta = make_meta(budget=LearningBudget(attempts_per_day=1))
    meta.install("basket_total", V1)
    shop = agent(meta)
    model = RespondingModel(shop_answers)
    with model.active():
        reports = [(await serve(shop, case_id)).metacognition for case_id in (*SERVE, "A", "B")]
    outcomes = [a["outcome"] for a in meta.store.learning()]
    assert outcomes == ["refused", "skipped: 1 attempts in the last 24 hours, the budget's limit"]
    assert reports[3].events[-1].endswith("refused")
    assert reports[5].events == ["learner: skipped basket_total: 1 attempts in the last 24 hours, the budget's limit"]
    learning = summarize(meta.store)["learning"]
    assert (learning["attempts"], learning["skipped"]) == (1, 1)


def test_a_dollar_budget_counts_what_attempts_spent():
    budget = LearningBudget(dollars_per_day=0.25)
    assert budget.refusal([{"outcome": "refused", "dollars": 0.2}]) is None
    assert "$0.3000 spent" in budget.refusal([{"outcome": "refused", "dollars": 0.2},
                                               {"outcome": "passed, installed", "dollars": 0.1}])
    assert budget.refusal([{"outcome": "skipped: ...", "dollars": 9.0}]) is None


# ------------------------------------------------------------------ the judge

NOT_ACCEPTED = text('{"accepted": false, "reason": "the tax is missing"}')


async def test_a_judge_is_recorded_but_neither_quarantines_nor_installs_alone():
    judge = Judge("The total includes each item's tax.")
    meta = make_meta(verifier=judge)
    assert meta.approval == "human"  # a judge's word never installs code by itself
    meta.install("basket_total", V1)
    model = ScriptedModel([*uses(["apple", "pear"], 5), NOT_ACCEPTED, *uses(["pear", "pear"], 6), NOT_ACCEPTED])
    with model.active():
        first = await serve(agent(meta), "D1")
        await serve(agent(meta), "D2")
    assert first.metacognition.verdict == Verdict(False, "the tax is missing", "judge")
    assert meta.store.trusted("basket_total")["version"] == 1  # judges don't drive the detector by default
    asked = model.requests[2]
    assert asked["model"] == "gpt-5.4-mini" and "each item's tax" in asked["messages"][0]["content"]
    shown = asked["messages"][1]["content"]
    assert shown.startswith("Request: What do apple, pear cost?")
    assert 'called basket_total({"items": ["apple", "pear"]})' in shown and shown.endswith("Final answer: 5")

    trusting = make_meta(verifier=judge, gating_sources=("verifier", "human", "harness", "judge"))
    trusting.install("basket_total", V1)
    with ScriptedModel([*uses(["apple", "pear"], 5), NOT_ACCEPTED, *uses(["pear", "pear"], 6),
                        NOT_ACCEPTED]).active():
        await serve(agent(trusting), "D1")
        await serve(agent(trusting), "D2")
    assert trusting.store.trusted("basket_total") is None  # opted in, a judge informs the detector


def test_a_judges_reply_is_read_or_left_unjudged():
    assert Judge.read('```json\n{"accepted": true, "reason": "right"}\n```') == Verdict(True, "right", "judge")
    assert Judge.read("I think it's fine.") is None
    assert Judge.read('{"accepted": "yes"}') is None


# ------------------------------------------------------------------ floors in the store


def sweep_of(arm: str, solved: dict[str, int], total: int = 8) -> EffortSweep:
    return EffortSweep(arm, {e: EffortCell(e, [RunRecord(f"t{i}", e, i < n) for i in range(total)])
                             for e, n in solved.items()})


async def test_floors_recorded_in_the_store_reach_the_controller(tmp_path):
    meta = make_meta(controller=Controller(), store=str(tmp_path / "floors.meta"))
    meta.install("basket_total", V2)
    floors = Floors({"memory": sweep_of("memory", {"none": 6, "low": 8, "medium": 8, "high": 8}),
                     "no_memory": sweep_of("no_memory", {"none": 2, "low": 5, "medium": 7, "high": 8})},
                    model="gpt-5.4-mini")
    floors.record(meta)
    assert meta.store.floors() == {"gpt-5.4-mini": {"table": {"memory": "low", "no_memory": "medium"}}}
    model = ScriptedModel(uses(["apple", "pear"], 7))
    with model.active():
        result = await serve(agent(meta), "A")
    assert result.metacognition.effort == "low" and model.requests[0]["reasoning_effort"] == "low"
    with pytest.raises(ValueError, match="name no model"):
        Floors(floors.sweeps).record(meta)
