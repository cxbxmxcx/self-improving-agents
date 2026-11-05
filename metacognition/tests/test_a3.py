"""A3: case files and `metacog calibrate`, `metacog discover`, webhooks, and `metacog serve` with its dashboard."""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path

import httpx
import pytest
from basket import CASES, PRICES, V2, agent, make_meta
from helix.memory.learning_mechanisms import Episode
from helix.observability.bus import get_bus

from examples.expenses.agent import REFERENCE, reimbursable_total
from examples.expenses.agent import assistant as expense_assistant
from examples.expenses.world import ExpenseWorld, task_by_id, task_set
from metacog import Agent, Recording, cli
from metacog.discover import discover, draft_stub, skeleton
from metacog.meta import Controller, Metacognition, MetaStore, Router, Verdict
from metacog.meta.events import publish, webhook
from metacog.offline import Case, calibrate_deployment, load_cases, save_cases
from metacog.service import create_app
from metacog.testing import RespondingModel, ScriptedModel, call, text

SMOKE = Path(__file__).parents[1] / "examples" / "expenses" / "recorded" / "smoke.jsonl"
HEADERS = {"X-Metacog": "1"}

# ------------------------------------------------------------------ case files


def test_case_files_round_trip_and_keep_labels(tmp_path):
    cases = [Case("A", "What do apple, pear cost?", {"items": ["apple", "pear"]}, "basket_total", True),
             Case("H", "When do you open?", None, None, True), Case("X", "Unlabeled request")]
    loaded = load_cases(save_cases(tmp_path / "cases.jsonl", cases))
    assert loaded == cases
    bad = tmp_path / "bad.jsonl"
    for content, message in (('{"id": "A"}', 'needs "id" and "input"'), ("{nope", "not JSON"),
                             ('{"id": "A", "input": "x"}\n{"id": "A", "input": "y"}', "more than once")):
        bad.write_text(content, encoding="utf-8")
        with pytest.raises(ValueError, match=message):
            load_cases(bad)


# ------------------------------------------------------------------ calibrate a deployment


def shop_model(request: dict):
    """Without the procedure the agent forgets the tax at `none` and gets it right at `high`."""
    last = request["messages"][-1]
    if last["role"] == "tool":
        return text(last["content"])
    found = re.search(r"What do (.*) cost\?", last["content"])
    if found is None:
        return text("9 to 5")
    items = found.group(1).split(", ")
    if any(t["function"]["name"] == "basket_total" for t in request.get("tools") or []):
        return call("basket_total", {"items": items})
    tax_each = 1 if request.get("reasoning_effort") == "high" else 0
    return text(str(sum(PRICES[item] + tax_each for item in items)))


BASKETS = [["apple", "pear"], ["pear", "pear"], ["apple", "apple"], ["pear", "apple", "apple"]]


def labeled_cases() -> list[Case]:
    shop = [Case(f"B{n}", f"What do {', '.join(items)} cost?", {**CASES["A"], "items": items,
                                                                   "expected": sum(PRICES[i] + 1 for i in items)},
                 "basket_total", True) for n, items in enumerate(BASKETS)]
    other = [Case(f"H{n}", request, {"expected": "9 to 5"}, None, True)
             for n, request in enumerate(["When do you open?", "Are you open on Sunday?", "What are your hours?"])]
    return shop + other


def world(data: dict) -> dict:
    return data


async def test_calibrating_a_deployment_records_floors_and_routing_in_its_store(tmp_path):
    meta = make_meta(controller=Controller(), store=str(tmp_path / "shop.meta"))
    meta.install("basket_total", V2)
    with RespondingModel(shop_model).active():
        result = await calibrate_deployment(agent(meta), labeled_cases(), world=world, efforts=("none", "high"))
    assert result.floors.table() == {"no_memory": "high", "memory": "none"}
    assert meta.store.floors() == {"gpt-5.4-mini": {"table": {"memory": "none", "no_memory": "high"}}}
    assert result.routing.accuracy == 1.0 and meta.store.setting("router")["thresholds"]["basket_total"] > 0

    reopened = make_meta(controller=Controller(), store=str(tmp_path / "shop.meta"))  # a later process
    assert isinstance(reopened.router, Router) and reopened.router.thresholds == meta.router.thresholds
    assert reopened._match("When do you open on Saturday?") == []
    assert reopened.setting("basket_total", False, "gpt-5.4-mini") == "high"


async def test_calibration_skips_the_memory_arm_until_a_procedure_is_trusted(tmp_path):
    meta = make_meta(controller=Controller(), store=str(tmp_path / "cold.meta"))
    with RespondingModel(shop_model).active():
        result = await calibrate_deployment(agent(meta), labeled_cases()[:4], world=world, efforts=("none", "high"))
    assert list(result.floors.sweeps) == ["no_memory"]
    # No labeled case lies outside the class, so there's no threshold to set and no router to save.
    assert result.skipped == ["memory: no procedure is trusted yet",
                              "routing: every procedure needs labeled requests both in and out of its class"]
    assert meta.store.setting("router") is None


# ------------------------------------------------------------------ discover


def record(store: MetaStore, n: int, request: str, tools: list[str], reasoning: int, *, offered=(), called=()):
    steps = [{"tool": tool, "arguments": {}, "result": 1} for tool in tools]
    store.record(run_id=f"run{n}", case_id=f"c{n}", case_data=None, request=request,
                 episode=Episode(f"c{n}", request, steps, True, ""), verdict=Verdict(True, "ok"), matched=[],
                 offered=list(offered), effort="high", dollars=0.01, reasoning_tokens=reasoning, seconds=1.0,
                 answer="7", called=list(called))


def busy_store() -> MetaStore:
    store = MetaStore()
    for n, items in enumerate(BASKETS[:3]):
        record(store, n, f"What do {', '.join(items)} cost?", ["price", "price", "tax"], 900)
    for n in (3, 4):
        record(store, n, f"What is delivery to zone {n}?", ["zone_rate"], 200)
    record(store, 5, "Cancel my order", ["cancel_order"], 5000)  # once only: not recurring
    record(store, 6, "What do pear, pear cost?", ["basket_total"], 0, offered=["basket_total@1"],
           called=["basket_total@1"])  # a procedure already does it
    return store


def test_discover_ranks_recurring_work_by_the_reasoning_it_spends():
    store = busy_store()
    groups = discover(store)
    assert [(g.tools, g.runs, g.reasoning_tokens) for g in groups] == [(("price", "tax"), 3, 2700),
                                                                        (("zone_rate",), 2, 400)]
    assert groups[0].requests[0] == "What do apple, pear cost?"
    stub = skeleton(groups[0], read_only=["price"])
    assert stub.startswith("@procedure(api=[price])\ndef name_this(...) -> dict:") and "Returns:" in stub


async def test_a_writer_drafts_a_stub_from_example_runs():
    store = busy_store()
    group = discover(store)[0]
    drafted = '@procedure(api=[price, tax])\ndef basket_total(items: list[str]) -> int:\n    """..."""\n    ...'
    model = ScriptedModel([text(f"Here it is:\n```python\n{drafted}\n```")])
    with model.active():
        stub = await draft_stub(group, store, model="gpt-5.4-mini", read_only=["price", "tax"])
    assert stub == drafted + "\n"
    asked = model.requests[0]["messages"]
    assert "@procedure(api=[price, tax])" in asked[0]["content"]
    assert "Request: What do apple, pear cost?" in asked[1]["content"] and "- price({}) -> 1" in asked[1]["content"]


# ------------------------------------------------------------------ webhooks


async def test_a_webhook_posts_the_events_a_person_may_act_on():
    posted: list[dict] = []
    client = httpx.AsyncClient(transport=httpx.MockTransport(
        lambda request: posted.append(json.loads(request.content)) or httpx.Response(204)))
    handler = webhook("https://hooks.example/metacog", client=client)
    try:
        await publish(kind="run", run_id="run_1", effort="none")
        await publish(kind="proposed", procedure="basket_total", version=2, detail="value_learning, gate 2/2, $0.0100")
    finally:
        get_bus().unsubscribe("metacognition", handler)
        await client.aclose()
    assert [p["kind"] for p in posted] == ["proposed"]
    assert posted[0]["text"] == "metacog: proposed basket_total@2: value_learning, gate 2/2, $0.0100"


# ------------------------------------------------------------------ the service


def served_expenses(tmp_path) -> Agent:
    """The expense assistant with a person judging runs, on a store of its own."""
    meta = Metacognition([reimbursable_total], recall_k=1, learning="inline", store=str(tmp_path / "jordan.meta"))
    meta.install("reimbursable_total", REFERENCE)
    return expense_assistant(metacognition=meta)


def client_for(app) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://metacog.test")


async def test_the_expense_agent_is_served_and_judged_over_http(tmp_path):
    if not SMOKE.exists():
        pytest.skip("no recorded smoke study")
    shop = served_expenses(tmp_path)
    app = create_app(shop, context=lambda body: ExpenseWorld(task_by_id(body["case_id"])))
    tasks = task_set("dev")[:2]
    async with client_for(app) as http:
        page = await http.get("/")
        assert page.status_code == 200 and "/api/stream" in page.text and "X-Metacog" in page.text
        refused = await http.post("/api/runs", json={"input": tasks[0]["text"], "case_id": tasks[0]["id"]})
        assert refused.status_code == 403
        with Recording(SMOKE).active():  # the smoke study's replies: no API key
            served = [(await http.post("/api/runs", json={"input": t["text"], "case_id": t["id"]},
                                       headers=HEADERS)).json() for t in tasks]
        assert [s["metacognition"]["offered"] for s in served] == [["reimbursable_total@1"]] * 2
        assert all(s["metacognition"]["effort"] == "none" and s["metacognition"]["verdict"] is None for s in served)

        queue = (await http.get("/api/queue")).json()
        assert [r["run_id"] for r in queue["unjudged"]] == [s["run_id"] for s in reversed(served)]
        judged = await http.post(f"/api/runs/{served[0]['run_id']}/feedback",
                                 json={"accepted": True, "reason": "matches the receipts"}, headers=HEADERS)
        assert (judged.json()["verdict"], judged.json()["verdict_source"]) == ("accepted", "human")
        assert len((await http.get("/api/queue")).json()["unjudged"]) == 1

        detail = (await http.get(f"/api/runs/{served[0]['run_id']}")).json()
        assert detail["called"] == ["reimbursable_total@1"] and detail["steps"][0]["tool"] == "read_trip"
        assert {step["tool"] for step in detail["steps"]} >= {"reimbursable_total", "file_claim"}
        procedures = (await http.get("/api/procedures")).json()
        assert procedures[0]["trusted"] == 1 and procedures[0]["recent"] == [True]
        summary = (await http.get("/api/summary")).json()
        assert (summary["runs"], summary["approval"], summary["learning_mode"]) == (2, "human", "inline")
        assert summary["learning"]["attempts"] == 0  # the report's learning totals, not the setting

        quarantined = await http.post("/api/procedures/reimbursable_total/quarantine", json={}, headers=HEADERS)
        assert quarantined.status_code == 200 and shop.metacognition.store.trusted("reimbursable_total") is None
        again = await http.post("/api/procedures/reimbursable_total/quarantine", json={}, headers=HEADERS)
        assert again.status_code == 409
        assert (await http.get("/api/runs/run_missing")).status_code == 404


async def test_a_proposal_is_reviewed_and_approved_through_the_service(tmp_path):
    from examples.calendar.drift_demo import RECORDED, build, replay

    if not (RECORDED / "online_loop.jsonl").exists():
        pytest.skip("book recordings missing")
    calendar = build(tmp_path / "drift.meta")
    lines: list[str] = []
    await replay(calendar, out=lines.append)  # chapter 6's E5, runs 1 to 12, with a person approving installs
    assert lines[-1].endswith("gate 6/6 proposed for approval")
    async with client_for(create_app(calendar)) as http:
        proposal = (await http.get("/api/queue")).json()["proposals"][0]
        assert (proposal["name"], proposal["version"], proposal["learned_from"], proposal["gate"]["total"]) == (
            "find_shared_starts", 2, 4, 6)
        version = (await http.get("/api/procedures/find_shared_starts/2")).json()
        assert version["compared_with"] == 1 and version["diff"].startswith("--- find_shared_starts@1")
        approved = await http.post("/api/procedures/find_shared_starts/2/approve", headers=HEADERS)
        assert approved.json()["state"] == "trusted"
        assert (await http.get("/api/queue")).json()["proposals"] == []
        procedures = (await http.get("/api/procedures")).json()
        assert procedures[0]["trusted"] == 2
        assert [v["state"] for v in procedures[0]["versions"]] == ["quarantined", "trusted"]
        twice = await http.post("/api/procedures/find_shared_starts/2/approve", headers=HEADERS)
        assert twice.status_code == 409


def test_the_stamp_changes_with_every_write(tmp_path):
    store = MetaStore(tmp_path / "s.meta")
    first = store.stamp()
    store.put_state("p", {"recent": [True]})
    second = store.stamp()
    other = sqlite3.connect(tmp_path / "s.meta" / "meta.sqlite")  # another process's write
    other.execute("UPDATE state SET value = '{}' WHERE name = 'p'")
    other.commit()
    other.close()
    assert len({first, second, store.stamp()}) == 3


# ------------------------------------------------------------------ the CLI


def write_app(tmp_path, name: str, *, verifier: bool) -> str:
    """A module that names the basket shop: judged by its verifier, or by a person."""
    options = "" if verifier else "verifier=None, "
    (tmp_path / f"{name}.py").write_text(
        "from basket import agent, make_meta\n"
        "from metacog.meta import Controller\n\n"
        f"shop = agent(make_meta({options}controller=Controller(), store={str(tmp_path / 'cli.meta')!r}))\n\n\n"
        "def world(data):\n    return data\n", encoding="utf-8")
    return f"{name}:shop"


def test_feedback_from_the_command_line(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    target = write_app(tmp_path, "cli_app_feedback", verifier=False)
    store = cli._open(target)
    record(store, 0, "What do apple, pear cost?", ["price"], 100)
    store.set_verdict("run0", Verdict(True, "", "verifier"))
    record(store, 1, "What do pear, pear cost?", ["price"], 100)
    assert cli.main(["feedback", target, "2", "reject", "--reason", "the tax is missing"]) == 0
    assert "run 2 (run1) is rejected: the tax is missing" in capsys.readouterr().out
    assert cli.main(["feedback", target, "run0", "accept"]) == 0
    assert cli.main(["feedback", target, "run_missing", "accept"]) == 1
    with pytest.raises(SystemExit, match="needs the agent"):  # a bare store has no detector to tell
        cli.main(["feedback", str(tmp_path / "cli.meta"), "run0", "accept"])


def test_calibrate_and_discover_from_the_command_line(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    target = write_app(tmp_path, "cli_app_calibrate", verifier=True)
    cases = save_cases(tmp_path / "cases.jsonl", labeled_cases())
    with RespondingModel(shop_model).active():
        code = cli.main(["calibrate", target, str(cases), "--world", "cli_app_calibrate:world",
                         "--efforts", "none", "high"])
    out = capsys.readouterr().out
    assert code == 0 and "recorded floors {'no_memory': 'high'} for gpt-5.4-mini" in out
    assert "skipped memory: no procedure is trusted yet" in out and "recorded routing: 100%" in out

    store = MetaStore(tmp_path / "cli.meta")
    assert store.floors() == {"gpt-5.4-mini": {"table": {"no_memory": "high"}}}
    assert cli.main(["discover", target, "--min-runs", "2"]) == 0
    found = capsys.readouterr().out  # calibration runs are frozen or off, so the store holds no runs to mine
    assert "no work recurs" in found
