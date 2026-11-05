"""A1: agents with more than one procedure, on more than one model.

The shop from basket.py gains a second procedure, delivery_fee, so a run can be
offered two procedures and call only one. The tests cover blame, routing
(including requests in no procedure's class), the most demanding floor, floors
keyed by model, effort ladders and LiteLLM's translation of them for Claude,
the store's migration from schema 1, background learning that leaves a
returned report alone, and recordings written with LF line endings.
"""

from __future__ import annotations

import json
import sqlite3

import pytest
from basket import CASES, V1, WRITTEN, agent, basket_total, make_meta, practice, price, serve, tax, uses
from litellm.llms.anthropic.chat.transformation import AnthropicConfig

from metacog import Agent, RunConfig, RunContextWrapper, Runner, function_tool
from metacog.ladders import family, ladder_for, most_demanding
from metacog.meta import Controller, Gate, Metacognition, MetaStore, Router, Verdict, procedure
from metacog.meta.store import SCHEMA_VERSION
from metacog.offline import Floors, calibrate_routing
from metacog.recording import Recording, _response
from metacog.testing import ScriptedModel, text


@function_tool(read_only=True)
def zone_rate(ctx: RunContextWrapper[dict], zone: int) -> int:
    """The delivery rate for one zone."""
    return ctx.context["rates"][zone]


@procedure(api=[zone_rate])
def delivery_fee(zone: int) -> int:
    """Work out the delivery fee for a zone.

    Returns: the fee as an integer.
    """
    ...


DELIVERY = "def delivery_fee(api, zone: int) -> int:\n    return api.zone_rate(zone)\n"
ZONES = {"Z2": {"rates": {2: 4}, "zone": 2, "expected": 4}}
DESCRIPTIONS = {"basket_total": basket_total.description, "delivery_fee": delivery_fee.description}
LABELED = [
    ("What do apple, pear cost?", "basket_total"), ("What do pear, pear cost?", "basket_total"),
    ("How much do apple and apple cost?", "basket_total"), ("Total for pear, apple, apple?", "basket_total"),
    ("What is delivery to zone 2?", "delivery_fee"), ("How much is delivery to zone 3?", "delivery_fee"),
    ("Delivery fee for zone 1?", "delivery_fee"), ("What does delivery to zone 4 cost?", "delivery_fee"),
    ("What are your opening hours?", None), ("Do you sell gift cards?", None),
    ("Where is the store?", None), ("Can I return a gift?", None),
]


def check(ctx: RunContextWrapper[dict], result) -> Verdict:
    expected = str(ctx.context["expected"])
    return Verdict(result.final_output.strip() == expected, f"the answer is {expected}")


def shop_meta(**kwargs) -> Metacognition:
    options = dict(verifier=check, practice=practice, recall_k=2, relearn_after=100, learning="inline",
                   controller=Controller({"memory": "none", "no_memory": "high"}))
    options.update(kwargs)
    meta = Metacognition([basket_total, delivery_fee], **options)
    meta.install("basket_total", V1)
    meta.install("delivery_fee", DELIVERY)
    return meta


def shop(meta: Metacognition, model: str = "gpt-5.4-mini") -> Agent:
    return Agent(name="Shop", instructions="Answer the customer.", tools=[price, tax, zone_rate], model=model,
                 metacognition=meta)


async def ask(shop_agent: Agent, case_id: str, request: str | None = None, context: dict | None = None):
    case = context or CASES.get(case_id) or ZONES[case_id]
    if request is None:
        request = (f"What do {', '.join(case['items'])} cost?" if "items" in case
                   else f"What is delivery to zone {case['zone']}?")
    return await Runner.run(shop_agent, request, context=case, run_config=RunConfig(case_id=case_id, case_data=case))


# ------------------------------------------------------------------ blame


@pytest.mark.parametrize(("attribution", "still_trusted"), [("called", ["delivery_fee"]), ("offered", [])])
async def test_a_rejected_run_is_blamed_on_the_procedures_it_called(attribution, still_trusted):
    meta = shop_meta(attribution=attribution)
    # v1 of basket_total ignores the tax, so both totals are wrong; delivery_fee is offered but never called.
    model = ScriptedModel([*uses(["apple", "pear"], 5), *uses(["pear", "pear"], 6)])
    with model.active():
        first = await ask(shop(meta), "D1")
        await ask(shop(meta), "D2")
    report = first.metacognition
    assert sorted(report.offered) == ["basket_total@1", "delivery_fee@1"]
    assert report.called == ["basket_total@1"] and not report.verdict.accepted
    assert [name for name in DESCRIPTIONS if meta.store.trusted(name)] == still_trusted
    assert meta.store.runs()[0]["called"] == ["basket_total@1"]


# ------------------------------------------------------------------ routing


def test_calibrated_routing_separates_the_classes():
    found = calibrate_routing([basket_total, delivery_fee], LABELED)
    assert found.accuracy == 1.0
    assert all(p["matched_outside"] == 0 and p["matched_inside"] == 4 for p in found.procedures.values())
    match = found.router.match
    assert match("What do pear, apple cost?", DESCRIPTIONS)[0] == ["basket_total"]
    assert match("How much is the delivery fee to zone 2?", DESCRIPTIONS)[0] == ["delivery_fee"]
    assert match("Are you open on Sunday?", DESCRIPTIONS)[0] == []
    again = Router.from_dict(json.loads(json.dumps(found.router.to_dict())))
    assert again.to_dict() == found.router.to_dict()


def test_routing_without_labels_outside_a_class_keeps_the_default():
    found = calibrate_routing([basket_total], [r for r in LABELED if r[1] == "basket_total"])
    assert found.router.thresholds == {} and "kept the default" in found.procedures["basket_total"]["note"]


async def test_a_request_in_no_procedures_class_runs_at_the_agents_own_setting():
    meta = shop_meta(router=calibrate_routing([basket_total, delivery_fee], LABELED).router)
    model = ScriptedModel([text("We open at 9."), *uses(["apple", "pear"], 7)])
    with model.active():
        unrelated = await ask(shop(meta), "H1", "Are you open on Sunday?", {"expected": "We open at 9."})
        basket = await ask(shop(meta), "D1")
    assert (unrelated.metacognition.matched, unrelated.metacognition.offered, unrelated.metacognition.effort) == (
        [], [], None)
    assert "reasoning_effort" not in model.requests[0]
    assert [t["function"]["name"] for t in model.requests[0]["tools"]] == ["price", "tax", "zone_rate"]
    # The basket request is offered its own procedure only.
    assert (basket.metacognition.offered, basket.metacognition.effort) == (["basket_total@1"], "none")
    # The unrelated run is no procedure's experience.
    assert [e.task_id for e in meta.store.episodes(procedure="basket_total")] == ["D1"]
    assert meta.store.episodes(procedure="delivery_fee") == []


# ------------------------------------------------------------------ the controller


async def test_the_most_demanding_floor_among_offered_procedures_wins():
    controller = Controller(per_procedure={"basket_total": {"memory": "none", "no_memory": "high"},
                                           "delivery_fee": {"memory": "low", "no_memory": "medium"}})
    meta = shop_meta(controller=controller)
    model = ScriptedModel(uses(["apple", "pear"], 7))
    with model.active():
        result = await ask(shop(meta), "D1")
    assert result.metacognition.effort == "low" and model.requests[0]["reasoning_effort"] == "low"


async def test_floors_are_keyed_by_model(tmp_path):
    controller = Controller.from_floors({"memory": "none", "no_memory": "high", "model": "openai/gpt-5.4-mini"},
                                        {"memory": "low", "model": "claude-haiku-5-5"},
                                        {"no_memory": "medium", "model": "claude-haiku-5-5"})
    assert controller.per_model == {"gpt-5.4-mini": {"table": {"memory": "none", "no_memory": "high"}},
                                    "claude-haiku-5-5": {"table": {"memory": "low", "no_memory": "medium"}}}
    assert controller.effort("p", True, "gpt-5.4-mini") == "none"
    assert controller.effort("p", False, "anthropic/claude-haiku-5-5") == "medium"
    assert controller.effort("p", True, "gemini-2.5-flash") is None  # not calibrated: the agent's own setting
    assert Floors.load(Floors(model="claude-haiku-5-5").save(tmp_path / "floors.json")).model == "claude-haiku-5-5"

    meta = shop_meta(controller=controller)
    model = ScriptedModel([*uses(["apple", "pear"], 7), *uses(["apple", "pear"], 7)])
    with model.active():
        claude = await ask(shop(meta, "claude-haiku-5-5"), "D1")
        gemini = await ask(shop(meta, "gemini-2.5-flash"), "D1")
    assert claude.metacognition.effort == "low" and model.requests[0]["reasoning_effort"] == "low"
    assert gemini.metacognition.effort is None and "reasoning_effort" not in model.requests[2]
    assert [run["model"] for run in meta.store.runs()] == ["claude-haiku-5-5", "gemini-2.5-flash"]


# ------------------------------------------------------------------ effort ladders


def test_ladders_name_families_and_order_rungs():
    models = ("gpt-5.4-mini", "o3", "openai/gpt-5.4-mini", "claude-haiku-5-5", "anthropic/claude-sonnet-5-5",
              "gemini-2.5-flash")
    assert [family(m) for m in models] == ["openai", "openai", "openai", "anthropic", "anthropic", "default"]
    assert ladder_for("claude-haiku-5-5").params("low") == {"reasoning_effort": "low"}
    with pytest.raises(ValueError):
        ladder_for("gpt-5.4-mini").params("max")
    assert most_demanding(["none", "low", "minimal"]) == "low"
    assert most_demanding(["high", None]) is None and most_demanding([]) is None


ADAPTIVE = {"type": "adaptive", "display": "summarized"}


@pytest.mark.parametrize(("model", "effort", "sent"), [
    ("claude-haiku-5-5", "none", {}),
    ("claude-haiku-5-5", "low", {"thinking": ADAPTIVE, "output_config": {"effort": "low"}}),
    ("claude-haiku-5-5", "high", {"thinking": ADAPTIVE, "output_config": {"effort": "high"}}),
    ("claude-haiku-4-5", "low", {"thinking": {"type": "enabled", "budget_tokens": 1024}}),
])
def test_litellm_translates_the_anthropic_ladder(model, effort, sent):
    """What LiteLLM sends to Anthropic for each rung the harness sends as reasoning_effort."""
    params = AnthropicConfig().map_openai_params(non_default_params=ladder_for(model).params(effort),
                                                 optional_params={}, model=model, drop_params=False)
    assert {k: v for k, v in params.items() if k in ("thinking", "output_config")} == sent


async def test_effort_param_overrides_the_ladder():
    meta = make_meta(effort_param="effort")
    meta.install("basket_total", V1)
    model = ScriptedModel(uses(["apple", "pear"], 5))
    with model.active():
        await serve(agent(meta), "S1")
    assert model.requests[0]["effort"] == "none" and "reasoning_effort" not in model.requests[0]


async def test_the_writer_defaults_to_the_agents_model():
    meta = make_meta()
    claude = Agent(name="Shop", instructions="Total the basket.", tools=[price, tax], model="claude-haiku-5-5",
                   metacognition=meta)
    model = ScriptedModel([text("7"), text("8"), WRITTEN, WRITTEN])
    with model.active():
        await serve(claude, "A")
        last = await serve(claude, "B")
    assert last.metacognition.events[-1].endswith("installed")
    assert {r["model"] for r in model.requests} == {"claude-haiku-5-5"}


# ------------------------------------------------------------------ the store

V1_SCHEMA = """
CREATE TABLE procedures (
    name TEXT NOT NULL, version INTEGER NOT NULL, source TEXT NOT NULL, state TEXT NOT NULL,
    created_by TEXT NOT NULL, parent INTEGER, created_at TEXT NOT NULL, info TEXT NOT NULL,
    PRIMARY KEY (name, version));
CREATE TABLE episodes (
    ord INTEGER PRIMARY KEY, run_id TEXT UNIQUE NOT NULL, case_id TEXT NOT NULL, case_data TEXT,
    request TEXT NOT NULL, content TEXT NOT NULL, verdict TEXT, verdict_source TEXT, reason TEXT,
    matched TEXT NOT NULL, offered TEXT NOT NULL, effort TEXT, dollars REAL, reasoning_tokens INTEGER,
    seconds REAL, created_at TEXT NOT NULL);
CREATE TABLE state (name TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE learning (
    id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL, mechanism TEXT NOT NULL, version INTEGER,
    episodes INTEGER NOT NULL, writer_calls INTEGER NOT NULL, dollars REAL NOT NULL, gate TEXT, outcome TEXT NOT NULL,
    at_ord INTEGER NOT NULL, created_at TEXT NOT NULL);
"""


async def test_a_store_from_before_a1_is_migrated(tmp_path):
    path = tmp_path / "old.meta"
    path.mkdir()
    conn = sqlite3.connect(path / "meta.sqlite")
    conn.executescript(V1_SCHEMA)
    conn.execute("INSERT INTO procedures VALUES ('basket_total', 1, ?, 'trusted', 'human', NULL, 'then', '{}')",
                 (V1,))
    content = {"task_id": "D1", "user_message": "What do apple, pear cost?", "steps": [], "success": False,
               "reason": "", "final_output": ""}
    conn.execute("INSERT INTO episodes VALUES (0, 'run_old', 'D1', 'null', 'What do apple, pear cost?', ?, NULL,"
                 " NULL, NULL, '[\"basket_total\"]', '[\"basket_total@1\"]', 'none', 0.0, 0, 0.0, 'then')",
                 (json.dumps(content),))
    conn.commit()
    conn.close()

    meta = make_meta(store=str(path))
    assert meta.store.schema_version == SCHEMA_VERSION
    old = meta.store.runs()[0]
    assert old["called"] is None and old["offered"] == ["basket_total@1"]
    await meta.feedback("run_old", False, "tax was missing")  # an old run is blamed on what it was offered
    assert meta.store.state("basket_total")["recent"] == [False]
    with ScriptedModel(uses(["apple", "pear"], 7)).active():
        await serve(agent(meta), "D1")
    new = meta.store.runs()[1]
    assert (new["called"], new["model"], sorted(new["scores"])) == (["basket_total@1"], "gpt-5.4-mini",
                                                                    ["basket_total"])
    meta.store.close()

    conn = sqlite3.connect(path / "meta.sqlite")
    conn.execute("UPDATE meta SET value = '99' WHERE key = 'schema_version'")
    conn.commit()
    conn.close()
    with pytest.raises(RuntimeError, match="newer metacog"):
        MetaStore(path)


def test_a_store_on_disk_uses_the_write_ahead_log(tmp_path):
    store = MetaStore(tmp_path / "s.meta")
    assert store._conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert store.schema_version == SCHEMA_VERSION


# ------------------------------------------------------------------ reports and recordings


async def test_background_learning_reports_to_the_store_not_to_a_returned_report():
    meta = make_meta(learning="background", gate=Gate(holdout=1, bar=1.0))
    meta.install("basket_total", V1)
    shop_agent = agent(meta)
    model = ScriptedModel([*uses(["apple", "pear"], 5), *uses(["pear", "pear"], 6), text("7"), text("8"), WRITTEN])
    with model.active():
        reports = [(await serve(shop_agent, case_id)).metacognition for case_id in ("D1", "D2", "A", "B")]
        events = [list(r.events) for r in reports]
        await meta.wait_for_learning()
    assert [list(r.events) for r in reports] == events  # nothing was added after the runs returned
    assert events[-1] == ["learner: learning basket_total in the background"]
    assert meta.store.learning()[-1]["outcome"] == "passed, installed"


async def test_recordings_are_written_with_lf(tmp_path, monkeypatch):
    async def model(**_kwargs):
        return _response({"content": "hi", "tool_calls": [], "usage": {"prompt_tokens": 1, "completion_tokens": 1}})

    monkeypatch.setattr("metacog.recording._call_model", model)
    recording = Recording(tmp_path / "calls.jsonl", mode="record")
    await recording(model="gpt-5.4-mini", messages=[{"role": "user", "content": "hello"}])
    await recording(model="gpt-5.4-mini", messages=[{"role": "user", "content": "again"}])
    data = (tmp_path / "calls.jsonl").read_bytes()
    assert data.count(b"\n") == 2 and b"\r" not in data
