"""M4: the subprocess sandbox, streaming, per-user metacognition, replayed practice, events, and concurrency."""

from __future__ import annotations

import asyncio
import json
import re

import pytest
from basket import CASES, V1, V2, agent, make_meta, price, serve, tax, uses, verifier
from helix.observability.bus import get_bus

from metacog import (
    Agent,
    MaxTurnsExceeded,
    RunConfig,
    RunContextWrapper,
    Runner,
    function_tool,
)
from metacog.meta import Metacognition, PerUser, ReplayPractice, SkillAPI, procedure, sandboxed_tool
from metacog.meta.sandbox import run_sandboxed
from metacog.testing import Reply, RespondingModel, ScriptedModel, call, text

# ------------------------------------------------------------------ the sandbox


async def test_sandboxed_code_runs_in_a_child_and_reaches_tools_through_the_parent():
    api = SkillAPI([price, tax], RunContextWrapper(context=CASES["D1"]), asyncio.get_running_loop())
    tool = sandboxed_tool("basket_total", "Add up the basket.", V2, api, timeout=5)
    assert list(tool.args_model.model_fields) == ["items"]
    assert await tool.call(items=["apple", "pear"]) == 7  # price is sync, tax async: both served from here


async def test_the_sandbox_reports_crashes_timeouts_and_missing_builtins():
    def no_api(tool, args, kwargs):
        raise AssertionError("not called")

    reads = 'def peek(api, path: str) -> str:\n    return open(path).read()\n'
    result = await asyncio.to_thread(run_sandboxed, "peek", reads, {"path": "x"}, no_api, [], 5.0)
    assert result == {"error": "peek failed: NameError: name 'open' is not defined"}
    spins = 'def spin(api) -> int:\n    n = 0\n    while True:\n        n += 1\n'
    result = await asyncio.to_thread(run_sandboxed, "spin", spins, {}, no_api, [], 0.5)
    assert result == {"error": "spin took longer than 0.5 seconds"}
    raises = 'def bad(api) -> int:\n    return 1 // 0\n'
    result = await asyncio.to_thread(run_sandboxed, "bad", raises, {}, no_api, [], 5.0)
    assert result["error"].startswith("bad failed: ZeroDivisionError")


async def test_metacognition_can_run_procedures_in_the_sandbox():
    meta = make_meta(executor="subprocess")
    meta.install("basket_total", V2)
    model = ScriptedModel(uses(["apple", "pear"], 7))
    with model.active():
        result = await serve(agent(meta), "A")
    assert result.metacognition.verdict.accepted
    assert json.loads(model.requests[1]["messages"][-1]["content"]) == 7


# ------------------------------------------------------------------ streaming


@function_tool
def get_weather(city: str) -> str:
    """Get the weather for a city."""
    return f"Sunny in {city}"


async def test_run_streamed_yields_items_as_the_loop_produces_them():
    weather = Agent(name="Assistant", instructions="Help.", tools=[get_weather])
    with ScriptedModel([call("get_weather", {"city": "Oslo"}), text("Sunny in Oslo.")]).active():
        result = Runner.run_streamed(weather, "Oslo?")
        events = [e async for e in result.stream_events()]
    assert [e.type for e in events][:1] == ["agent_updated_stream_event"]
    assert [e.name for e in events[1:]] == ["tool_called", "tool_output", "message_output_created"]
    assert events[2].item.output == "Sunny in Oslo" and events[2].item.call_id == "call_get_weather"
    assert result.is_complete and result.final_output == "Sunny in Oslo."


async def test_a_failed_streamed_run_raises_after_its_events():
    weather = Agent(name="Assistant", instructions="Help.", tools=[get_weather])
    with ScriptedModel([call("get_weather", {"city": "Oslo"})] * 2).active():
        result = Runner.run_streamed(weather, "Loop", max_turns=2)
        seen = []
        with pytest.raises(MaxTurnsExceeded):
            async for event in result.stream_events():
                seen.append(event.type)
    assert seen.count("run_item_stream_event") == 4


# ------------------------------------------------------------------ per-user metacognition


async def test_each_user_gets_their_own_procedures():
    def for_user(user: str) -> Metacognition:
        meta = make_meta()
        if user == "maya":
            meta.install("basket_total", V2)
        return meta

    shop = agent(PerUser(for_user))
    from helix.memory.base import MemoryContext

    model = ScriptedModel([*uses(["apple", "pear"], 7), text("7")])
    with model.active():
        maya = await Runner.run(shop, "What do apple, pear cost?", context=CASES["A"],
                                run_config=RunConfig(memory_context=MemoryContext(user_id="maya")))
        omar = await Runner.run(shop, "What do apple, pear cost?", context=CASES["A"],
                                run_config=RunConfig(memory_context=MemoryContext(user_id="omar")))
    assert maya.metacognition.offered == ["basket_total@1"] and omar.metacognition.offered == []
    assert shop.metacognition.by_user["maya"].store.count() == 1
    assert shop.metacognition.by_user["omar"].store.count() == 1


# ------------------------------------------------------------------ replayed practice


async def test_replay_practice_checks_answers_in_a_read_only_world():
    meta = make_meta(practice=ReplayPractice())
    shop = agent(meta)
    original = [call("price", {"item": "apple"}), call("price", {"item": "pear"}), call("tax"), text("7")]
    with ScriptedModel([*original, *uses(["apple", "pear"], 7), *uses(["apple", "pear"], 5)]).active():
        await serve(shop, "A")  # the agent works it out itself, and is accepted
        good = await meta._practice_for("basket_total")(V2, "A")
        stale = await meta._practice_for("basket_total")(V1, "A")
    assert good[:2] == (1.0, "the answer matches the accepted run")
    assert stale[0] == 0.0 and "answered '5' where the accepted run answered '7'" in stale[1]


@function_tool(read_only=True)
def stock(ctx: RunContextWrapper[dict], item: str) -> int:
    """How many of an item are on the shelf."""
    return ctx.context["stock"][item]


@function_tool
def place_order(ctx: RunContextWrapper[dict], item: str, quantity: int, note: str = "") -> dict:
    """Order more of an item."""
    ctx.context.setdefault("orders", []).append((item, quantity))
    return {"ordered": quantity}


@procedure(api=[stock])
def reorder_quantity(item: str) -> int:
    """How many to order to bring an item back to five on the shelf."""
    ...


async def test_replay_practice_compares_actions_and_never_runs_side_effects():
    meta = Metacognition([reorder_quantity], verifier=lambda ctx, result: True, practice=ReplayPractice(
        ignore={"place_order": ["note"]}), recall_k=1, learning="inline")
    shop = Agent(name="Stock", instructions="Keep five of each.", tools=[stock, place_order], metacognition=meta)
    world = {"stock": {"apple": 3}}
    right = "def reorder_quantity(api, item: str) -> int:\n    return 5 - api.stock(item)\n"
    wrong = "def reorder_quantity(api, item: str) -> int:\n    return 5\n"
    script = [call("stock", {"item": "apple"}), call("place_order", {"item": "apple", "quantity": 2, "note": "low"}),
              text("Ordered 2."),
              call("reorder_quantity", {"item": "apple"}), call("place_order", {"item": "apple", "quantity": 2}),
              text("Ordered."),
              call("reorder_quantity", {"item": "apple"}), call("place_order", {"item": "apple", "quantity": 5}),
              text("Ordered.")]
    with ScriptedModel(script).active():
        await Runner.run(shop, "Restock apples", context=world, run_config=RunConfig(case_id="apples"))
        good = await meta._practice_for("reorder_quantity")(right, "apples")
        bad = await meta._practice_for("reorder_quantity")(wrong, "apples")
    assert good[:2] == (1.0, "the actions match the accepted run")
    assert bad[0] == 0.0 and "quantity\": 5" in bad[1]
    assert world["orders"] == [("apple", 2)]  # only the real run ordered anything


# ------------------------------------------------------------------ events and concurrency


async def test_events_reach_bus_subscribers():
    seen = []
    get_bus().subscribe("metacognition", lambda event: seen.append((event.kind, event.procedure, event.version)))
    meta = make_meta()
    meta.install("basket_total", V1)
    shop = agent(meta)
    with ScriptedModel([*uses(["apple", "pear"], 5), *uses(["pear", "pear"], 6), text("7"), text("8"),
                        text(f"```python\n{V2}```"), text(f"```python\n{V2}```")]).active():
        for case_id in ("D1", "D2", "A", "B"):
            await serve(shop, case_id)
    await asyncio.sleep(0)  # let scheduled publishes run
    kinds = [kind for kind, _, _ in seen]
    assert kinds.count("run") == 4
    assert ("quarantined", "basket_total", 1) in seen and ("installed", "basket_total", 2) in seen


def respond(request: dict) -> Reply:
    last = request["messages"][-1]
    if last["role"] == "tool":
        return text(last["content"])
    items = re.search(r"What do (.*) cost\?", last["content"]).group(1).split(", ")
    return call("basket_total", {"items": items})


async def test_concurrent_runs_keep_one_ordered_record():
    meta = make_meta()
    meta.install("basket_total", V2)
    shop = agent(meta)
    model = RespondingModel(respond)
    with model.active():
        results = await asyncio.gather(*(serve(shop, case_id) for case_id in ["A", "B", "D3"] * 10))
    assert all(r.metacognition.verdict.accepted for r in results)
    assert sorted(r.metacognition.order for r in results) == list(range(30))
    assert meta.store.count() == 30 and {r["reasoning_effort"] for r in model.requests} == {"none"}
    assert verifier  # the basket verifier judged every run
