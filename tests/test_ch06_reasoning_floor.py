"""Chapter 6, stage 0 and 1: the task class, the floor, skill memory, and recordings."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from chapters.ch06.assistant.environment import SchedulingEnvironment, to_hhmm, to_minutes
from chapters.ch06.assistant.generator import (CONVENTIONS, SPLITS, USER, face_value_starts, generate,
                                              solutions, task_set)
from chapters.ch06.assistant.harness import (BOOTSTRAPPED_MEMORY, MEMORIES, PRICES, TEXT_PROCEDURE,
                                            ReadOnlyCalendar, install_memory, run_task)
from chapters.ch06.assistant.recording import Recording
from chapters.ch06.assistant.tools import make_tools
from helix.agent import _usage_of
from helix.eval.effort import ADEQUATE, SOLID, EffortSweep, ReasoningFloorSignal, sweep
from helix.memory.base import MemoryContext
from helix.memory.code_skills import UnsafeSkill, check_source, recalled_skill_tools, skill_tool
from helix.signals.cost import call_dollars


def _usage(prompt=1000, completion=200, reasoning=0, cached=0):
    return SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion,
                           completion_tokens_details=SimpleNamespace(reasoning_tokens=reasoning),
                           prompt_tokens_details=SimpleNamespace(cached_tokens=cached))


def _reply(content="", call=None, number=0, reasoning=0):
    calls = None if call is None else [SimpleNamespace(
        id=f"call-{number}", type="function",
        function=SimpleNamespace(name=call[0], arguments=json.dumps(call[1])))]
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=calls))],
                           usage=_usage(reasoning=reasoning))


def _careless_start(task):
    """The first start found by reading the calendars at face value, which
    the generator guarantees is wrong; None when face value finds nothing."""
    naive = face_value_starts(task)
    return next(((d, to_hhmm(naive[d][0])) for d in task["truth"]["days"] if naive[d]), None)


def scripted_model(tasks):
    """A stand-in model: careful with the skill or at high effort, otherwise
    it reads the calendars at face value. It never sees the truth; the
    plan it follows is computed from the calendars, as a careful model would."""
    by_text = {t["text"]: t for t in tasks}

    async def model(**kwargs):
        messages = kwargs["messages"]
        task = by_text[messages[1]["content"]]
        system = messages[0]["content"] if isinstance(messages[0]["content"], str) else messages[0]["content"][0]["text"]
        careful = "## Learned procedures" in system or kwargs.get("reasoning_effort") == "high"
        done = sum(1 for m in messages if m["role"] == "tool")
        truth = task["truth"]
        if careful:
            plan = []
            for day, starts in solutions(task).items():
                plan.append((day, starts[0]))
            plan = plan[:truth["count"]]
        else:
            careless = _careless_start(task)
            plan = [careless] * truth["count"] if careless else []
        if done < len(plan):
            day, start = plan[done]
            return _reply(call=("book_time", {"title": "Meeting", "day": day, "start": start,
                                              "minutes": truth["minutes"], "attendees": truth["attendees"]}),
                          number=done, reasoning=800 if careful and "## Learned" not in system else 0)
        return _reply("Booked.")

    return model


def test_generator_gives_the_same_task_for_the_same_seed():
    assert generate(3) == generate(3)
    assert generate(3)["text"] != generate(4)["text"]


def test_every_generated_task_has_a_solution_and_a_face_value_trap():
    for split in SPLITS:
        for task in task_set(split):
            found = solutions(task)
            assert len(found) >= task["truth"]["count"]
            careless = _careless_start(task)
            assert careless is None or careless[1] not in found.get(careless[0], [])


def test_each_convention_decides_tasks_in_every_set():
    for split in SPLITS:
        tasks = task_set(split)
        for convention in CONVENTIONS:
            assert sum(convention in t["decided_by"] for t in tasks) >= 4


def test_checker_accepts_a_solution_and_refuses_the_face_value_answer():
    task = next(t for t in task_set("dev") if _careless_start(t) is not None)
    day, starts = next(iter(solutions(task).items()))
    env = SchedulingEnvironment(task)
    env.book_time("Sync", day, starts[0], task["truth"]["minutes"], task["truth"]["attendees"])
    assert env.check() == {"success": True, "reason": "accepted"}

    trap = SchedulingEnvironment(task)
    trap_day, trap_start = _careless_start(task)
    trap.book_time("Sync", trap_day, trap_start, task["truth"]["minutes"], task["truth"]["attendees"])
    assert not trap.check()["success"]


def _tiny_world(events, *, offsets=None, external=()):
    people = {"maya": "Maya Rivera", "dana": "Dana Park"}
    profiles = {pid: {"utc_offset": (offsets or {}).get(pid, "-04:00"), "hours": ["09:00", "17:00"]}
                for pid in people}
    for pid in external:
        profiles[pid].update(external=True, note="Please leave 15 minutes between any meeting with me and my other meetings.")
    week = {d: f"2026-10-{19 + i}" for i, d in enumerate(("mon", "tue", "wed", "thu", "fri"))}
    return {"id": "T", "text": "", "truth": {"kind": "booking", "attendees": ["Dana Park"], "minutes": 30,
                                             "count": 1, "days": ["mon"]},
            "world": {"user": "maya", "week": week, "hours": ["09:00", "17:00"], "people": people,
                      "profiles": profiles, "events": events}}


def _event(start, minutes=60, show_as="busy"):
    return {"id": f"E{start}", "owner": "dana", "title": "x", "day": "mon", "start": start,
            "minutes": minutes, "show_as": show_as, "attendees": ["dana"]}


def test_an_event_shown_as_free_does_not_block():
    env = SchedulingEnvironment(_tiny_world([_event("10:00", show_as="free")]))
    assert env.acceptable("mon", 10 * 60, ["dana"], 30) is None


def test_an_external_guest_gets_their_buffer():
    env = SchedulingEnvironment(_tiny_world([_event("10:00")], external=["dana"]))
    assert "15 minutes between meetings" in env.acceptable("mon", 11 * 60, ["dana"], 30)
    assert env.acceptable("mon", 11 * 60 + 15, ["dana"], 30) is None


def test_calendars_show_local_time_and_meetings_respect_working_hours():
    env = SchedulingEnvironment(_tiny_world([_event("13:00")], offsets={"dana": "-07:00"}))
    page = env.read_calendar("Dana Park", "mon")
    assert page["utc_offset"] == "-07:00" and page["working_hours"] == "09:00-17:00"
    assert page["days"][0]["events"][0]["start"] == "10:00"  # 13:00 for Maya is 10:00 for Dana
    assert "outside their working hours" in env.acceptable("mon", 9 * 60, ["dana"], 30)
    assert env.acceptable("mon", 12 * 60, ["dana"], 30) is None


def test_other_peoples_calendars_show_events_but_not_their_details():
    task = generate(2)
    env = SchedulingEnvironment(task)
    shown = env.read_calendar(task["truth"]["attendees"][0], task["truth"]["days"][0])
    events = [e for d in shown["days"] for e in d["events"]]
    assert events and all("id" not in e and "attendees" not in e and "show_as" in e for e in events)
    assert "free" not in shown["days"][0]


def test_book_time_description_no_longer_teaches_the_procedure():
    tools = {t.name: t for t in make_tools(SchedulingEnvironment(generate(1)))}
    assert "attendees' calendars" not in tools["book_time"].description
    assert set(tools) == {"read_calendar", "book_time", "cancel_event"}


def test_usage_records_reasoning_and_cached_tokens():
    usage = _usage_of(SimpleNamespace(usage=_usage(1000, 300, reasoning=250, cached=600)))
    assert usage == {"prompt_tokens": 1000, "completion_tokens": 300, "total_tokens": 1300,
                     "reasoning_tokens": 250, "cached_tokens": 600}


def test_cached_input_is_priced_at_the_cached_rate_when_given():
    usage = {"prompt_tokens": 1_000_000, "completion_tokens": 0, "cached_tokens": 600_000}
    assert call_dollars("m", usage, {"m": (1.0, 4.0, 0.1)}) == pytest.approx(0.4 + 0.06)
    assert call_dollars("m", usage, {"m": (1.0, 4.0)}) == pytest.approx(1.0)


def test_skill_memory_reaches_the_model_and_the_effort_setting_reaches_the_call():
    task = generate(1)
    seen = []

    async def model(**kwargs):
        seen.append(kwargs)
        return _reply("Done.")

    async def go():
        skills = await install_memory(TEXT_PROCEDURE)
        with patch("helix.agent.helix_acompletion", side_effect=model):
            return await run_task(task, "low", memory=skills)

    solved, reason, trajectory = asyncio.run(go())
    system = seen[0]["messages"][0]["content"]
    system = system if isinstance(system, str) else system[0]["text"]
    assert "## Learned procedures" in system and "Read the calendar of every attendee" in system
    assert seen[0]["reasoning_effort"] == "low"
    assert trajectory.metadata["memory_recalled"][0]["tier"] == "procedural"
    assert not solved and "booked 0" in reason


def test_sweep_finds_a_lower_floor_with_the_skill_and_the_signal_measures_the_gap():
    tasks = task_set("dev")

    async def go():
        skills = await install_memory(TEXT_PROCEDURE)
        with patch("helix.agent.helix_acompletion", side_effect=scripted_model(tasks)):
            bare = await sweep("no_memory", tasks, lambda t, e: run_task(t, e), prices=PRICES)
            skilled = await sweep("bootstrap_skill", tasks, lambda t, e: run_task(t, e, memory=skills),
                                  prices=PRICES)
        return bare, skilled

    bare, skilled = asyncio.run(go())
    assert bare.floor(ADEQUATE) == "high" and bare.floor(SOLID) == "high"
    assert skilled.floor(ADEQUATE) == "none" and skilled.floor(SOLID) == "none"
    assert bare.cells["high"].mean("reasoning_tokens") > skilled.cells["none"].mean("reasoning_tokens")

    signal = ReasoningFloorSignal(target="low")
    assert signal.measure_sweep(skilled).score == 1.0 and not signal.measure_sweep(skilled).triggered
    gap = signal.measure_sweep(bare)
    assert gap.triggered and gap.raw_value == 2.0 and gap.score == pytest.approx(1 / 3)
    assert EffortSweep.from_dict(json.loads(json.dumps(bare.to_dict()))).floor() == "high"


def test_floor_is_not_reached_when_no_setting_meets_the_threshold():
    result = EffortSweep("x")
    from helix.eval.effort import EffortCell, RunRecord

    result.cells["high"] = EffortCell("high", [RunRecord(f"t{i}", "high", i < 6) for i in range(8)])
    assert result.floor(ADEQUATE) is None
    assert ReasoningFloorSignal(target="low").measure_sweep(result).score == 0.0


def test_a_recorded_run_replays_exactly_with_its_recorded_seconds(tmp_path):
    tasks = task_set("dev")[:2]
    path = tmp_path / "recording.jsonl"
    live = scripted_model(tasks)

    async def go(mode):
        recording = Recording(path, mode=mode)
        with patch("helix.llm_call.acompletion", side_effect=live), recording.active():
            out = [await run_task(t, "medium") for t in tasks]
        return recording, out

    recorded, first = asyncio.run(go("record"))
    assert recorded.live_calls > 0
    replayed, second = asyncio.run(go("replay"))
    assert replayed.live_calls == 0
    assert [(s, r) for s, r, _ in first] == [(s, r) for s, r, _ in second]
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    steps = [s for s in second[0][2].steps if s.kind.value == "model_call"]
    assert [s.payload["seconds"] for s in steps] == [r["seconds"] for r in rows[:len(steps)]]


def test_replay_fails_loudly_on_a_call_the_recording_lacks(tmp_path):
    async def go():
        with Recording(tmp_path / "empty.jsonl").active():
            return await run_task(generate(1), "none")

    with pytest.raises(KeyError, match="no recorded reply"):
        asyncio.run(go())


def _source():
    return (MEMORIES / BOOTSTRAPPED_MEMORY["file"]).read_text(encoding="utf-8")


def test_the_bootstrapped_memory_finds_exactly_the_starts_the_checker_accepts():
    for task in task_set("dev"):
        env = SchedulingEnvironment(task)
        tool = skill_tool("find_shared_starts", "", _source(), bind={"api": ReadOnlyCalendar(env)})
        truth = task["truth"]
        result = asyncio.run(tool.call(attendees=truth["attendees"], days=truth["days"], minutes=truth["minutes"],
                                       earliest=truth.get("earliest", ""), latest_end=truth.get("latest_end", "")))
        assert result == {"starts": solutions(task)}


def test_recalled_memory_reaches_the_model_as_a_function_and_a_line_of_context():
    task = generate(1)
    seen = []

    async def model(**kwargs):
        seen.append(kwargs)
        return _reply("Done.")

    async def go():
        skills = await install_memory(BOOTSTRAPPED_MEMORY)
        with patch("helix.agent.helix_acompletion", side_effect=model):
            return await run_task(task, "none", memory=skills)

    asyncio.run(go())
    names = [spec["function"]["name"] for spec in seen[0]["tools"]]
    assert "find_shared_starts" in names
    parameters = next(s["function"]["parameters"] for s in seen[0]["tools"]
                      if s["function"]["name"] == "find_shared_starts")
    assert "api" not in parameters["properties"]  # bound by the harness, never seen by the model
    system = seen[0]["messages"][0]["content"]
    system = system if isinstance(system, str) else system[0]["text"]
    assert "Use the tool `find_shared_starts`." in system and "def find_shared_starts" not in system


def test_the_guard_refuses_imports_underscores_and_code_that_runs_on_load():
    good = "def f(x: int) -> int:\n    return x + 1\n"
    check_source(good, "f")
    for bad in ("import os\n" + good,
                "def f(x: int) -> int:\n    return x.__class__\n",
                "def f(x):\n    return open('a')\n" if False else "def f(x):\n    from os import path\n",
                "y = sum(range(10 ** 12))\n" + good,
                "f = lambda x: x\n"):
        with pytest.raises(UnsafeSkill):
            check_source(bad, "f")
    with pytest.raises(UnsafeSkill, match="does not define"):
        check_source(good, "g")


def test_the_guard_runs_skills_without_dangerous_builtins():
    tool = skill_tool("f", "", "def f() -> str:\n    return open('x').read()\n")
    result = asyncio.run(tool.call())
    assert "NameError" in result["error"]


def test_a_failing_or_slow_skill_returns_its_error_as_the_tool_result():
    broken = skill_tool("f", "", "def f(x: int) -> int:\n    return x // 0\n")
    assert "ZeroDivisionError" in asyncio.run(broken.call(x=1))["error"]
    slow = skill_tool("f", "", "def f() -> int:\n    n = 0\n    while True:\n        n += 1\n", timeout=0.2)
    assert "longer than" in asyncio.run(slow.call())["error"]


def test_only_a_recalled_function_becomes_callable():
    context = MemoryContext(user_id="maya")

    async def go():
        text = await install_memory(TEXT_PROCEDURE)
        code = await install_memory(BOOTSTRAPPED_MEMORY)
        await code.save_skill("unrelated_helper", "Summarize an email thread.", "def unrelated_helper() -> str:\n"
                              "    return ''\n", body_kind="python")
        request = generate(1)["text"]
        return (await recalled_skill_tools(text, request, context, k=1),
                await recalled_skill_tools(code, request, context, k=1))

    from_text, from_code = asyncio.run(go())
    assert from_text == [] and [t.name for t in from_code] == ["find_shared_starts"]
