"""Chapter 7's listings from end to end, with scripted models in place of the paid ones.

A careful agent stands in for gpt-5.4-mini, a builder that returns known code stands in for the model that writes
code, and Jev answers from each task's truth. Nothing here is a result. The tests check each listing's control flow:
the gate, the search, the days and nights, and the note change in the middle of the week.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

from chapters.ch06.assistant.environment import SchedulingEnvironment, to_hhmm
from chapters.ch06.assistant.generator import _starts
from chapters.ch06.assistant.learning import MEMORIES, bootstrapped_source
from chapters.ch07.assistant import harness, requests
from chapters.ch07.assistant.components import hand_written_component
from helix.metacognition import recording
from tests.test_ch07_calendar_desk import Scripted

LISTINGS = Path(__file__).resolve().parents[1] / "chapters" / "ch07"
KINDS = [{"name": "book_meeting", "description": "Requests to book a meeting with people this week.",
          "guidance": "Call find_shared_starts once, then book with book_time.", "acts": True},
         {"name": "not_a_booking", "description": "Questions about meetings, or requests to cancel or move one.",
          "guidance": "Answer from the calendars and change nothing.", "acts": False}]
READS_OLD_NOTE = '        words = (note or "").split()\n'


def every_task() -> dict[str, dict[str, Any]]:
    names = ("experience", "search", "gate", "changed_gate", "held_out", "second_experience", "second_search",
             "second_gate", "second_held_out")
    tasks = [r for name in names for r in requests.load(name)]
    tasks += [r for rows in requests.week().values() for r in rows]
    return {task["text"]: task for task in tasks}


TASKS = every_task()


def relearned() -> str:
    """Chapter 6's first reference function, taught to read Dana's reworded note."""
    source = (MEMORIES / "find_shared_starts.py").read_text(encoding="utf-8")
    assert source.count(READS_OLD_NOTE) == 1
    return source.replace(READS_OLD_NOTE, '        if "half an hour" in (note or "").lower():\n'
                                          "            return 30\n" + READS_OLD_NOTE)


def reply(content: str = "", tool_calls: Any = (), prompt: int = 2500, completion: int = 120) -> Any:
    return recording._response({"content": content, "tool_calls": list(tool_calls),
                                "usage": {"prompt_tokens": prompt, "completion_tokens": completion}})


def bookings(task: dict[str, Any], starts: dict[str, list[str]]) -> list[dict[str, Any]]:
    truth = task["truth"]
    days = [day for day in truth["days"] if starts.get(day)][:truth["count"]]
    return [{"id": f"call_book_{n}", "name": "book_time",
             "arguments": json.dumps({"title": "Meeting", "day": day, "start": starts[day][0],
                                      "minutes": truth["minutes"], "attendees": truth["attendees"]})}
            for n, day in enumerate(days)]


def careful_agent(kwargs: dict[str, Any], careless: bool = False) -> Any:
    """Books what its function returns, works the week out itself without one, and never acts on a near miss.

    A careless one books on a near miss too, as the real agent sometimes does at no reasoning.
    """
    messages = kwargs["messages"]
    task = TASKS[next(m["content"] for m in messages if m["role"] == "user")]
    tools = [tool["function"]["name"] for tool in kwargs.get("tools") or []]
    results = [m for m in messages if m["role"] == "tool"]
    if "book_time" not in tools:
        if "find_shared_starts" in tools and not results:  # read-only: it can look, and then can do nothing
            looked = {"attendees": task["truth"]["attendees"], "days": task["truth"]["days"],
                      "minutes": task["truth"]["minutes"], "earliest": "", "latest_end": ""}
            return reply(tool_calls=[{"id": "call_find", "name": "find_shared_starts",
                                      "arguments": json.dumps(looked)}])
        return reply("I can't change the calendar from here.")
    if task["expect"] == "handoff" and not careless:
        return reply("Nothing in this request is a booking, so nothing was changed.")
    truth = task["truth"]
    asked = {"attendees": truth["attendees"], "days": truth["days"], "minutes": truth["minutes"],
             "earliest": truth.get("earliest") or "", "latest_end": truth.get("latest_end") or ""}
    if task["kind"] == "same time":  # one start that works on every day, booked on each of them
        if not results:
            return reply(tool_calls=[{"id": "call_find", "name": "find_common_starts", "arguments": json.dumps(asked)}])
        if len(results) > 1:
            return reply("Booked.")
        starts = json.loads(results[0]["content"]).get("starts") or []
        return reply(tool_calls=bookings(task, {day: starts for day in truth["days"]})) if starts else reply("None.")
    if "find_shared_starts" not in tools:
        if results:
            return reply("Booked.")
        found = _starts(SchedulingEnvironment(task), truth)
        return reply(tool_calls=bookings(task, {day: [to_hhmm(s) for s in starts] for day, starts in found.items()}))
    if not results:
        return reply(tool_calls=[{"id": "call_find", "name": "find_shared_starts", "arguments": json.dumps(asked)}])
    if len(results) == 1:
        said = results[0]["content"]
        made = bookings(task, (json.loads(said) if said.strip().startswith("{") else {}).get("starts") or {})
        return reply(tool_calls=made) if made else reply("Nobody is free together.")
    return reply("Booked.")


def builder(kwargs: dict[str, Any]) -> Any:
    """A rewrite is the hand-written component, a seed's question gets kinds, and a function is the relearned one."""
    text = "\n".join(str(m.get("content")) for m in kwargs["messages"])
    if "You are improving a learning component" in text[:400]:
        return reply(hand_written_component(), prompt=30000, completion=3000)
    if "List the kinds of request these runs show" in text:
        return reply(json.dumps(KINDS), prompt=20000, completion=400)
    return reply("```python\n" + relearned() + "\n```", prompt=8000, completion=900)


class Models:
    live_calls = 0

    def __init__(self, careless: bool = False) -> None:
        self.careless = careless
        self.builder_calls = 0

    async def __call__(self, **kwargs: Any) -> Any:
        if "claude" in str(kwargs.get("model")):
            self.builder_calls += 1
            return builder(kwargs)
        return careful_agent(copy.deepcopy(kwargs), self.careless)

    @contextmanager
    def active(self) -> Iterator[Models]:
        recording.install()
        token = recording._transport.set(self)
        try:
            yield self
        finally:
            recording._transport.reset(token)


class Jev:
    name = "scripted"

    async def ask_with_usage(self, state: Any, questions: dict[str, Any]) -> Any:
        return await Scripted(TASKS[state["request"]]).ask_with_usage(state, questions)


@pytest.fixture
def listing(monkeypatch, tmp_path):
    """Load a listing by file name and run it on the scripted models; returns what it saved."""
    monkeypatch.setattr(harness, "kept_procedure", lambda: (bootstrapped_source(), "reference"))

    async def run(name: str, saved: str, **flags: Any) -> dict[str, Any]:
        spec = importlib.util.spec_from_file_location(f"listing_{name[:2]}", LISTINGS / name)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.recordings = lambda record, budget: (Models(), Jev())
        flags = {"population": 6, "generations": 1, **flags} if name.startswith("03") else flags
        await module.main(argparse.Namespace(record=False, budget=0.0, output=tmp_path, **flags))
        return json.loads((tmp_path / saved).read_text(encoding="utf-8"))

    return run


def served(summary: dict[str, Any], stage: str, kind: str) -> dict[str, int]:
    return summary[stage]["kinds"][kind]["served_by"]


async def test_listing_2_measures_the_gap_with_and_without_the_hand_written_component(listing):
    out = await listing("02_service_gap.py", "service_gap.json")
    summary = out["summary"]
    assert list(summary) == ["model at none", "model at high", "hand-written component"]
    assert summary["model at none"]["gap"] == pytest.approx(0.4)  # right, but the model did all of it
    component = summary["hand-written component"]
    assert served(summary, "hand-written component", "one meeting") == {"program": 24}
    assert served(summary, "hand-written component", "two meetings") == {"model at none": 8}
    assert served(summary, "hand-written component", "near miss") == {"read-only": 20}
    assert (component["right"], component["wrong_actions"]) == (52, 0)
    assert component["gap"] == pytest.approx(8 * 0.4 / 52, abs=0.001)
    assert [m["route"] for m in out["memories"]] == ["program", "none", "none"]
    by_program = [row for row in out["rows"] if row["outcome"] == "program"]
    assert {row["memory"] for row in by_program} == {"one_meeting"}


async def test_listing_3_searches_from_the_model_only_seed_and_installs_what_the_gate_admits(listing, tmp_path):
    out = await listing("03_code_search.py", "code_search_dgm.json", method="dgm")
    outcomes = [row["outcome"] for row in out["search"]]
    assert outcomes == ["refused on its own cases", "admitted"]  # the seed, then the first rewrite
    assert any(event.startswith("gap: the LLM spends") for event in out["events"])
    assert (tmp_path / "component_dgm.py").read_text(encoding="utf-8").strip() == hand_written_component().strip()
    assert served(out["summary"], "experience", "one meeting") == {"model at none": 48}
    assert served(out["summary"], "searched component", "one meeting") == {"program": 24}
    assert out["summary"]["searched component"]["wrong_actions"] == 0


async def test_listing_3_runs_the_same_search_with_gepa_when_the_method_is_swapped(listing, tmp_path):
    out = await listing("03_code_search.py", "code_search_gepa.json", method="gepa", population=2, generations=1)
    outcomes = [row["outcome"] for row in out["search"]]
    assert outcomes[0] == "refused on its own cases" and "admitted" in outcomes[1:]  # the seed, then its rewrites
    assert (tmp_path / "component_gepa.py").exists()
    assert served(out["summary"], "searched component", "one meeting") == {"program": 24}


async def test_listing_4_serves_by_day_improves_by_night_and_recovers_from_the_changed_note(listing):
    out = await listing("04_online_week.py", "online_week.json")
    nights, summary = out["nights"], out["summary"]
    assert any("passed, installed" in event for event in nights["Monday"])
    assert nights["Tuesday"] == [] and nights["Thursday"] == []
    assert served(summary, "Monday", "one meeting") == {"model at none": 48}
    assert served(summary, "Tuesday", "one meeting") == {"program": 12}
    # Wednesday: two bookings go wrong before the program's action is switched off, the stale function is
    # quarantined, and the loop relearns it during the day.
    assert summary["Wednesday"]["wrong_actions"] == 2
    events = [event for row in out["rows"] if row["stage"] == "Wednesday" for event in row["events"]]
    assert any(event.startswith("program: switched off book_time") for event in events)
    assert any(event.startswith("detector: quarantined find_shared_starts") for event in events)
    assert any("wrote find_shared_starts" in event and "installed" in event for event in events)
    # That night the program is tested again on cases from the changed weeks, and it serves again on Thursday.
    assert any("passed, switched back on: book_time" in event for event in nights["Wednesday"])
    assert served(summary, "Thursday", "one meeting") == {"program": 6}
    assert summary["Thursday"]["right"] == summary["Thursday"]["requests"]


async def test_listing_5_applies_the_found_component_to_a_second_kind_of_work(listing, tmp_path):
    component = tmp_path / "found_component.py"
    component.write_text(hand_written_component(), encoding="utf-8")
    out = await listing("05_generalization.py", "generalization.json", component=component)
    summary = out["summary"]
    assert list(summary) == ["model at none", "experience", "found component"]
    assert any("found_component wrote" in event and "installed" in event for event in out["events"])
    # Bookings of one meeting leave the model, and work that needs several actions stays on it.
    assert served(summary, "found component", "one meeting") == {"program": 24}
    assert served(summary, "found component", "same time") == {"model at none": 32}
    assert served(summary, "found component", "two meetings") == {"model at none": 8}
    assert served(summary, "found component", "near miss") == {"read-only": 20}
    assert summary["found component"]["wrong_actions"] == 0
    assert summary["found component"]["gap"] < summary["model at none"]["gap"]


# The first paid run of listing 7.2 quarantined the scheduling function and paid for a relearning nobody needed. Two
# kinds of rejected run had been held against the function, and neither says anything about it.

READ_ONLY_ROUTER = 'async def decide(request, kit):\n    return {"handoff": "not a booking", "read_only": True}\n'


def test_listing_5s_harness_cannot_quarantine_a_function(monkeypatch):
    """Its gate holds cases for the first function only, so a relearning of the second could never pass."""
    monkeypatch.setattr(harness, "kept_procedure", lambda: (bootstrapped_source(), "reference"))
    assert harness.harness(second_class=True)[1].gating_sources == ()
    assert harness.harness()[1].gating_sources == ("verifier",)


async def test_acting_on_near_misses_is_not_held_against_the_scheduling_function(monkeypatch):
    monkeypatch.setattr(harness, "kept_procedure", lambda: (bootstrapped_source(), "reference"))
    agent, meta = harness.harness()
    near_misses = [r for r in requests.load("experience") if r["expect"] == "handoff"][:6]
    models, log = Models(careless=True), []
    with models.active():
        await harness.serve(agent, near_misses, "careless", log)
    assert [row["right"] for row in log] == [False] * 6  # it booked every time
    assert meta.store.trusted("find_shared_starts") is not None and models.builder_calls == 0


async def test_work_sent_on_read_only_is_not_held_against_the_scheduling_function(monkeypatch):
    monkeypatch.setattr(harness, "kept_procedure", lambda: (bootstrapped_source(), "reference"))
    tier = harness.decision_tier(Jev(), [("sends_everything_on", "")])
    agent, meta = harness.harness(tier)
    meta.store.add_version("decision_program", READ_ONLY_ROUTER, state="trusted", created_by="test")
    bookings_wanted = [r for r in requests.load("experience") if r["expect"] == "act"][:6]
    models, log = Models(), []
    with models.active():
        await harness.serve(agent, bookings_wanted, "lost", log)
    assert [row["outcome"] for row in log] == ["lost"] * 6  # nothing could book, so nothing was done
    assert meta.store.trusted("find_shared_starts") is not None and models.builder_calls == 0
