"""Scripted stand-ins for the three models of the calendar desk, and the requests they are scripted from.

No model is called. The agent books what its function returns and works the week out itself when it has none, the
builder returns known code, and the decision model answers from each request's truth. Each can be made to misbehave
the way the real ones did: the agent acts on a share of near misses and gets a share of the harder requests wrong
when it has no function for them, the decision model misreads a share of near misses and sends a share of requests
to the wrong function, and the builder writes a function that reads only the note in force when it was asked, or
now and then the note before it.

These runs test the harness: its limits, its state, and its recovery. They say nothing about how a model behaves.
"""

from __future__ import annotations

import asyncio
import json
import zlib
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from chapters.ch06.assistant.environment import SchedulingEnvironment, to_hhmm
from chapters.ch06.assistant.generator import _starts
from chapters.ch06.assistant.generator_same_time import common
from chapters.ch06.assistant.learning import MEMORIES
from chapters.ch07.assistant.components import hand_written_component
from helix.deciders import Answer, DecisionUsage
from helix.metacognition import recording
from helix.metacognition.exceptions import Unavailable

KINDS = [{"name": "book_meeting", "description": "Requests to book a meeting with people this week.",
          "guidance": "Call find_shared_starts once, then book with book_time.", "acts": True},
         {"name": "not_a_booking", "description": "Questions about meetings, or requests to cancel or move one.",
          "guidance": "Answer from the calendars and change nothing.", "acts": False}]
READS_THE_NUMBER = '        words = (note or "").split()\n'
TASKS: dict[str, dict[str, Any]] = {}  # every request by its text, which carries a reference that makes it unique
NOW = {"conditions": "ordinary", "before": "ordinary"}  # the note in force, and the one before it
# What a function written under each note looks for in it. None reads the number, as Chapter 6's function does.
READS = {"ordinary": None, "fifteen": None, "changed": ("half an hour", 30), "half_hour": ("half an hour", 30),
         "three_quarters": ("three quarters of an hour", 45)}


def one_in(task_id: str, share: float, salt: str = "") -> bool:
    """Whether this request falls in a fixed share of all requests: the same answer on every run."""
    return zlib.crc32(f"{salt}{task_id}".encode()) % 10_000 < share * 10_000


def function_for(conditions: str, name: str = "find_shared_starts") -> str:
    """Chapter 6's function for `name`, reading only the note it was written under, as a literal learner's would."""
    source = (MEMORIES / f"{name}.py").read_text(encoding="utf-8")
    reads = READS[conditions]
    if reads is None:
        return source
    assert source.count(READS_THE_NUMBER) == 1
    words, minutes = reads
    return source.replace(READS_THE_NUMBER, f'        if "{words}" in (note or "").lower():\n'
                                            f"            return {minutes}\n        return 0\n" + READS_THE_NUMBER)


class World(SchedulingEnvironment):
    """A week whose calendar can be made unreadable, as a dependency that fails would leave it."""

    def __init__(self, task: dict[str, Any]) -> None:
        super().__init__(task)
        self.style = task.get("style", "plain")
        self.down = bool(task.get("outage"))

    def read_calendar(self, person: str = "", day: str = "") -> dict[str, Any]:
        if self.down:
            raise Unavailable("the calendar service did not answer")
        return super().read_calendar(person, day)


def _reply(content: str = "", tool_calls: Any = (), prompt: int = 2500, completion: int = 120) -> Any:
    return recording._response({"content": content, "tool_calls": list(tool_calls),
                                "usage": {"prompt_tokens": prompt, "completion_tokens": completion}})


def _bookings(task: dict[str, Any], starts: Any) -> list[dict[str, Any]]:
    """The bookings a request asks for, from the starts a function found: by day, or one start for every day."""
    truth = task["truth"]
    if truth.get("same_start"):
        each = [(day, starts[0]) for day in truth["days"]] if isinstance(starts, list) and starts else []
    else:
        starts = starts if isinstance(starts, dict) else {}
        each = [(day, starts[day][0]) for day in truth["days"] if starts.get(day)][:truth["count"]]
    return [{"id": f"call_book_{n}", "name": "book_time",
             "arguments": json.dumps({"title": "Meeting", "day": day, "start": start,
                                      "minutes": truth["minutes"], "attendees": truth["attendees"]})}
            for n, (day, start) in enumerate(each)]


def _worked_out(task: dict[str, Any]) -> Any:
    """The starts the agent finds by reading the calendars itself."""
    found = _starts(SchedulingEnvironment(task), task["truth"])
    if task["truth"].get("same_start"):
        return [to_hhmm(start) for start in common(found)]
    return {day: [to_hhmm(start) for start in starts] for day, starts in found.items()}


class Models:
    """The agent and the builder. `careless` is the share of near misses the agent acts on when it can, and
    `struggles` the share of requests for the same time on several days it books wrongly when it has no function
    for them. `stale` is the share of functions the builder writes for the note before the one in force."""

    live_calls = 0

    def __init__(self, careless: float = 0.2, agent_seconds: float = 0.0, builder_seconds: float = 0.0,
                 struggles: float = 0.0, stale: float = 0.0) -> None:
        self.careless, self.agent_seconds, self.builder_seconds = careless, agent_seconds, builder_seconds
        self.struggles, self.stale = struggles, stale
        self.calls: Counter[str] = Counter()

    async def __call__(self, **kwargs: Any) -> Any:
        if "claude" in str(kwargs.get("model")):
            self.calls["builder"] += 1
            if self.builder_seconds:
                await asyncio.sleep(self.builder_seconds)
            return self._builder(kwargs)
        self.calls["agent"] += 1
        if self.agent_seconds:
            await asyncio.sleep(self.agent_seconds)
        return self._agent(kwargs)

    def _agent(self, kwargs: dict[str, Any]) -> Any:
        messages = kwargs["messages"]
        task = TASKS[next(m["content"] for m in messages if m["role"] == "user")]
        tools = [tool["function"]["name"] for tool in kwargs.get("tools") or []]
        results = [m for m in messages if m["role"] == "tool"]
        truth = task["truth"]
        if "book_time" not in tools:
            return _reply("I can't change the calendar from here.")
        if task["expect"] == "handoff" and not one_in(task["id"], self.careless, "careless"):
            return _reply("Nothing in this request is a booking, so nothing was changed.")
        function = "find_common_starts" if truth.get("same_start") else "find_shared_starts"
        if function not in tools:  # no trusted function for this kind of request: it works the week out itself
            if results:
                return _reply("Booked.")
            if task.get("outage"):
                return _reply("I could not read the calendars, so nothing was booked.")
            made = _bookings(task, _worked_out(task))
            if truth.get("same_start") and one_in(task["id"], self.struggles, "struggles"):
                made = made[:1]  # it books the first day and misses that every day was asked for
            return _reply(tool_calls=made) if made else _reply("Nobody is free together.")
        if not results:
            asked = {"attendees": truth["attendees"], "days": truth["days"], "minutes": truth["minutes"],
                     "earliest": truth.get("earliest") or "", "latest_end": truth.get("latest_end") or ""}
            return _reply(tool_calls=[{"id": "call_find", "name": function, "arguments": json.dumps(asked)}])
        if len(results) == 1:
            said = results[0]["content"]
            found = json.loads(said) if said.strip().startswith("{") else {}
            if "withdrawn" in str(found.get("error")):  # under its hands: it works the week out itself, as told
                found = {"starts": _worked_out(task)}
            made = _bookings(task, found.get("starts"))
            return _reply(tool_calls=made) if made else _reply("Nobody is free together.")
        return _reply("Booked.")

    def _builder(self, kwargs: dict[str, Any]) -> Any:
        text = "\n".join(str(m.get("content")) for m in kwargs["messages"])
        if "You are improving a learning component" in text[:400]:
            self.calls["builder: rewrite"] += 1
            return _reply(hand_written_component(), prompt=30000, completion=3000)
        if "List the kinds of request these runs show" in text:
            self.calls["builder: kinds"] += 1
            return _reply(json.dumps(KINDS), prompt=20000, completion=400)
        self.calls["builder: function"] += 1
        name = "find_common_starts" if "def find_common_starts" in text else "find_shared_starts"
        late = one_in(f"{name}{self.calls['builder: function']}", self.stale, "stale")
        written = function_for(NOW["before"] if late else NOW["conditions"], name)
        return _reply("```python\n" + written + "\n```", prompt=8000, completion=900)

    @contextmanager
    def active(self) -> Iterator[Models]:
        recording.install()
        token = recording._transport.set(self)
        try:
            yield self
        finally:
            recording._transport.reset(token)


class Jev:
    """The decision model, answering from the request's truth. `misread` is the share of near misses it takes for
    a booking, which makes a decision program act wrongly. `misroute` is the share of requests it sends to the
    wrong function when asked which one a request needs."""

    name = "scripted"

    def __init__(self, misread: float = 0.0, seconds: float = 0.0, misroute: float = 0.0) -> None:
        self.misread, self.seconds, self.misroute = misread, seconds, misroute
        self.calls = 0

    async def ask_with_usage(self, state: Any, questions: dict[str, Any]) -> tuple[dict[str, Answer], DecisionUsage]:
        self.calls += 1
        if self.seconds:
            await asyncio.sleep(self.seconds)
        task = TASKS[state["request"]]
        truth = task["truth"]
        books = task["expect"] == "act" or one_in(task["id"], self.misread, "misread")
        answers = {}
        for name in questions:
            if name == "books":
                answers[name] = Answer("probability", 0.95 if books else 0.05)
            elif name == "several":
                answers[name] = Answer("probability", 0.95 if truth["count"] > 1 else 0.05)
            elif name == "limit":
                answers[name] = Answer("choice", "earliest" if truth.get("earliest") else "latest_end", 0.9)
            elif name == "procedure":  # which function the request needs
                same = bool(truth.get("same_start")) != one_in(task["id"], self.misroute, "misroute")
                answers[name] = Answer("choice", "find_common_starts" if same else "find_shared_starts", 0.9)
            else:
                answers[name] = Answer("probability", 0.95 if name.removeprefix("on_") in truth["days"] else 0.05)
        return answers, DecisionUsage(calls=1, input_tokens=100, dollars=100 * 42 / 1e9, seconds=0.15)
