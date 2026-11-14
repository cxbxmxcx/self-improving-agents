"""Scripted stand-ins for the inbox's three models, for a soak that calls no model and spends nothing.

The agent files what the policy asks for the message's true topic, and misbehaves the way a real one may: it names a
topic of another team for a share of messages, files a ticket for a share of messages outside the bank's services,
and when it has no routing function it forgets the premium rule now and then. The builder writes a routing function
that knows only the policy version in force when it was asked, or now and then the version before. The decision model
answers from each message's truth, and misreads a share of the messages outside the bank's services.
"""

from __future__ import annotations

import asyncio
import json
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from helix.deciders import Answer, DecisionUsage
from helix.metacognition import recording

from examples.inbox.desk import PROGRAMS, hand_written_component
from examples.inbox.policy import HELP, POLICIES, TEAMS, TOPICS
from examples.inbox.stream import TASKS
from examples.inbox.world import customer
from examples.scale.scripted import _reply, one_in

NOW = {"policy": "2026-1", "before": "2026-1"}  # the policy version in force, and the one before it
KINDS = [{"name": "file_ticket", "description": "Messages about the bank's services that need a ticket.",
          "guidance": "Name the topic, call route_ticket, file what it says.", "acts": True},
         {"name": "no_ticket", "description": "Messages the help pages answer, or not about the bank at all.",
          "guidance": "Answer and file nothing.", "acts": False}]


def literal_function(policy_key: str) -> str:
    """A route_ticket that learned one version of the policy by heart: its table, urgency and premium rule."""
    policy = POLICIES[policy_key]
    return (f"TEAMS = {dict(policy.teams)!r}\nURGENT = {list(policy.urgent)!r}\n"
            f"PREMIUM = {list(policy.premium_plans)!r}\nKEPT = {list(policy.kept_teams)!r}\n\n\n"
            "def route_ticket(api, topic: str, customer_id: str) -> dict:\n"
            "    if topic not in TEAMS:\n"
            "        return {\"ticket\": False, \"team\": None, \"urgent\": False}\n"
            "    team = TEAMS[topic]\n"
            f"    if team == {HELP!r}:\n"
            "        return {\"ticket\": False, \"team\": None, \"urgent\": False}\n"
            "    urgent = topic in URGENT\n"
            "    plan = api.read_customer(customer_id)[\"plan\"]\n"
            "    if plan in PREMIUM and not urgent and team not in KEPT:\n"
            "        team = \"Premium care\"\n"
            "    return {\"ticket\": True, \"team\": team, \"urgent\": urgent}\n")


def _families() -> dict[str, list[str]]:
    namespace: dict[str, Any] = {}
    exec((PROGRAMS / "route_message.py").read_text(encoding="utf-8").split("\n\n\ndef ")[0], namespace)  # constants
    return namespace["FAMILIES"]


FAMILIES = _families()
FAMILY_OF = {topic: family for family, topics in FAMILIES.items() for topic in topics}


def _confused(task: dict[str, Any]) -> str:
    """A topic of another team, the one the agent names when it gets a message wrong."""
    others = [t for t in TOPICS if POLICIES[task["policy"]].teams[t] != POLICIES[task["policy"]].teams[task["topic"]]]
    return others[len(task["id"]) % len(others)]


class InboxModels:
    """The agent and the builder. `confuses` is the share of messages the agent names a wrong topic for, `careless`
    the share of messages outside the bank it files a ticket for, `forgets` the share of premium customers whose rule
    it forgets without a routing function, and `stale` the share of functions the builder writes for the old policy."""

    live_calls = 0

    def __init__(self, confuses: float = 0.05, careless: float = 0.15, forgets: float = 0.2, stale: float = 0.15,
                 agent_seconds: float = 0.0) -> None:
        self.confuses, self.careless, self.forgets, self.stale = confuses, careless, forgets, stale
        self.agent_seconds = agent_seconds
        self.calls: Counter[str] = Counter()

    async def __call__(self, **kwargs: Any) -> Any:
        if "claude" in str(kwargs.get("model")):
            self.calls["builder"] += 1
            return self._builder(kwargs)
        self.calls["agent"] += 1
        if self.agent_seconds:
            await asyncio.sleep(self.agent_seconds)
        return self._agent(kwargs)

    def _agent(self, kwargs: dict[str, Any]) -> Any:
        messages = kwargs["messages"]
        task = TASKS[next(m["content"] for m in messages if m["role"] == "user")]
        tools = [tool["function"]["name"] for tool in kwargs.get("tools") or []]
        results = [json.loads(m["content"]) if m["content"].strip().startswith("{") else {"error": m["content"]}
                   for m in messages if m["role"] == "tool"]
        if any("error" in r for r in results) and "filed" not in str(results[-1]):
            return _reply("I could not reach the policy service, so nothing was filed.")
        if "file_ticket" not in tools or any("filed" in r for r in results):
            return _reply("Done.")
        if not task["topic"]:  # outside the bank's services
            if one_in(task["id"], self.careless, "careless"):
                return _reply(tool_calls=[_file("account_help", "Account and identity", False)])
            return _reply("This inbox handles the bank's own services, so nothing was filed.")
        topic = _confused(task) if one_in(task["id"], self.confuses, "confuses") else task["topic"]
        if "route_ticket" in tools:
            if not results:
                return _reply(tool_calls=[{"id": "call_route", "name": "route_ticket",
                                           "arguments": json.dumps({"topic": topic, "customer_id": task["customer"]})}])
            said = results[0]
            if not said.get("ticket"):
                return _reply("The help pages answer this, so no ticket was filed.")
            return _reply(tool_calls=[_file(topic, said.get("team"), said.get("urgent"))])
        if len(results) < 2:  # no function: it reads the policy and the record itself
            return _reply(tool_calls=[{"id": f"call_read_{len(results)}",
                                       "name": "read_policy" if not results else "read_customer",
                                       "arguments": json.dumps({} if not results
                                                               else {"customer_id": task["customer"]})}])
        plan = customer(task["customer"])["plan"]
        if one_in(task["id"], self.forgets, "forgets"):
            plan = "standard"
        routed = POLICIES[task["policy"]].route(topic, plan)
        if not routed["ticket"]:
            return _reply("The help pages answer this, so no ticket was filed.")
        return _reply(tool_calls=[_file(topic, routed["team"], routed["urgent"])])

    def _builder(self, kwargs: dict[str, Any]) -> Any:
        text = "\n".join(str(m.get("content")) for m in kwargs["messages"])
        if "You are improving a learning component" in text[:400]:
            self.calls["builder: rewrite"] += 1
            return _reply(hand_written_component(), prompt=30000, completion=3000)
        if "List the kinds of request these runs show" in text:
            self.calls["builder: kinds"] += 1
            return _reply(json.dumps(KINDS), prompt=20000, completion=400)
        self.calls["builder: function"] += 1
        late = one_in(f"route{self.calls['builder: function']}", self.stale, "stale")
        written = literal_function(NOW["before"] if late else NOW["policy"])
        return _reply("```python\n" + written + "\n```", prompt=8000, completion=900)

    @contextmanager
    def active(self) -> Iterator[InboxModels]:
        recording.install()
        token = recording._transport.set(self)
        try:
            yield self
        finally:
            recording._transport.reset(token)


def _file(topic: str, team: Any, urgent: Any) -> dict[str, Any]:
    return {"id": "call_file", "name": "file_ticket",
            "arguments": json.dumps({"topic": topic, "team": team, "urgent": bool(urgent)})}


class InboxJev:
    """The decision model, answering from each message's truth. `misread` is the share of messages outside the
    bank's services it takes for the bank's. `confuses` is the share of choices it gets wrong with low confidence, as
    a calibrated model does, and `confident` the share it gets wrong and is sure of."""

    name = "scripted"

    def __init__(self, misread: float = 0.02, confuses: float = 0.05, confident: float = 0.003) -> None:
        self.misread, self.confuses, self.confident = misread, confuses, confident
        self.calls = 0

    async def ask_with_usage(self, state: Any, questions: dict[str, Any]) -> tuple[dict[str, Answer], DecisionUsage]:
        self.calls += 1
        task = TASKS[state["request"]]
        topic = task["topic"]
        answers = {}
        for name, question in questions.items():
            kind = question["type"] if isinstance(question, dict) else type(question).__name__.lower()
            criteria = question.get("criteria") if isinstance(question, dict) else getattr(question, "criteria", {})
            if kind in ("noul", "probability"):
                about = bool(topic) or one_in(task["id"], self.misread, "misread")
                answers[name] = Answer("probability", 0.95 if about else 0.05)
            elif kind == "choice":
                options = list(criteria or {})
                wanted = topic if topic in options else FAMILY_OF.get(topic, "")
                sure = 0.9
                if wanted not in options or one_in(task["id"] + name, self.confuses, "jev-confuses"):
                    wanted, sure = next((o for o in options if o != wanted), options[0]), 0.4
                elif one_in(task["id"] + name, self.confident, "jev-confident"):
                    wanted = next((o for o in options if o != wanted), options[0])
                answers[name] = Answer("choice", wanted, sure)
            else:
                answers[name] = Answer("score", 0.5, 0.5)
        return answers, DecisionUsage(calls=1, input_tokens=300, dollars=300 * 42 / 1e9, seconds=0.15)

    def usage(self) -> DecisionUsage:
        return DecisionUsage(calls=self.calls)


__all__ = ["NOW", "InboxModels", "InboxJev", "literal_function", "TEAMS"]
