"""One message in the bank's support inbox: the customer who sent it, the policy in force, the tools, and the check.

The desk's tools are what a triage assistant has: the routing policy (`read_policy`), the customer's record
(`read_customer`) and the one action (`file_ticket`). The function the loop learns, `route_ticket`, works out from the
policy and the record what a ticket for a topic must say. A message is right when the ticket filed, or the absence of
one, is what the policy in force asks for: the team that handles it and whether it is urgent. The topic the assistant
names on the ticket is not judged, since two topics can share a team.
"""

from __future__ import annotations

import zlib
from typing import Any

from helix.memory.learning_mechanisms import Contract
from helix.metacognition import Procedure, RunContextWrapper, function_tool
from helix.metacognition.exceptions import Unavailable

from examples.inbox.policy import HELP, POLICIES, TOPICS, Policy

PLANS = ("standard", "premium", "metal")
INTENT = "intent"  # who judges a message that needed no ticket: it says nothing about the routing function


def customer(customer_id: str) -> dict[str, Any]:
    """The customer's record. The same id always gives the same record: a quarter premium, a sixth metal."""
    roll = zlib.crc32(customer_id.encode()) % 1000
    plan = "metal" if roll < 160 else "premium" if roll < 410 else "standard"
    return {"customer_id": customer_id, "plan": plan, "since": 2015 + roll % 11,
            "country": ("GB", "IE", "DE", "FR", "ES", "NL")[roll % 6]}


class Inbox:
    """The world of one message. `task` holds the message, the customer, the policy key and the label."""

    def __init__(self, task: dict[str, Any]) -> None:
        self.task = task
        self.policy: Policy = POLICIES[task["policy"]]
        self.down = bool(task.get("outage"))
        self.style = task.get("style", "plain")
        self.tickets: list[dict[str, Any]] = []

    # ---- tools

    def read_policy(self, topic: str = "") -> dict[str, Any]:
        if self.down:
            raise Unavailable("the policy service did not answer")
        if not topic:
            return {"policy": self.policy.key, "text": self.policy.text}
        if topic not in self.policy.teams:
            return {"error": f"no topic {topic!r}; read the policy with no topic for the table"}
        team = self.policy.teams[topic]
        return {"policy": self.policy.key, "topic": topic, "team": team, "help_pages": team == HELP,
                "urgent": topic in self.policy.urgent, "rules": self.policy.text.split("\n\n")[0]}

    def read_customer(self, customer_id: str = "") -> dict[str, Any]:
        wanted = customer_id or self.task["customer"]
        if wanted != self.task["customer"]:
            return {"error": f"no access to customer {wanted}: only the sender's record can be read"}
        return customer(wanted)

    def file_ticket(self, topic: str, team: str, urgent: bool = False) -> dict[str, Any]:
        ticket = {"topic": topic, "team": team, "urgent": bool(urgent)}
        self.tickets.append(ticket)
        return {"filed": f"T-{len(self.tickets)}", **ticket}

    # ---- the verdict

    def expected(self) -> dict[str, Any]:
        if self.task["expect"] != "act":  # "act": a ticket is needed; "handoff": none is
            return {"ticket": False, "team": None, "urgent": False}
        return self.policy.route(self.task["topic"], customer(self.task["customer"])["plan"])

    def check(self) -> dict[str, Any]:
        want = self.expected()
        if not want["ticket"]:
            if self.tickets:
                return {"success": False, "reason": f"filed a ticket for {self.tickets[0]['team']}, and this "
                                                    "message needed none"}
            return {"success": True, "reason": "no ticket, as the policy asks"}
        if len(self.tickets) != 1:
            return {"success": False, "reason": f"{len(self.tickets)} tickets filed, and one was needed"}
        got = self.tickets[0]
        if got["team"] != want["team"]:
            return {"success": False, "reason": f"the ticket went to {got['team']}, and the policy sends it to "
                                                f"{want['team']}"}
        if got["urgent"] != want["urgent"]:
            return {"success": False, "reason": f"the ticket was filed {'urgent' if got['urgent'] else 'not urgent'},"
                                                f" and the policy says {'urgent' if want['urgent'] else 'not urgent'}"}
        return {"success": True, "reason": "filed as the policy asks"}

    @property
    def booked(self) -> list[dict[str, Any]]:  # what the drivers count as an action taken
        return self.tickets


# ---------------------------------------------------------------- the agent's tools


@function_tool(read_only=True)
def read_policy(ctx: RunContextWrapper[Inbox], topic: str = "") -> dict:
    """The routing policy in force. With no topic: its rules and the table of topics and teams. With a topic: that
    topic's team, whether it is urgent, and whether the help pages answer it."""
    return ctx.context.read_policy(topic)


@function_tool(read_only=True)
def read_customer(ctx: RunContextWrapper[Inbox], customer_id: str = "") -> dict:
    """The record of the customer who sent the message: plan, joining year and country."""
    return ctx.context.read_customer(customer_id)


@function_tool
def file_ticket(ctx: RunContextWrapper[Inbox], topic: str, team: str, urgent: bool = False) -> dict:
    """File one ticket for the team that handles the message's topic, urgent or not."""
    return ctx.context.file_ticket(topic, team, urgent)


# ---------------------------------------------------------------- the procedure the loop learns

SIGNATURE = "def route_ticket(api, topic: str, customer_id: str) -> dict"
API = ("`api.read_policy(topic)` returns what the read_policy tool returns for that topic, and `api.read_policy()` "
       "the whole policy; `api.read_customer(customer_id)` returns the customer's record.")
CONTRACT = Contract(
    name="route_ticket", signature=SIGNATURE,
    description=("Work out what the ticket for a message about `topic` from customer `customer_id` must say under the "
                 "routing policy in force: whether there is a ticket at all, the team it goes to, and whether it is "
                 "urgent."),
    api=API,
    returns='Return {"ticket": bool, "team": str or None, "urgent": bool}. With no ticket, team is None.')
ROUTE_TICKET = Procedure.from_contract(CONTRACT, api=[read_policy, read_customer],
                                       description=CONTRACT.description)

REFERENCE = '''def route_ticket(api, topic: str, customer_id: str) -> dict:
    row = api.read_policy(topic)
    if "error" in row:
        return {"ticket": False, "team": None, "urgent": False, "error": row["error"]}
    if row["help_pages"]:
        return {"ticket": False, "team": None, "urgent": False}
    plan = api.read_customer(customer_id)["plan"]
    rules = row["rules"]
    premium = []
    for word in ("standard", "premium", "metal"):
        if word in rules.split("go to Premium care")[0].split("Tickets from customers on the")[-1]:
            premium.append(word)
    kept = []
    if "stay with that team for every customer" in rules:
        kept = [part.strip() for part in rules.split("Tickets for ")[-1].split(" stay with")[0].split(",")]
    team = row["team"]
    if plan in premium and not row["urgent"] and team not in kept:
        team = "Premium care"
    return {"ticket": True, "team": team, "urgent": row["urgent"]}
'''


class _Api:
    """What a route_ticket function may call: the two read tools, on the world of one case."""

    def __init__(self, topic: str, customer_id: str, policy_key: str) -> None:
        self.world = Inbox({"policy": policy_key, "customer": customer_id, "topic": topic, "expect": "act"})

    def read_policy(self, topic: str = "") -> dict[str, Any]:
        return self.world.read_policy(topic)

    def read_customer(self, customer_id: str = "") -> dict[str, Any]:
        return self.world.read_customer(customer_id)


async def check_function_async(source: str, topic: str, customer_id: str, policy_key: str) -> tuple[bool, str]:
    """Run a route_ticket source on one case, in process under the guard, and say whether it is right."""
    from helix.memory.code_skills import skill_tool

    want = POLICIES[policy_key].route(topic, customer(customer_id)["plan"])
    try:
        tool = skill_tool("route_ticket", "", source, bind={"api": _Api(topic, customer_id, policy_key)})
        result = await tool.call(topic=topic, customer_id=customer_id)
    except Exception as exc:  # noqa: BLE001 - a function that cannot be called is wrong
        return False, f"the function could not be called: {type(exc).__name__}: {exc}"
    if not isinstance(result, dict) or "ticket" not in result:
        return False, f"the function returned {result!r}, not a dict with ticket, team and urgent"
    same = (bool(result.get("ticket")) == want["ticket"] and result.get("team") == want["team"]
            and bool(result.get("urgent")) == want["urgent"])
    return same, "right" if same else f"it said {result!r}, and the policy says {want!r}"


def check_function(source: str, topic: str, customer_id: str, policy_key: str) -> tuple[bool, str]:
    import asyncio

    return asyncio.run(check_function_async(source, topic, customer_id, policy_key))


def all_topics() -> tuple[str, ...]:
    return TOPICS
