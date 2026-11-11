"""The support inbox's assistant, its loop and its decision tier, set up as the calendar desk's are.

Everything here is a setting of helix: the agent and its tools, Chapter 6's loop around the routing function, and a
decision tier that a code search can fill. The hand-written component in `programs/` is the reference a searched one
is measured against, and what a scripted builder returns in a soak.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from helix.memory.code_skills import skill_tool
from helix.memory.learning_mechanisms import Contract
from helix.memory.mechanism_runtime import pack_memories
from helix.metacognition import Agent, Controller, LearningBudget, Metacognition, MetaStore, ValueLearning
from helix.metacognition.decide import KIT_GUIDE, RETURNS_ROUTED, Desk
from helix.metacognition.decisions import ProgramSearch, ProgramTier
from helix.metacognition.model_settings import BUILDER_MODEL
from helix.metacognition.pricing import PRICES

from examples.inbox.policy import TOPICS
from examples.inbox.stream import Stream, verify
from examples.inbox.world import INTENT, REFERENCE, ROUTE_TICKET, Inbox, _Api, file_ticket, read_customer, read_policy

HERE = Path(__file__).parent
PROGRAMS = HERE / "programs"
SEEDS = Path(__import__("helix").__file__).parent / "metacognition" / "programs"
MODEL = "gpt-5.4-mini"
PROMPT = """You are the triage assistant of a bank's support inbox. Each message names the customer who sent it.
Work out which topic in the routing policy's table the message is about. File one ticket for the team the policy
names, urgent if the policy says so, or file nothing when the policy says the help pages answer that topic or when
the message is not about the bank's own services. If you have route_ticket, call it with the topic and the customer
id and file what it says. File at most one ticket per message, and reply in a sentence or two."""
JOB = ("Serve one message to a bank's support inbox. Its job is filing one ticket for the team that handles the "
       "message's topic, urgent or not, as the routing policy in force says. A message whose topic the help pages "
       "answer gets no ticket, and a message that is not about the bank's own services gets a short reply and no "
       "ticket: both go to the agent read-only. The customer id is in the message, as C followed by five digits. "
       "Read tools: read_policy(topic) answers one topic's team, whether it is urgent, whether the help pages answer "
       "it, and the policy's rules, or the whole policy when topic is empty; read_customer(customer_id) answers the "
       "sender's record. Procedure: route_ticket(topic, customer_id), answering {\"ticket\": bool, \"team\": str or "
       "None, \"urgent\": bool} under the policy in force. Action: file_ticket, with the arguments {\"topic\": str, "
       "\"team\": str, \"urgent\": bool}. The topics are these 77: " + ", ".join(TOPICS) + ".")


def assistant(**kwargs: Any) -> Agent:
    return Agent(name="Bank support triage", instructions=PROMPT, model=MODEL,
                 tools=[read_policy, read_customer, file_ticket], **kwargs)


def desk() -> Desk:
    """The inbox for a decision program on its own. Inside the tier the loop calls the trusted route_ticket."""
    def procedures(world: Inbox) -> dict[str, Any]:
        async def route_ticket(**arguments: Any) -> Any:
            api = _Api(world.task["topic"], world.task["customer"], world.task["policy"])
            return await skill_tool("route_ticket", "", REFERENCE, bind={"api": api}).call(**arguments)

        return {"route_ticket": route_ticket}

    def act(world: Inbox, topic: str, team: str, urgent: bool = False) -> dict:
        return world.file_ticket(topic, team, urgent)

    return Desk(world=Inbox, reads=lambda world: {"read_policy": world.read_policy,
                                                  "read_customer": world.read_customer},
                procedures=procedures, actions={"file_ticket": act}, verdict=lambda world, reply: world.check())


def contract() -> Contract:
    return Contract(name="decide", signature="async def decide(request, kit) -> dict", description=JOB, api=KIT_GUIDE,
                    returns=RETURNS_ROUTED)


def memories() -> list[dict[str, Any]]:
    return [
        {"name": "route_message", "description": "Messages about the bank's own services, to be filed or answered "
                                                 "from the help pages.",
         "body": (PROGRAMS / "route_message.py").read_text(encoding="utf-8"), "route": "program", "read_only": False},
        {"name": "not_for_the_bank", "description": "Messages that are not about the bank's own services.",
         "body": "This message is not about the bank's services. Reply briefly that the inbox handles the bank's "
                 "cards, payments, transfers, top-ups, currency and accounts, and file nothing.",
         "route": "none", "read_only": True},
    ]


def hand_written_component() -> str:
    """The reference component: `learn` stores the memories and returns the router."""
    router = (PROGRAMS / "router.py").read_text(encoding="utf-8")
    stored = [(m["name"], m["description"], m["body"], m["route"], m["read_only"]) for m in memories()]
    return (f"ROUTER = {router!r}\nSTORED = {stored!r}\n\n\nasync def learn(lab):\n"
            "    for name, description, body, route, read_only in STORED:\n"
            "        lab.remember(name, description, body, route=route, read_only=read_only)\n"
            "    return ROUTER\n")


def hand_written_program() -> str:
    return pack_memories((PROGRAMS / "router.py").read_text(encoding="utf-8"), memories())


def seed(name: str = "component_llm") -> tuple[str, str]:
    return name, (SEEDS / f"{name}.py").read_text(encoding="utf-8")


def code_search(archive: Path | str = ":memory:", **changes: Any) -> ProgramSearch:
    settings: dict[str, Any] = {"target": "mechanism", "population": 6, "generations": 1,
                                "designer_model": BUILDER_MODEL, "penalty": 2.0, "stop_when_admitted": True, "seed": 7,
                                "label": "inbox, dgm"}
    return ProgramSearch(method="dgm", archive=str(archive), **{**settings, **changes})


def decision_tier(jev: Any, stream: Stream, day: Any, *, search: ProgramSearch | None = None,
                  **settings: Any) -> ProgramTier:
    gate, seen = stream.held_back(day)
    return ProgramTier(desk=desk(), decider=jev, contract=contract(), mechanisms=[seed()], search=search,
                       labeled=gate, search_cases=seen, world=Inbox, bar=0.9, max_wrong=0, routed_bar=0.7,
                       read_only_handoffs=True, writer_model=BUILDER_MODEL, gating_sources=("verifier", INTENT),
                       **settings)


def harness(stream: Stream, tier: ProgramTier | None = None, *, function: str | None = None,
            **changes: Any) -> tuple[Agent, Metacognition]:
    """The assistant with its loop. `function` is the route_ticket it starts with, trusted; None starts untrusted."""
    settings: dict[str, Any] = {
        "verifier": verify, "controller": Controller({"memory": "none", "no_memory": "high"}), "recall_k": 1,
        "learner": ValueLearning(), "writer_model": BUILDER_MODEL, "practice": stream.practice,
        "gate": stream.function_gate(stream.days[0]), "budget": LearningBudget(attempts_per_day=3),
        "relearn_after": 4, "learning": "inline", "prices": PRICES, "programs": tier,
        "gating_sources": ("verifier",), "store": MetaStore(keep=2000)}
    meta = Metacognition([ROUTE_TICKET], **{**settings, **changes})
    if function is not None:
        meta.install("route_ticket", function, created_by="learned")
    if tier is not None:
        meta.programs.paused = True
    meta.agent = assistant(metacognition=meta)
    return meta.agent, meta
