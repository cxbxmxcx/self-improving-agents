"""The support inbox's stream for the production-scale test: 4,000 real messages over 20 days, and what changes.

Every message is one a person wrote, with the topic the dataset's annotators gave it. Banking77's training split
supplies the stream and its test split what is held back and the final exam, so no message is both served and judged
elsewhere. Each message comes from a customer, and the customer's plan decides some routings.

Five things change, each of a kind a production agent meets.

    day 5    the policy moves disputed payments to a new Disputes team, which keeps them for premium customers too
    day 8    messages start arriving from people who wrote CLINC150's banking queries, in their own words
    day 11   the policy service does not answer for one message in ten
    day 14   the bank launches top-ups, and messages about them start arriving
    day 17   the policy narrows premium care to the metal plan

Once the policy has changed, a third of the messages that need a ticket are ones the change decides: disputed payments
after day 5, and premium customers' ordinary tickets after day 17.
"""

from __future__ import annotations

import csv
import json
import random
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from helix.metacognition import Gate

from examples.inbox.policy import DISPUTED, HELP, POLICIES, TOP_UPS, TOPICS, URGENT, Policy
from examples.inbox.world import INTENT, Inbox, customer

DATA = Path(__file__).parents[1] / "scale" / "runs" / "outside_data"
# CLINC150's banking intents that mean exactly one of Banking77's topics. Other banking intents are left out, since
# matching them to a topic would be our judgment and not the annotators'.
CLINC_TOPICS = {"report_lost_card": "lost_or_stolen_card", "pin_change": "change_pin", "exchange_rate": "exchange_rate",
                "card_declined": "declined_card_payment"}
TASKS: dict[str, dict[str, Any]] = {}  # every request by its text, for the scripted models of a soak


@dataclass(frozen=True)
class Day:
    number: int
    policy: Policy
    new_writers: bool = False  # CLINC150's banking queries arrive
    outage: bool = False  # the policy service does not answer for one message in ten
    top_ups: bool = False  # messages about top-ups arrive
    change: str = ""


def plan(days: int = 20) -> list[Day]:
    def at(day: int) -> int:
        return max(2, round(day * days / 20))

    disputes, writers, outage, top_ups, metal = at(5), at(8), at(11), at(14), at(17)
    changes = {disputes: "disputed payments move to a new Disputes team", writers: "CLINC150's writers arrive",
               outage: "the policy service is down for one message in ten", top_ups: "top-ups launch",
               metal: "premium care narrows to the metal plan"}
    first, second, third = POLICIES["2026-1"], POLICIES["2026-2"], POLICIES["2026-3"]
    return [Day(n, first if n < disputes else second if n < metal else third, new_writers=n >= writers,
                outage=n == outage, top_ups=n >= top_ups, change=changes.get(n, "")) for n in range(1, days + 1)]


def mix(count: int, new_writers: bool) -> list[str]:
    """Out of scope 10%; CLINC150's banking writers 10% once they arrive; the rest Banking77's own messages."""
    shares = {"b77": 0.80, "clinc": 0.10, "out": 0.10} if new_writers else {"b77": 0.90, "out": 0.10}
    exact = {kind: share * count for kind, share in shares.items()}
    whole = {kind: int(value) for kind, value in exact.items()}
    for kind in sorted(exact, key=lambda k: exact[k] - whole[k], reverse=True)[:count - sum(whole.values())]:
        whole[kind] += 1
    return [kind for kind, n in whole.items() for _ in range(n)]


# ---------------------------------------------------------------- the data


class Data:
    """Banking77 and CLINC150 as pools of (text, topic), split so served, held-back and exam messages never meet."""

    def __init__(self, root: Path = DATA) -> None:
        bank, clinc = root / "banking77", root / "clinc150"
        if not (bank / "train.csv").exists() or not (clinc / "data_full.json").exists():
            raise FileNotFoundError(f"the datasets are not in {root}; see examples/inbox/README.md for where they "
                                    "come from")
        self.train = self._b77(bank / "train.csv")
        self.test = self._b77(bank / "test.csv")
        full = json.loads((clinc / "data_full.json").read_text(encoding="utf-8"))
        self.clinc_served = [(t, CLINC_TOPICS[i]) for t, i in full["train"] + full["val"] if i in CLINC_TOPICS]
        self.clinc_held = [(t, CLINC_TOPICS[i]) for t, i in full["test"] if i in CLINC_TOPICS]
        # CLINC150 has 200 out-of-scope queries for training and 1,000 for testing; the stream needs more than 200.
        self.out_served = [(t, "") for t, _ in full["oos_train"] + full["oos_val"] + full["oos_test"][:600]]
        self.out_held = [(t, "") for t, _ in full["oos_test"][600:]]

    @staticmethod
    def _b77(path: Path) -> list[tuple[str, str]]:
        with path.open(encoding="utf-8") as handle:
            return [(row["text"].strip(), row["category"]) for row in csv.DictReader(handle)]


# ---------------------------------------------------------------- the stream


def _customer_on(plans: tuple[str, ...], seed: int) -> str:
    """A customer id whose plan is one of `plans`, found from `seed` onward."""
    n = seed
    while customer(f"C{n % 100000:05d}")["plan"] not in plans:
        n += 1
    return f"C{n % 100000:05d}"


def decided_by_change(policy: Policy, topic: str, plan_: str) -> bool:
    """Whether the version of the policy before `policy` routes this topic and customer differently."""
    before = {"2026-2": "2026-1", "2026-3": "2026-2"}.get(policy.key)
    return before is not None and POLICIES[before].route(topic, plan_) != policy.route(topic, plan_)


class Stream:
    """The messages of every day, what is held back from each, and the exam."""

    def __init__(self, per_day: int = 200, days: int = 20, *, data: Data | None = None, held: int = 16) -> None:
        self.per_day, self.days, self.held = per_day, plan(days), held
        self.data = data or Data()
        self.cases: dict[str, dict[str, Any]] = {}  # every request and function case by its id, for practice
        self._built: dict[tuple[str, int], list[dict[str, Any]]] = {}
        self._used: set[tuple[str, str]] = set()

    # ---- drawing messages

    def _draw(self, pool: list[tuple[str, str]], rng: random.Random, wanted: Any) -> tuple[str, str]:
        """A message from `pool` whose topic `wanted` accepts, not used before in this stream."""
        for _ in range(len(pool) * 2):
            text, topic = pool[rng.randrange(len(pool))]
            if (text, topic) not in self._used and wanted(topic):
                self._used.add((text, topic))
                return text, topic
        raise RuntimeError("the pool ran out of messages for this stream")

    def _request(self, kind: str, day: Day, rng: random.Random, reference: str, role: str,
                 sensitive: bool = False) -> dict[str, Any]:
        served = role in ("served", "start")  # what is served comes from the training split; the rest from the test
        if kind == "out":
            text, topic = self._draw(self.data.out_served if served else self.data.out_held, rng, lambda t: True)
        elif kind == "clinc":
            text, topic = self._draw(self.data.clinc_served if served else self.data.clinc_held, rng, lambda t: True)
        else:
            open_topics = set(TOPICS) if day.top_ups else set(TOPICS) - set(TOP_UPS)
            if sensitive and day.policy.key == "2026-2":
                open_topics &= set(DISPUTED)
            elif sensitive and day.policy.key == "2026-3":
                open_topics &= {t for t in TOPICS if day.policy.teams[t] not in (HELP, "Disputes") and t not in URGENT}
            text, topic = self._draw(self.data.train if served else self.data.test, rng, lambda t: t in open_topics)
        seed = zlib.crc32(reference.encode())
        if sensitive and day.policy.key == "2026-3":
            cid = _customer_on(("premium",), seed)
        else:
            cid = f"C{seed % 100000:05d}"
        plan_ = customer(cid)["plan"]
        needs_ticket = bool(topic) and day.policy.teams[topic] != HELP
        request = {"id": f"{reference}", "text": f"Message from customer {cid}: {text}", "message": text,
                   "topic": topic, "customer": cid, "plan": plan_, "policy": day.policy.key, "kind": kind,
                   "group": kind,
                   "expect": "act" if needs_ticket else "handoff",
                   "style": "out_of_scope" if kind == "out" else "help" if not needs_ticket else "plain",
                   "sensitive": bool(topic) and decided_by_change(day.policy, topic, plan_),
                   "outage": False, "asked": "out" if kind == "out" else "ticket" if needs_ticket else "help"}
        TASKS[request["text"]] = request
        self.cases[request["id"]] = request
        return request

    def _set(self, role: str, day: Day, count: int) -> list[dict[str, Any]]:
        rng = random.Random(f"{role}-{day.number}")
        kinds = mix(count, day.new_writers)
        rng.shuffle(kinds)
        out, ticketed = [], 0
        for n, kind in enumerate(kinds):
            sensitive = kind == "b77" and day.policy.key != "2026-1" and ticketed % 3 == 0
            ticketed += kind == "b77"
            request = self._request(kind, day, rng, f"{role}{day.number}-{n}", role, sensitive)
            if role == "served" and day.outage:
                request["outage"] = zlib.crc32(request["id"].encode()) % 10 == 0
            out.append(request)
        return out

    def _word(self) -> None:
        """Draw every message once, in one fixed order, so no set depends on which is asked for first."""
        if self._built:
            return
        # The day before the stream, when the assistant's routing function is first learned, and the messages the
        # search repeats are measured on, under the same first policy.
        self._built["start", 0] = self._set("start", self.days[0], 40)
        self._built["held_out", 0] = self._set("held_out", self.days[0], 52)
        for day in self.days:
            self._built["served", day.number] = self._set("served", day, self.per_day)
            for n, request in enumerate(self._built["served", day.number]):
                request.update(day=day.number, seq=(day.number - 1) * self.per_day + n)
            for role in ("gate", "search"):
                self._built[role, day.number] = self._set(role, day, self.held)
        last = self.days[-1]
        self._built["exam", 0] = self._set("exam", last, self.per_day)
        for n, request in enumerate(self._built["exam", 0]):
            request.update(day=len(self.days) + 1, seq=len(self.days) * self.per_day + n)

    def served(self, day: Day) -> list[dict[str, Any]]:
        self._word()
        return self._built["served", day.number]

    def held_back(self, day: Day) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
        self._word()
        return ({r["id"]: r for r in self._built["gate", day.number]},
                {r["id"]: r for r in self._built["search", day.number]})

    def exam(self) -> list[dict[str, Any]]:
        self._word()
        return self._built["exam", 0]

    def held_out(self) -> list[dict[str, Any]]:
        """Fifty-two messages under the first policy, never served in the stream: what the search repeats measure."""
        self._word()
        return self._built["held_out", 0]

    def start(self) -> list[dict[str, Any]]:
        """Forty messages under the first policy, served before the stream so the routing function is learned once
        and every assistant starts with the same one."""
        self._word()
        return self._built["start", 0]

    # ---- the function's own cases

    def function_gate(self, day: Day) -> Gate:
        """Twelve cases that judge a relearned route_ticket under the policy in force, half of them decided by its
        last change, and every one needing a ticket or the help pages."""
        rng = random.Random(f"function-{day.policy.key}")
        ids = []
        for n in range(12):
            sensitive = day.policy.key != "2026-1" and n % 2 == 0
            open_topics = [t for t in TOPICS if (day.top_ups or t not in TOP_UPS)]
            if sensitive and day.policy.key == "2026-2":
                open_topics = list(DISPUTED)
            elif sensitive and day.policy.key == "2026-3":
                open_topics = [t for t in open_topics if day.policy.teams[t] not in (HELP, "Disputes")
                               and t not in URGENT]
            topic = open_topics[rng.randrange(len(open_topics))]
            cid = _customer_on(("premium",) if sensitive else ("standard", "premium", "metal"), rng.randrange(99999))
            case_id = f"F{day.policy.key}-{n}"
            self.cases[case_id] = {"id": case_id, "topic": topic, "customer": cid, "policy": day.policy.key,
                                   "expect": "act" if day.policy.teams[topic] != HELP else "handoff"}
            ids.append(case_id)
        return Gate(cases={"route_ticket": ids})

    async def practice(self, source: str, case_id: str) -> tuple[float, str, str]:
        """Try a route_ticket source on the case behind a request or a function case."""
        from examples.inbox.world import check_function_async

        case = self.cases[case_id.split("~")[0]]
        if not case.get("topic"):
            return 1.0, "a message outside the bank's services needs no routing", ""
        right, reason = await check_function_async(source, case["topic"], case["customer"], case["policy"])
        return (1.0 if right else 0.0), reason, f"topic {case['topic']}, customer {customer(case['customer'])}"


def verify(ctx: Any, result: Any) -> Any:
    """The verdict on one run. A message outside the bank's services is judged on intent alone: it says nothing about
    the routing function."""
    from helix.metacognition import Verdict

    world: Inbox = ctx.context
    verdict = world.check()
    source = INTENT if world.task.get("style") == "out_of_scope" else "verifier"
    return Verdict(bool(verdict["success"]), verdict["reason"], source)
