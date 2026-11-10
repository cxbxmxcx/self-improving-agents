"""Varied wordings of D1's and D3's requests, where a decision program has to understand the request.

D1's and D3's own requests are templated: one regular expression finds every id. This split asks the same work
in other words, once, with a model (`python -m examples.decisions.wordings --build`), and keeps the result in
wordings.json so nothing needs to be generated again:

    casual       an informal message, with the id written in full
    shorthand    the id as its number only ("invoice 201")
    distractor   a second id that must be left alone
    buried       the request in the middle of a longer note

Out-of-scope requests are written by hand. They name real ids and sound like the work ("cancel the payment on
INV-201") but ask for something the desk can't do, so the right move is to hand them off. Development and test
requests use different out-of-scope templates, and different base requests, so the test split is unseen.
D3's vendor questions keep their own wording: they need a written answer, so a program should hand them off.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
from pathlib import Path
from typing import Any

from metacog import Recording

HERE = Path(__file__).parent
FILE = HERE / "wordings.json"
STYLES = ("casual", "shorthand", "distractor", "buried")
MODEL = "gpt-5.4-mini"

OUT_OF_SCOPE: dict[str, dict[str, list[str]]] = {
    "d1": {
        "dev": ["What's the status of my claim for trip {trip}?", "Please cancel the expense claim I filed for trip {trip}.",
                "How much is the daily meal allowance for trip {trip}'s city?", "Can you split trip {trip} into two claims?"],
        "test": ["Did finance already pay out trip {trip}?", "Withdraw my claim on {trip}, I filed it by mistake.",
                 "What exchange rate does the company use for the receipts on {trip}?",
                 "Please book me a flight for trip {trip}."],
    },
    "d3": {
        "dev": ["What's the status of invoice {invoice}?", "Cancel the payment scheduled for invoice {invoice}.",
                "Please dispute invoice {invoice} with the vendor.", "Delete document {document} from the ledger."],
        "test": ["Has invoice {invoice} been paid yet?", "Put invoice {invoice} on hold until further notice.",
                 "Send the vendor a reminder about invoice {invoice}.", "Reject document {document}, it is a duplicate."],
    },
}

# A harder test of knowing what's being asked: requests in the work's own words that ask for something else.
# None of a learner's runs include these, and each should be handed off.
NEAR_MISS: dict[str, list[str]] = {
    "d1": ["Did you already file my expense claim for trip {trip}?",
           "Don't file my expense claim for {trip} yet, I still have receipts to add.",
           "Can you check whether the expense claim for {trip} went through?",
           "Undo the claim you filed for {trip}.",
           "Please amend the expense claim on {trip} to add a taxi receipt.",
           "How much would the claim for {trip} come to? Don't file anything.",
           "My manager wants a copy of the expense claim for {trip}.",
           "Who approves the expense claim for {trip}?",
           "I filed the claim for {trip} myself yesterday, so nothing is needed.",
           "Remind me on Friday to file the claim for {trip}."],
    "d3": ["Did we already schedule the payment for invoice {invoice}?",
           "Don't pay invoice {invoice} until I confirm with the vendor.",
           "What do we owe on invoice {invoice}? Just tell me, don't schedule anything.",
           "Move the scheduled payment for {invoice} to next month.",
           "Was the payment for invoice {invoice} late?",
           "Who approved the payment of {invoice}?",
           "Is document {document} already entered in the ledger?",
           "Don't enter document {document} in the ledger, it's a copy.",
           "Take document {document} back out of the ledger.",
           "Which ledger account does document {document} go to?"],
}

# Near misses a learner may practice on, in other words than either held-out set.
NEAR_MISS_DEV: dict[str, list[str]] = {
    "d1": ["Has my expense claim for trip {trip} been filed?", "Hold off on filing the claim for {trip} until Monday.",
           "Cancel whatever you filed on {trip}.", "Change the claim on {trip}, the hotel amount is wrong.",
           "Estimate the claim for {trip}, but leave it unfiled.", "Email me the claim you filed for {trip}."],
    "d3": ["Has the payment for invoice {invoice} been scheduled?", "Hold the payment on invoice {invoice} for now.",
           "Tell me the amount due on invoice {invoice}, nothing else.",
           "Reschedule the payment for {invoice} to the 30th.", "Has document {document} been entered yet?",
           "Leave document {document} out of the ledger."],
}
NEAR_MISS_FRESH: dict[str, list[str]] = {
    "d1": ["Is the claim for {trip} still pending with finance?",
           "Skip filing a claim for {trip}, the client is paying directly.", "Scrap the expense claim from {trip}.",
           "Add the parking receipt to my claim for {trip}.",
           "Roughly what will I get back for {trip}? No need to submit it.",
           "Forward the claim for {trip} to my manager.", "When will the claim for {trip} be reimbursed?",
           "Whoever files the claim for {trip} should ask me first.",
           "We're good on {trip}, the claim was handled last week.",
           "Next month, please file the claim for {trip}; not today."],
    "d3": ["Is invoice {invoice} on the schedule already?", "Please stop the payment on invoice {invoice}.",
           "How much is due on {invoice}? Leave it unscheduled.", "Push {invoice}'s payment date back two weeks.",
           "Did invoice {invoice} get paid on time?", "Who signed off on paying {invoice}?",
           "Has anyone keyed document {document} into the ledger?",
           "Skip entering document {document}, we already have it.", "Remove document {document} from the books.",
           "What account code should document {document} post to?"],
}

# The gate's own near misses: the same kinds as the others, in words no search or learner sees.
NEAR_MISS_GATE: dict[str, list[str]] = {
    "d1": ["Has finance received the claim for {trip} yet?",
           "Please wait before you file anything for {trip}; I'm still checking the receipts.",
           "Did the claim for {trip} get approved?", "Withdraw the expense claim for {trip}.",
           "The claim on {trip} needs the airport shuttle added.",
           "Give me a ballpark for the {trip} claim, but hold the filing.",
           "Send a copy of the {trip} claim over to accounting.", "Which person signs off on the claim for {trip}?",
           "No need to touch {trip}, my assistant already filed that claim.",
           "File the claim for {trip} once I'm back from leave, not now."],
    "d3": ["Can you confirm whether {invoice} has been queued for payment?",
           "Keep invoice {invoice} unpaid until legal has reviewed it.",
           "I only need the payable figure for {invoice}, please don't set anything up.",
           "Bring the payment date on {invoice} forward by a week.", "Did {invoice} go out before its due date?",
           "Which of our approvers cleared {invoice} for payment?",
           "Was document {document} ever booked into the ledger?",
           "Document {document} is a duplicate, so it shouldn't go into the ledger.",
           "Reverse the ledger entry made from document {document}.",
           "What cost centre would document {document} fall under?"],
}

PROMPT = """Someone at a company sent their assistant this request:

{text}

Rewrite it four ways and reply with only a JSON object with the keys casual, shorthand, distractor and buried.
Each must ask for the same job on the same {noun}, and add no other instructions.
- casual: an informal chat message that contains {target} exactly as written.
- shorthand: refers to it only by its number, such as "{noun} {digits}". It must not contain "{target}".
- distractor: contains both {target} and {other}, and makes clear that only {target} is to be acted on and that
  {other} is a different one that needs nothing. Mention either one first, in natural words.
- buried: three or four sentences of ordinary workplace context, with the request and {target} in the middle.{again}"""


def _tasks(domain: str):
    if domain == "d1":
        from examples.expenses.world import task_set
    else:
        from examples.invoices.world import task_set
    return task_set


def _ids(domain: str, task: dict[str, Any]) -> tuple[str, str, str, str] | None:
    """The request's target id, its number, a different id of the same form, and its noun."""
    pattern, noun = ((r"T-\d+", "trip") if domain == "d1" else (r"INV-\d+", "invoice") if task["kind"] == "payment"
                     else (r"DOC-\d+", "document") if task["kind"] == "intake" else (None, None))
    if pattern is None:
        return None
    target = re.search(pattern, task["text"]).group(0)
    prefix, digits = target.split("-")
    return target, digits, f"{prefix}-{int(digits) + 40}", noun


def _valid(reply: dict[str, Any], target: str, digits: str, other: str) -> bool:
    if not all(isinstance(reply.get(style), str) and reply[style].strip() for style in STYLES):
        return False
    return (target in reply["casual"] and target in reply["buried"] and target in reply["distractor"]
            and other in reply["distractor"] and target not in reply["shorthand"] and digits in reply["shorthand"])


async def _rewrite(domain: str, task: dict[str, Any]) -> dict[str, str]:
    from helix import llm_call

    target, digits, other, noun = _ids(domain, task)
    for attempt in range(4):
        again = "" if attempt == 0 else f"\n\nAttempt {attempt + 1}: follow every rule about the ids exactly."
        prompt = PROMPT.format(text=task["text"], noun=noun, target=target, digits=digits, other=other, again=again)
        response = await llm_call.acompletion(model=MODEL, messages=[{"role": "user", "content": prompt}],
                                              reasoning_effort="low")
        text = (response.choices[0].message.content or "").strip()
        text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        try:
            reply = json.loads(text)
        except json.JSONDecodeError:
            continue
        if _valid(reply, target, digits, other):
            return {style: reply[style].strip() for style in STYLES}
    raise ValueError(f"no valid rewrite of {task['id']}")


def _out_of_scope(domain: str, split: str, tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    entries = []
    if domain == "d1":
        bases = tasks[:2]
    else:
        bases = [next(t for t in tasks if t["kind"] == "payment"), next(t for t in tasks if t["kind"] == "intake")]
        payment, intake = bases
    for n, template in enumerate(OUT_OF_SCOPE[domain][split]):
        if domain == "d1":
            base = bases[n % 2]
            text = template.format(trip=re.search(r"T-\d+", base["text"]).group(0))
        else:
            base = intake if "{document}" in template else payment
            text = template.format(invoice=re.search(r"INV-\d+", payment["text"]).group(0),
                                   document=re.search(r"DOC-\d+", intake["text"]).group(0))
        entries.append({"id": f"{base['id']}~out{n}", "base": base["id"], "domain": domain, "split": split,
                        "style": "out_of_scope", "text": text, "expect": "handoff"})
    return entries


async def build(args: argparse.Namespace) -> None:
    from dotenv import load_dotenv

    load_dotenv(Path.cwd() / ".env")
    recording = Recording(HERE / "recorded" / "wordings.jsonl", mode="record")
    entries: list[dict[str, Any]] = []
    gate = asyncio.Semaphore(8)

    async def one(domain: str, split: str, task: dict[str, Any]) -> list[dict[str, Any]]:
        if _ids(domain, task) is None:  # a vendor question keeps its wording and goes to the agent
            return [{"id": task["id"], "base": task["id"], "domain": domain, "split": split, "style": "question",
                     "text": task["text"], "expect": "handoff"}]
        async with gate:
            rewrites = await _rewrite(domain, task)
        return [{"id": f"{task['id']}~{style}", "base": task["id"], "domain": domain, "split": split, "style": style,
                 "text": text, "expect": "act"} for style, text in rewrites.items()]

    with recording.active():
        for domain in ("d1", "d3"):
            task_set = _tasks(domain)
            for split in ("dev", "test"):
                tasks = task_set(split)[:16]
                for found in await asyncio.gather(*(one(domain, split, t) for t in tasks)):
                    entries.extend(found)
                entries.extend(_out_of_scope(domain, split, tasks))
    FILE.write_text(json.dumps(entries, indent=1), encoding="utf-8", newline="\n")
    print(f"{len(entries)} requests in {FILE.name}; {recording.live_calls} live calls, ${recording.spent():.4f}")


def _near_miss(domain: str, split: str, templates: list[str], tag: str) -> list[dict[str, Any]]:
    """Each near-miss template on two of a split's requests' worlds."""
    tasks = _tasks(domain)(split)[:16]
    out = []
    if domain == "d1":
        for base in tasks[:2]:
            trip = re.search(r"T-\d+", base["text"]).group(0)
            out += [{**base, "id": f"{base['id']}~{tag}{n}", "text": t.format(trip=trip), "style": "near_miss",
                     "expect": "handoff"} for n, t in enumerate(templates)]
        return out
    payments = [t for t in tasks if t["kind"] == "payment"][:2]
    intakes = [t for t in tasks if t["kind"] == "intake"][:2]
    for payment, intake in zip(payments, intakes, strict=True):
        for n, template in enumerate(templates):
            base = intake if "{document}" in template else payment
            text = template.format(invoice=re.search(r"INV-\d+", payment["text"]).group(0),
                                   document=re.search(r"DOC-\d+", intake["text"]).group(0))
            out.append({**base, "id": f"{base['id']}~{tag}{n}", "text": text, "style": "near_miss", "expect": "handoff"})
    return out


def load(domain: str, split: str) -> list[dict[str, Any]]:
    """The split's requests as tasks: each is its base task's world with the new words and what's expected.

    Besides dev and test: `dev+near` adds near misses for a learner to practice on, `stress` is the first
    held-out near-miss set, `stress2` a second, written before any program learned with near misses existed, and
    `gate` a third, for a gate that must not share wordings with what a search sees.
    """
    if domain in ("d5", "d6", "d7"):
        from importlib import import_module

        return import_module(f"examples.{ {'d5': 'shifts', 'd6': 'billing', 'd7': 'triage'}[domain] }.world").load(split)
    if split == "gate":
        return _near_miss(domain, "test", NEAR_MISS_GATE[domain], "gate")
    if split == "stress":
        return _near_miss(domain, "test", NEAR_MISS[domain], "near")
    if split == "stress2":
        return _near_miss(domain, "test", NEAR_MISS_FRESH[domain], "fresh")
    if split == "dev+near":
        return load(domain, "dev") + _near_miss(domain, "dev", NEAR_MISS_DEV[domain], "neardev")
    task_set = _tasks(domain)
    bases = {t["id"]: t for t in task_set(split)}
    out = []
    for entry in json.loads(FILE.read_text(encoding="utf-8")):
        if entry["domain"] == domain and entry["split"] == split:
            out.append({**bases[entry["base"]], "id": entry["id"], "text": entry["text"], "style": entry["style"],
                        "expect": entry["expect"]})
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--build", action="store_true", help="generate wordings.json with live model calls")
    options = parser.parse_args()
    if options.build:
        asyncio.run(build(options))
