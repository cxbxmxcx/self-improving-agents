"""Write requests.md, a readable view of requests.json; rerun it after editing the requests.

    python chapter_appendices/ch06_decision_scaffold_legacy/assistant/requests_view.py
"""

import json
from pathlib import Path

HERE = Path(__file__).parent
world = json.loads((HERE / "world.json").read_text(encoding="utf-8"))
requests = json.loads((HERE / "requests.json").read_text(encoding="utf-8"))
people = {k: v["name"] for k, v in world["people"].items()}
events = {e["id"]: e for e in world["events"]}
messages = {m["id"]: m for m in world["messages"]}
DAY = {"mon": "Mon", "tue": "Tue", "wed": "Wed", "thu": "Thu", "fri": "Fri"}


def needs(groups):
    return "; ".join(" or ".join(f'"{w}"' for w in group) for group in groups)


def truth_text(truth):
    kind = truth["kind"]
    if kind == "booking":
        parts = [f"{truth.get('count', 1)} x {truth['minutes']} min"]
        if truth.get("days"):
            parts.append("on " + "/".join(DAY[d] for d in truth["days"]))
        if truth.get("start"):
            parts.append(f"at {truth['start']}")
        if truth.get("earliest"):
            parts.append(f"from {truth['earliest']}")
        if truth.get("attendees"):
            parts.append("with " + ", ".join(people[p] for p in truth["attendees"]))
        if truth.get("per_day_max"):
            parts.append(f"at most {truth['per_day_max']} a day")
        if "max_meetings" in truth:
            parts.append(f"only on days with at most {truth['max_meetings']} other meetings")
        parts.append("everyone free")
        text = "Book " + ", ".join(parts)
        if truth.get("accepted_start"):
            text += f"; hidden preference {truth['accepted_start']}"
        return text
    if kind == "move":
        e = events[truth["event"]]
        text = f"Move {e['title']} ({truth['event']}) to {DAY[truth['day']]} {truth['start']}"
        if truth.get("confirm_ok"):
            text += f", or hold it for confirmation ({people[e['owner']]} organized it)"
        return text
    if kind == "answer":
        return "Answer mentions " + needs(truth["must_mention"])
    if kind == "reply":
        m = messages[truth["message"]]
        verb = "Send" if truth.get("sent") else "Draft or hold"
        return f"{verb} a reply to {truth['message']} ({people[m['from']]}) mentioning {needs(truth['must_mention'])}"
    if kind == "ask":
        return f"Ask, naming {needs(truth['candidates'])}; or do the intended thing: {truth_text(truth['intended'])}"
    return kind


lines = ["# The chapter 6 request set", "",
         "Generated from `requests.json`; edit that file, not this view. Each request has hidden truth "
         "(what the user accepts), the user's feedback, and the scripted stand-in's actions on each path. "
         "The last column is what those scripted actions score: fast, deliberate, ask, with no monitor.", "",
         "Splits: 40 training requests feed the search; 20 held-out requests are scored only in the final "
         "comparison; the walkthrough request appears only in listing 6.2.", ""]
for category, title in (("fast", "The fast path pays"), ("deliberate", "Deliberation pays"),
                        ("ask", "Asking pays"), ("confirm", "Confirming pays"), ("memory", "Memory pays")):
    rows = [r for r in requests if r["category"] == category]
    lines += [f"## {title} ({len(rows)})", "", "| id | split | request | what the user accepts | scripted |",
              "|---|---|---|---|---|"]
    for r in rows:
        extra = []
        if r["user"] != "maya":
            extra.append(f"user {people[r['user']]}")
        if r["confirmed"]:
            extra.append("the request is the go-ahead")
        if r["memory"]:
            extra.append("memory: " + " ".join(m.get("lesson") or m.get("final_output", "") for m in r["memory"]))
        note = f" ({'; '.join(extra)})" if extra else ""
        e = r["expected"]
        lines.append(f"| {r['id']} | {r['split']} | {r['text']}{note} | {truth_text(r['truth'])} | "
                     f"{e['fast']:.0f} / {e['deliberate']:.0f} / {e['ask']:.0f} |")
    lines.append("")
(HERE / "requests.md").write_text("\n".join(lines), encoding="utf-8")
print("wrote requests.md")
