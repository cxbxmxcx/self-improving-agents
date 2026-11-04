"""Listing 6.1: a decision model answers typed questions about the assistant's requests.

The questions are the seed configuration's: two probabilities that route a
request, and a choice that checks a tool call before it runs. Each answer is
data with a confidence, and each call reports its tokens, seconds, and price.
The last request repeats an ambiguous one with a recalled lesson in the
state, which is how memory reaches the meta level.

By default the answers are replayed from a recording of Jev, so no key is
needed; --decider jev-latest asks Jev directly.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from importlib import import_module
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from helix.deciders import ask_measured, decider_from_name
from helix.memory.base import MemoryEntry, Scope, ScopeKey
from helix.memory.hooks import render_episodic
from helix.scaffold import DecisionConfig

from chapter_appendices.ch06_decision_scaffold_legacy.assistant.environment import AssistantEnvironment, load_requests
from chapter_appendices.ch06_decision_scaffold_legacy.assistant.recording import decision_model

listing = import_module("chapter_appendices.ch06_decision_scaffold_legacy.02_metacognitive_assistant")
CONFIG = DecisionConfig.model_validate(listing.SEED_CONFIG)


def recalled(request: dict) -> str:
    """The memory section the harness would add for this request's prepared lesson."""
    entries = [MemoryEntry(id=str(n), scope_key=ScopeKey(Scope.USER, request["user"]), content=item)
               for n, item in enumerate(request["memory"])]
    return render_episodic(entries).strip()


def show(name: str, answer) -> str:
    if answer.kind == "choice":
        return f"{name} = {answer.value} (confidence {answer.confidence:.2f})"
    return f"{name} = {float(answer.value):.2f}"


async def main(mode: str = "replay", name: str = "jev-latest") -> None:
    # Live asks the decision model directly, so its timings are today's.
    decider = decider_from_name(name) if mode == "live" else decision_model(mode)
    requests = {r["id"]: r for r in load_requests()}
    env = AssistantEnvironment()
    print("Decision model:", decider.name)
    for rid in ("F01", "D01", "A01", "M05"):
        request = requests[rid]
        env.begin(request)
        state = {"request": request["text"]}
        if request["memory"]:
            state["memory"] = recalled(request)
        state = env.decision_state("route", state, None)
        answers, call = await ask_measured(decider, state, CONFIG.questions_for("route"))
        print(f"{rid} {request['text']}")
        print("   " + "; ".join(show(n, a) for n, a in answers.items()),
              f"| {call.input_tokens} tokens, {call.seconds:.2f} s, ${call.dollars:.6f}")
    # A tool check: the state is the proposed call and its target.
    env.begin(requests["C01"])
    proposed = {"request": requests["C01"]["text"], "tool": "send_reply",
                "arguments": {"message_id": "M1", "body": "Hi Dana, Thursday at 13:00 works."}}
    state = env.decision_state("tool", proposed, None)
    answers, call = await ask_measured(decider, state, CONFIG.questions_for("tool"))
    print("C01 send_reply to Dana Park")
    print("   " + "; ".join(show(n, a) for n, a in answers.items()),
          f"| {call.input_tokens} tokens, {call.seconds:.2f} s, ${call.dollars:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--record", action="store_true", help="Add missing answers to the recording with live Jev.")
    group.add_argument("--live", action="store_true", help="Ask the decision model directly.")
    parser.add_argument("--decider", default="jev-latest", help="With --live: jev-latest, openai/<model>, or llm/<model>.")
    args = parser.parse_args()
    mode = "live" if args.live else "record" if args.record else "replay"
    if mode != "replay":
        from helix.env import load_env
        load_env()
    asyncio.run(main(mode, args.decider))
