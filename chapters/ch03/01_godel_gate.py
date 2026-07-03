"""Chapter 3: the Godel machine's proof gate, on a live agent.

Schmidhuber's Godel machine accepts a change to itself only when it can PROVE
the change helps. Real agents cannot prove anything useful about themselves,
so the rest of this book trades the proof for a measurement. This demo shows
the one place the proof still exists: a claim whose truth is computable.

We reuse the chapter-2 agent pattern (one system prompt artifact, no tools) and
ask it three questions. Two have a computable truth, so a one-line Python
expression is a complete proof and the gate's verdict is certain. The third has
no checker at all, and that is where measurement, and this chapter, begins. Run:

    python chapters/ch03/01_godel_gate.py
"""
from __future__ import annotations

import asyncio
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from helix.agent import Agent
from helix.artifact import Subtype, genesis
from helix.env import load_env

load_env()

MODEL = "claude-haiku-4-5"  # the chapter-2 default; any model makes the point

LINE = "strawberry fields are forever and ever"


def build_agent() -> Agent:
    """The chapter-2 agent pattern, minus the retrieve tool: one genesis prompt
    artifact and a model. Counting letters needs no corpus."""
    prompt = genesis(
        id="prompt.godelgate.system",
        kind=Subtype.PROMPT,
        content="You are a careful assistant. When asked for a number, reply "
                "with the number alone and nothing else.",
        created_by="human",
    )
    return Agent(system_prompt=prompt, model=MODEL, temperature=0.0)


def first_int(text: str) -> int | None:
    match = re.search(r"-?\d+", text)
    return int(match.group()) if match else None


def proof_gate(claimed: int | None, truth: int) -> str:
    """The Godel condition in miniature: derive the truth, compare, done.
    No sample, no judge, no confidence interval. The verdict is certain."""
    return "ACCEPT (certain)" if claimed == truth else "REJECT (certain)"


async def main_async() -> None:
    agent = build_agent()

    print(f'The line: "{LINE}"')
    print()

    # Claim 1: arithmetic. The proof is the computation itself.
    answer, _ = await agent.run("What is 1 + 1 + 1?")
    claimed = first_int(answer)
    print("Q1: What is 1 + 1 + 1?")
    print(f"  agent's answer : {claimed}")
    print(f"  the proof      : 1 + 1 + 1 = {1 + 1 + 1}  (computed, not sampled)")
    print(f"  proof gate     -> {proof_gate(claimed, 1 + 1 + 1)}")
    print()

    # Claim 2: a letter count. The proof is one expression: LINE.count("r").
    answer, _ = await agent.run(
        f'How many times does the letter r appear in "{LINE}"?'
    )
    claimed = first_int(answer)
    print("Q2: How many r's in the line?")
    print(f"  agent's answer : {claimed}")
    print(f'  the proof      : LINE.count("r") = {LINE.count("r")}  (computed, not sampled)')
    print(f"  proof gate     -> {proof_gate(claimed, LINE.count('r'))}")
    print()

    # Question 3: no checker exists. This is almost everything an agent does.
    answer, _ = await agent.run(f'Summarize this line in five words: "{LINE}"')
    print("Q3: Summarize the line in five words.")
    print(f"  agent's answer : {answer.strip()}")
    print("  the proof      : none exists. You can count the words, but no")
    print("                   expression computes whether the summary is good.")
    print()
    print("Where a truth is computable, the gate is certain and free. Almost")
    print("nothing an agent produces has one, so the gate degrades to a")
    print("measurement, and making that measurement trustworthy is this chapter.")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
