"""HelixAgent v0: a working RAG + ReAct agent, no self-improvement.

This is the starting line for the chapter: a basic agent you can run before any
archive, signal, or search exists. One system prompt (an Artifact), one
retrieval tool against the corpus, working memory only. Pure agent loop. SPEC
section 11.1.

The whole listing is the "build once, run many" block in `main_async`. Building
the agent opens the corpus index, so build it once and reuse it across
questions; that reuse is safe because `Agent.run()` rebuilds its working memory
per call and v0 wires no persistent memory tiers, so nothing leaks between
questions. The band-4 trap question has no answer in the corpus, so watch
whether v0 abstains or guesses; that weakness is what section 2.4 improves, and
the only change v1 makes is to read its prompt from the archive instead.

The shared v0 building blocks (genesis prompt, retrieve tool, agent factory)
live in `agent_v0.py` next to this file.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from helix.agent import Agent
from helix.trajectory import StepKind, Trajectory

from chapters.ch02.agent_v0 import (
    DEFAULT_MODEL,
    build_retrieve_tool,
    build_system_prompt,
)

# An open synthesis question, a single-paper factual lookup, and a band-4 trap
# whose answer is not in the corpus.
QUESTIONS = [
    "Summarize the core idea of self-improving agents in the corpus.",
    "What four operators does ExpeL use to manage memory entries?",
    "What benchmark score is reported for 'Helix-9' agent on MMLU?",
]



def build_agent(model: str = DEFAULT_MODEL) -> Agent:
    """The v0 agent: genesis prompt, one retrieve tool, no archive."""
    return Agent(
        system_prompt=build_system_prompt(),
        tools=[build_retrieve_tool()],
        model=model,
    )


def print_trace(question: str, output: str, trajectory: Trajectory) -> None:
    """Print the answer plus one line per retrieve call the agent made."""
    print(f"\nQ: {question}\n\n{output}\n")
    for step in trajectory.steps:
        if step.kind == StepKind.TOOL_CALL:
            args = ", ".join(f"{k}={v!r}" for k, v in step.payload["arguments"].items())
            print(f"  -> {step.payload['name']}({args})")
    print(f"  [{len(trajectory.steps)} steps, outcome={trajectory.outcome.value}]")


async def main_async(model: str = DEFAULT_MODEL) -> None:
    agent = build_agent(model=model)  # build once: this opens the corpus index
    for question in QUESTIONS:
        output, trajectory = await agent.run(question)
        print_trace(question, output, trajectory)


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
