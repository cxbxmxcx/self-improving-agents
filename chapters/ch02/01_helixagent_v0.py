"""HelixAgent v0: a working RAG + ReAct agent, no self-improvement.

This is the starting line for the chapter: a basic agent you can run before any
archive, signal, or search exists. One system prompt (an Artifact), one
retrieval tool against the corpus, working memory only. Pure agent loop. SPEC
section 11.1.

The point of this listing is the "before." Run it, watch it answer, and note
where it struggles (band-4 trap questions in the eval set have no answer in the
corpus; v0 tends to guess). Section §2.4 produces a candidate prompt that beats
this baseline, and the diff against `02_helixagent_v1.py` is the chapter's whole
thesis: v1 is the same agent, the only change is that v1 reads its system prompt
from the archive instead of a hardcoded genesis. The agent loop never changes.

The shared v0 building blocks (genesis prompt, retrieve tool, agent factory)
live in `agent_v0.py` next to this file; this listing imports them and runs the
agent where the reader can see it. The fully annotated build, including the Ch 3
tool-description scaffolding, lives in
`chapter_appendices/getting_started/helixagent_v0.py`.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from helix.agent import Agent
from helix.env import load_env
from helix.trajectory import Trajectory

from chapters.ch02.agent_v0 import (
    DEFAULT_MODEL,
    build_retrieve_tool,
    build_system_prompt,
)

load_env()


async def build_agent(model: str = DEFAULT_MODEL) -> Agent:
    """The v0 agent: genesis system prompt, one retrieve tool, no archive.

    The system prompt is the genesis Artifact, taken directly. There is no
    archive and no live champion yet; that is exactly the difference v1
    introduces in `02_helixagent_v1.py`.
    """
    return Agent(
        system_prompt=build_system_prompt(),
        tools=[build_retrieve_tool()],
        model=model,
    )


async def ask_one(question: str, model: str = DEFAULT_MODEL) -> tuple[str, Trajectory]:
    agent = await build_agent(model=model)
    output, trajectory = await agent.run(question)
    return output, trajectory


def main() -> None:
    """Run a single sample question against the v0 baseline."""
    question = "Summarize the main idea of self-improving agents based on the corpus."
    output, trajectory = asyncio.run(ask_one(question))

    print("=" * 70)
    print(f"Q: {question}")
    print("=" * 70)
    print(f"\n{output}\n")
    print("-" * 70)
    print(f"Trajectory: {trajectory.id}")
    print(f"Steps: {len(trajectory.steps)}  Outcome: {trajectory.outcome.value}")
    print(f"Artifacts used: {trajectory.artifacts_used}")
    print()
    print("This is the baseline. For the measured 'before' across all 20 eval")
    print("questions, run `python chapters/ch02/eval_harness.py` (no judge).")
    print("Then `python chapters/ch02/02_helixagent_v1.py` to see the same agent")
    print("read its prompt from the archive instead.")


if __name__ == "__main__":
    main()
