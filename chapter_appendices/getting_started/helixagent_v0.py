"""HelixAgent v0 extensions: the Ch 3 tool-description scaffolding.

The v0 building blocks (genesis prompt, retrieve tool, agent factory) live in
`chapters/ch02/agent_v0.py` and are re-exported here so existing importers keep
working. This module adds the pieces Chapter 3 needs: the retrieve tool's
description promoted to a TOOL_DESCRIPTION artifact, and a tool whose
description a search method can improve. SPEC §16.2.2.

Run it directly for a single sample question against the v0 baseline, or
import the builders from a chapter script or harness.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from helix.artifact import Artifact, genesis, Subtype
from helix.env import load_env
from helix.tools import TextDescriptionTool

# The single source of truth for the v0 agent. Re-exported so chapter and
# pending-chapter scripts can keep importing everything from this module.
from chapters.ch02.agent_v0 import (
    CORPUS_PATH,
    DEFAULT_MODEL,
    RETRIEVE_DESCRIPTION_V0,
    SYSTEM_PROMPT_V0,
    _retrieve_index,
    ask_one,
    build_agent,
    build_retrieve_tool,
    build_system_prompt,
)

load_env()

# Ch 2 bakes the description string into the @tool decorator; Ch 3 promotes it
# to a TOOL_DESCRIPTION artifact so a search method can improve it.
RETRIEVE_DESCRIPTION_ID = "tool_description.helixagent.retrieve"


def build_retrieve_description_artifact() -> Artifact:
    """The genesis TOOL_DESCRIPTION artifact for the retrieve tool. Ch 3
    aims search methods at this artifact to improve how the LLM understands
    when and how to call the tool."""
    return genesis(
        id=RETRIEVE_DESCRIPTION_ID,
        kind=Subtype.TOOL_DESCRIPTION,
        content=RETRIEVE_DESCRIPTION_V0,
        created_by="human",
    )


def build_retrieve_tool_with_artifact_description(
    description_artifact: Artifact | None = None,
    corpus_path: Path = CORPUS_PATH,
) -> TextDescriptionTool:
    """The Ch 3 retrieve tool: same Python implementation, but the LLM-facing
    description is a TOOL_DESCRIPTION artifact a search method can improve.

    Pass a specific description artifact (e.g. a candidate under test) or
    omit it to use the genesis description.
    """
    index = _retrieve_index(corpus_path)

    async def retrieve(query: str, k: int = 5) -> list[dict]:
        hits = index.search(query, k=k, mode="hybrid")
        return [
            {"source": h.source, "page": h.page, "text": h.text, "score": h.score}
            for h in hits
        ]

    return TextDescriptionTool(
        name="retrieve",
        fn=retrieve,
        description_artifact=description_artifact or build_retrieve_description_artifact(),
    )


def main() -> None:
    """Run a single sample question. Replace the question to experiment."""
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


if __name__ == "__main__":
    main()
