"""HelixAgent v0: RAG + ReAct, no self-improvement.

The minimal agent from Spec section 11.1. One system prompt (an Artifact), one
retrieval tool against the LanceDB corpus, working memory only. Pure agent
loop. No archive, no signal, no search. This is the baseline that v1 improves
upon.

Run it directly (F5 / Run in your editor) or import and call build_agent() /
ask_one() from a harness.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from helix.agent import Agent
from helix.artifact import Artifact, ArtifactKind, genesis, Subtype
from helix.env import load_env
from helix.retrieval.index import open_index
from helix.tools import TextDescriptionTool, Tool, tool
from helix.trajectory import Trajectory

load_env()


CORPUS_PATH = REPO_ROOT / "data" / "helix_corpus.lance"
DEFAULT_MODEL = "claude-haiku-4-5"

# The retrieve tool's LLM-facing description, as a string. Ch 2 bakes this
# into the @tool decorator; Ch 3 promotes it to a TOOL_DESCRIPTION artifact
# so a search method can improve it. SPEC §16.2.2.
RETRIEVE_DESCRIPTION_ID = "tool_description.helixagent.retrieve"
RETRIEVE_DESCRIPTION_V0 = (
    "Search the document corpus for passages relevant to a query. Returns the "
    "top matching passages with their source filename and page."
)

SYSTEM_PROMPT_V0 = """You are HelixAgent, a research assistant.

When the user asks a question, use the `retrieve` tool to look up relevant
passages from the corpus, then answer based on what you found. Quote or cite
the source filename when you use retrieved content. If the corpus does not
contain the answer, say so directly rather than guessing.
"""


def _retrieve_index(corpus_path: Path = CORPUS_PATH):
    if not corpus_path.exists():
        raise FileNotFoundError(
            f"No corpus found at {corpus_path}. "
            "Run `python ingestion/download_corpus.py` then `python ingestion/build_index.py`."
        )
    return open_index(corpus_path)


def build_retrieve_tool(corpus_path: Path = CORPUS_PATH) -> Tool:
    """The Ch 2 retrieve tool: description baked into the @tool decorator."""
    index = _retrieve_index(corpus_path)

    @tool(description=RETRIEVE_DESCRIPTION_V0)
    async def retrieve(query: str, k: int = 5) -> list[dict]:
        hits = index.search(query, k=k, mode="hybrid")
        return [
            {"source": h.source, "page": h.page, "text": h.text, "score": h.score}
            for h in hits
        ]

    return retrieve


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


def build_system_prompt() -> Artifact:
    """The genesis system prompt artifact. v1 mutates this."""
    return genesis(
        id="prompt.helixagent.system",
        kind=Subtype.PROMPT,
        content=SYSTEM_PROMPT_V0,
        created_by="human",
    )


def build_agent(model: str = DEFAULT_MODEL) -> Agent:
    return Agent(
        system_prompt=build_system_prompt(),
        tools=[build_retrieve_tool()],
        model=model,
    )


async def ask_one(question: str, model: str = DEFAULT_MODEL) -> tuple[str, Trajectory]:
    agent = build_agent(model=model)
    output, trajectory = await agent.run(question)
    return output, trajectory


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
