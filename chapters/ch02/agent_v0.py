"""Shared v0 building blocks for Chapter 2.

The genesis system prompt, the retrieve tool, and the agent factory that the
chapter's listings build on. This is support code, not a numbered listing: the
runnable baseline is `01_helixagent_v0.py`, which imports from here, and
`02_helixagent_v1.py` reuses the same prompt text and tool so that the only
difference between v0 and v1 is where the system prompt comes from.

Only the pieces Chapter 2 needs live here. The Ch 3 tool-description
scaffolding (the TOOL_DESCRIPTION artifact and the artifact-backed retrieve
tool) stays in `chapter_appendices/getting_started/helixagent_v0.py`, which is
where Chapter 3 imports it from. SPEC section 11.1.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from helix.agent import Agent
from helix.artifact import Artifact, genesis, Subtype
from helix.env import load_env
from helix.retrieval.index import open_index
from helix.tools import Tool, tool
from helix.trajectory import Trajectory

load_env()


CORPUS_PATH = REPO_ROOT / "data" / "helix_corpus.lance"
DEFAULT_MODEL = "claude-haiku-4-5"

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
