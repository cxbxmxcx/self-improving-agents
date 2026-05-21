"""Researcher agent: retrieval-only, intentionally without improvers.

A minimal agent spec demonstrating the no-improvers code path. The spec
exports only `build()` and `build_genesis_prompt()` — it does NOT export
`build_improvers()` or `list_improvable_artifacts()`.

The chat UI's improver panel detects the absence and shows a message
pointing users at this file rather than offering attachment. This makes
clear that self-improvement is opt-in per spec, not a platform default
forced on every agent.

Use this when you want a stable reference agent that should never mutate
its prompt under live traffic (e.g., as a baseline against which other
agents are compared).
"""

from __future__ import annotations

from pathlib import Path

from helix.agent import Agent
from helix.artifact import Artifact, ArtifactKind, genesis
from helix.retrieval.index import open_index
from helix.tools import Tool, tool


DESCRIPTION = (
    "Retrieval-only research agent with a frozen prompt. No improvers "
    "declared — useful as a stable baseline."
)

DEFAULT_MODEL = "claude-haiku-4-5"
SYSTEM_PROMPT_ID = "prompt.researcher.system"

REPO_ROOT = Path(__file__).resolve().parent.parent
CORPUS_PATH = REPO_ROOT / "data" / "helix_corpus.lance"


DEFAULT_SYSTEM_PROMPT = """You are a careful research assistant.

When the user asks a question:
1. Call `retrieve` to look up relevant passages from the corpus.
2. Answer strictly from what you retrieved. Cite the source filename.
3. If the corpus doesn't cover the topic, say so plainly. Do not guess.

Keep answers concise and grounded in citations."""


def build_genesis_prompt() -> Artifact:
    return genesis(
        id=SYSTEM_PROMPT_ID,
        kind=ArtifactKind.PROMPT,
        content=DEFAULT_SYSTEM_PROMPT,
        created_by="human",
    )


_retrieval_index = None


def _get_index():
    global _retrieval_index
    if _retrieval_index is None:
        if not CORPUS_PATH.exists():
            raise FileNotFoundError(
                f"No corpus found at {CORPUS_PATH}. "
                "Run `python ingestion/build_index.py` first."
            )
        _retrieval_index = open_index(CORPUS_PATH)
    return _retrieval_index


def _build_retrieve_tool() -> Tool:
    index = _get_index()

    @tool(description=(
        "Search the document corpus for passages relevant to a query. "
        "Returns the top matching passages with source filename and page."
    ))
    async def retrieve(query: str, k: int = 5) -> list[dict]:
        hits = index.search(query, k=k, mode="hybrid")
        return [
            {"source": h.source, "page": h.page, "text": h.text, "score": h.score}
            for h in hits
        ]

    return retrieve


def build(
    system_prompt: Artifact | None = None,
    *,
    model: str = DEFAULT_MODEL,
    max_iterations: int = 10,
    max_tool_calls: int = 20,
    skip_memory: bool = False,  # accepted but unused; researcher has no memory
    **overrides,
) -> Agent:
    """Construct a fresh researcher Agent.

    No memory tiers, no improvers — just a retrieval tool and a frozen
    prompt. `skip_memory` is accepted to keep the call signature
    compatible with the loader and improver-test harnesses.
    """
    if system_prompt is None:
        system_prompt = build_genesis_prompt()

    return Agent(
        system_prompt=system_prompt,
        tools=[_build_retrieve_tool()],
        model=model,
        max_iterations=max_iterations,
        max_tool_calls=max_tool_calls,
        memory_tiers={},
        **overrides,
    )
