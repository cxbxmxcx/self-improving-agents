"""HelixAgent v1: the same agent as v0, but the system prompt is an Artifact
read from the archive.

The diff against `01_helixagent_v0.py` is one idea: `build_agent` no longer takes
the genesis prompt directly, it reads the current live champion from a
persistent SQLite Archive. If nothing has been promoted yet, v1 falls back to
v0's genesis prompt and records it as the seed. Everything else, the tools, the
loop, the trace printing, is identical.

Subsequent runs read whatever has been promoted via `archive.promote()`. SPO
can produce a higher-scoring candidate every round (section 2.4), but the agent
does not serve it until promotion happens. The improvement is delivered by
promoting a new Artifact in the archive, not by changing the agent loop. SPEC
section 11.2; DESIGN_NOTES.md section 10.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from helix.agent import Agent
from helix.archive import SQLiteArchive
from helix.artifact import Artifact, genesis, Subtype
from helix.trajectory import StepKind, Trajectory

from chapters.ch02.agent_v0 import (
    SYSTEM_PROMPT_V0,
    DEFAULT_MODEL,
    build_retrieve_tool,
)

ARCHIVE_PATH = Path(__file__).parent / "runs" / "helix_archive.sqlite"
PROMPT_ARTIFACT_ID = "prompt.helixagent.system"

# Same three questions as v0, so the answers are directly comparable.
QUESTIONS = [
    "Summarize the core idea of self-improving agents in the corpus.",
    "What four operators does ExpeL use to manage memory entries?",
    "What benchmark score is reported for 'Helix-9' agent on MMLU?",
]



def open_archive(path: Path = ARCHIVE_PATH) -> SQLiteArchive:
    path.parent.mkdir(parents=True, exist_ok=True)
    return SQLiteArchive(path)


async def get_or_create_seed(archive: SQLiteArchive) -> Artifact:
    """Return the live champion prompt, or v0's genesis if nothing is promoted.

    The running agent serves the live champion, which changes only when
    something calls `archive.promote()`. `archive.best()` ranks all candidates
    by score, but a candidate is not what the agent serves until it is
    explicitly promoted; an offline improver can leave a winning candidate
    sitting unpromoted in the archive. See DESIGN_NOTES.md section 10.

    On first run the archive is empty, so we instantiate v0's genesis prompt and
    store the artifact only, with no measurement attached: a prompt has no score
    until a Signal measures it. Callers fall back to this genesis until the
    first promotion happens. The fallback path lives entirely here.
    """
    live = await archive.live_champion(PROMPT_ARTIFACT_ID)
    if live is not None:
        return live

    existing = await archive.by_id(PROMPT_ARTIFACT_ID, version=1)
    if existing is not None:
        return existing.artifact

    seed = genesis(
        id=PROMPT_ARTIFACT_ID,
        kind=Subtype.PROMPT,
        content=SYSTEM_PROMPT_V0,
        created_by="human",
    )
    await archive.put_artifact(seed)
    return seed


async def build_agent(model: str = DEFAULT_MODEL) -> Agent:
    """The v1 agent: same as v0, but the prompt comes from the archive."""
    system_prompt = await get_or_create_seed(open_archive())  # <- the only change from v0
    return Agent(
        system_prompt=system_prompt,
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
    agent = await build_agent(model=model)  # build once: prompt comes from the archive
    print(f"Serving {agent.system_prompt.id} v{agent.system_prompt.version} (live champion)")
    for question in QUESTIONS:
        output, trajectory = await agent.run(question)
        print_trace(question, output, trajectory)


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
