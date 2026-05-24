"""HelixAgent v1: same agent as v0, but the system prompt is an Artifact under
search.

The difference from v0 is one line: `build_system_prompt()` now reads the
current live champion prompt from a persistent SQLite Archive. If no version
has been promoted yet, v1 falls back to v0's genesis prompt and records it as
the seed.

Subsequent runs read whatever has been promoted as the live champion via
`archive.promote()`. SPO can produce a higher-scoring candidate every round,
but the agent does not serve it until promotion happens. There is no other
change to the agent. The improvement is delivered by promoting a new Artifact
in the archive, not by changing the agent loop or the tool wiring. SPEC
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
from helix.artifact import Artifact, ArtifactKind, genesis
from helix.env import load_env
from helix.search.base import Variant
from helix.signal import GapMeasurement
from helix.trajectory import Trajectory

from chapter_appendices.getting_started.helixagent_v0 import (
    SYSTEM_PROMPT_V0,
    DEFAULT_MODEL,
    build_retrieve_tool,
)

load_env()

ARCHIVE_PATH = Path(__file__).parent / "runs" / "helix_archive.sqlite"
PROMPT_ARTIFACT_ID = "prompt.helixagent.system"


def open_archive(path: Path = ARCHIVE_PATH) -> SQLiteArchive:
    path.parent.mkdir(parents=True, exist_ok=True)
    return SQLiteArchive(path)


async def get_or_create_seed(archive: SQLiteArchive) -> Artifact:
    """Return the live champion prompt, or v0's genesis if nothing is promoted.

    The running agent serves the live champion, which changes only when
    something calls `archive.promote()`. `archive.best()` ranks all
    candidates by score, but a candidate is not what the agent serves until
    it is explicitly promoted. Offline improvers can produce a winning
    candidate that sits unpromoted in the archive; the agent ignores it
    until a human (or an online auto-promotion hook) promotes it. See
    DESIGN_NOTES.md section 10.

    On first run the archive is empty. We instantiate v0's genesis prompt
    but DO NOT attach a measurement to it: a prompt has no score until a
    Signal measures it. Callers fall back to this genesis until the first
    promotion happens. The fallback path lives entirely here.
    """
    live = await archive.live_champion(PROMPT_ARTIFACT_ID)
    if live is not None:
        return live

    # Look up an unmeasured genesis if we recorded one previously
    existing = await archive.by_id(PROMPT_ARTIFACT_ID, version=1)
    if existing is not None:
        return existing.artifact

    # First time ever: instantiate genesis, store the artifact only (no
    # measurement), and return it as the unscored fallback.
    seed = genesis(
        id=PROMPT_ARTIFACT_ID,
        kind=ArtifactKind.PROMPT,
        content=SYSTEM_PROMPT_V0,
        created_by="human",
    )
    archive._store_artifact(seed)  # type: ignore[attr-defined]
    archive._conn.commit()  # type: ignore[attr-defined]
    return seed


async def build_agent(model: str = DEFAULT_MODEL) -> Agent:
    archive = open_archive()
    system_prompt = await get_or_create_seed(archive)
    return Agent(
        system_prompt=system_prompt,
        tools=[build_retrieve_tool()],
        model=model,
    )


async def ask_one(question: str, model: str = DEFAULT_MODEL) -> tuple[str, Trajectory]:
    agent = await build_agent(model=model)
    output, trajectory = await agent.run(question)
    return output, trajectory


def main() -> None:
    """Run a single sample question against the current-best prompt."""
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
    print("Tip: run `python chapters/ch02/spo_offline_loop.py` to improve the system prompt offline.")
    print("     Ch 3 adds evolutionary search; Ch 8 covers online auto-promotion.")


if __name__ == "__main__":
    main()
