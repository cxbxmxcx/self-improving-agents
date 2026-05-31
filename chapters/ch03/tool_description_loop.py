"""Chapter 3 §3.1 warmup (RAG agent example) — improving a tool description offline.

Everything you learned in Ch 2 §2.4 carries over to a different artifact.
Here the artifact under improvement is the retrieve tool's DESCRIPTION —
the natural-language string the LLM reads to decide whether and how to
call the tool — not the system prompt.

The agent is built with a TextDescriptionTool: a plain Python retrieve
implementation whose description is a TOOL_DESCRIPTION artifact in the
archive. An OfflineImprover targets that artifact and runs SPO rounds.
The Improver, Signal, Search, and Archive are exactly the Ch 2 shapes;
only the target_artifact_id changes.

Run:
    python chapters/ch03/tool_description_loop.py

Candidates are written to the archive but not served until a human
promotes one (the Ch 2 deploy-gate pattern). Re-run for more rounds.

Cost: ~$0.30 with the default knobs.
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
from helix.artifact import Artifact, Subtype
from helix.env import load_env
from helix.eval import FixedEvalSet, load_eval_set
from helix.improvement import OfflineImprover, ImproverPolicy, Schedule
from helix.observability import attach_console_renderer
from helix.search.spo import SPO
from helix.signals.pairwise_judge import PairwiseJudge, SwapAndAgree

from chapter_appendices.getting_started.helixagent_v0 import (
    DEFAULT_MODEL,
    RETRIEVE_DESCRIPTION_ID,
    SYSTEM_PROMPT_V0,
    build_retrieve_description_artifact,
    build_retrieve_tool_with_artifact_description,
)
from helix.artifact import genesis

load_env()

# Configuration -----------------------------------------------------------
AGENT_MODEL = DEFAULT_MODEL
PROPOSER_MODEL = "claude-sonnet-4-6"
JUDGE_MODEL = "claude-sonnet-4-6"
QUESTIONS_PER_ROUND: int | None = None
ROUNDS_TO_DRIVE = 3

ARCHIVE_PATH = Path(__file__).parent / "runs" / "ch03_archive.sqlite"
EVAL_QUESTIONS_PATH = REPO_ROOT / "chapters" / "ch02" / "eval_questions.json"
PROMPT_ARTIFACT_ID = "prompt.helixagent.system"


def open_archive(path: Path = ARCHIVE_PATH) -> SQLiteArchive:
    path.parent.mkdir(parents=True, exist_ok=True)
    return SQLiteArchive(path)


async def get_or_create_description(archive: SQLiteArchive) -> Artifact:
    """Return the live champion tool description, or genesis if none promoted."""
    live = await archive.live_champion(RETRIEVE_DESCRIPTION_ID)
    if live is not None:
        return live
    existing = await archive.by_id(RETRIEVE_DESCRIPTION_ID, version=1)
    if existing is not None:
        return existing.artifact
    seed = build_retrieve_description_artifact()
    await archive.put_artifact(seed)
    return seed


def _system_prompt_artifact() -> Artifact:
    """A fixed system prompt for the agent. In this chapter the system prompt
    is NOT under improvement — only the tool description is — so we use the
    genesis prompt directly."""
    return genesis(
        id=PROMPT_ARTIFACT_ID,
        kind=Subtype.PROMPT,
        content=SYSTEM_PROMPT_V0,
        created_by="human",
    )


async def main_async() -> None:
    attach_console_renderer(verbose=True)

    archive = open_archive()
    description = await get_or_create_description(archive)

    # Build the agent with an artifact-backed tool description. The system
    # prompt is fixed; the tool description is the artifact under search.
    agent = Agent(
        system_prompt=_system_prompt_artifact(),
        tools=[build_retrieve_tool_with_artifact_description(description)],
        model=AGENT_MODEL,
    )

    judge = SwapAndAgree(PairwiseJudge(model=JUDGE_MODEL))
    search = SPO(
        proposer_model=PROPOSER_MODEL,
        first_round_model=AGENT_MODEL,
        rounds=1,
    )
    eval_source = FixedEvalSet(load_eval_set(EVAL_QUESTIONS_PATH))
    policy = ImproverPolicy(
        schedule=Schedule.MANUAL,
        questions_per_round=QUESTIONS_PER_ROUND,
        promote_threshold_win_rate=0.5,
    )

    # The only thing different from Ch 2 §2.4: target_artifact_id points at
    # the tool description, not the system prompt. Everything else is the
    # same Improver / Signal / Search / Archive.
    improver = OfflineImprover(
        agent=agent,
        target_artifact_id=RETRIEVE_DESCRIPTION_ID,
        signal=judge,
        search=search,
        archive=archive,
        eval_source=eval_source,
        policy=policy,
        seed_fallback=description,
        improver_id="imp-tool-desc",
    )
    agent.attach_improver(improver)

    await improver.start()
    try:
        for _ in range(ROUNDS_TO_DRIVE):
            await improver.trigger_round()
    finally:
        await improver.stop()

    s = improver.status
    print()
    print("=" * 70)
    print(f"OfflineImprover {s.improver_id} stopped (target: tool description).")
    print(f"  rounds completed: {s.rounds_completed}")
    if s.last_round_result is not None:
        r = s.last_round_result
        cscore = f"{r.candidate_score:.3f}" if r.candidate_score is not None else "n/a"
        print(f"  last round: cand v{r.candidate_version} score={cscore}  promoted={r.promoted}")
    print(f"  archive: {ARCHIVE_PATH}")
    print()
    print("The agent's retrieve tool description is now improvable like any L1")
    print("artifact. Promote a winning candidate via the dashboard to make it live.")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
