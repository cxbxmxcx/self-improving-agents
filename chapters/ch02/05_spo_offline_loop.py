"""Chapter 2 §2.4 — the offline improvement loop end to end.

This script runs SPO against a fixed eval set with labeled reference
answers, using a pairwise LLM-as-judge as the Signal. Candidates are
written to the archive with measurements, but the running agent does
not switch to a new prompt automatically: a human (or the dashboard's
promote button) decides when to call `archive.promote()`. Auto-promotion
is a Ch 8 topic, paired with HITL and live feedback.

Recipe:

  1. Build an Agent and identify which Artifact to improve.
  2. Construct an Improver with a Signal, a Search, an EvalSource, an
     Archive, and a Policy.
  3. Attach the Improver to the Agent. Start it.
  4. Drive rounds (manually, on an interval, or continuously).
  5. The console renderer prints every step as it happens.

Re-run the script and the next round picks up the highest-scoring
candidate from the archive as the new reference. No state to hand around:
the Archive is the contract.
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
from helix.eval import FixedEvalSet, load_eval_set
from helix.improvement import OfflineImprover, ImproverPolicy, Schedule
from helix.observability import attach_console_renderer
from helix.search.spo import SPO
from helix.signals.pairwise_judge import PairwiseJudge, SwapAndAgree

# Listing 02's module name starts with a digit, so the v1 archive helpers are
# loaded with importlib rather than a plain `import`.
import importlib

_v1 = importlib.import_module("chapters.ch02.02_helixagent_v1")
ARCHIVE_PATH = _v1.ARCHIVE_PATH
PROMPT_ARTIFACT_ID = _v1.PROMPT_ARTIFACT_ID
get_or_create_seed = _v1.get_or_create_seed
open_archive = _v1.open_archive

from chapters.ch02.agent_v0 import build_retrieve_tool

load_env()

# Configuration -----------------------------------------------------------
AGENT_MODEL = "claude-haiku-4-5"
PROPOSER_MODEL = "claude-sonnet-4-6"
JUDGE_MODEL = "claude-sonnet-4-6"
QUESTIONS_PER_ROUND: int | None = None  # None = all 20
ROUNDS_TO_DRIVE = 3

EVAL_QUESTIONS_PATH = Path(__file__).parent / "eval_questions.json"


async def main_async() -> None:
    # 1) Console subscriber so we see every step stream to the terminal.
    attach_console_renderer(verbose=True)

    # 2) Persistent archive + genesis seed if empty.
    archive = open_archive()
    seed = await get_or_create_seed(archive)

    # 3) Define the Agent. The OfflineImprover clones this definition via
    #    `agent.with_artifacts({PROMPT_ARTIFACT_ID: candidate})` to test
    #    each candidate against the eval set. SPEC §15.2.
    agent = Agent(
        system_prompt=seed,
        tools=[build_retrieve_tool()],
        model=AGENT_MODEL,
    )

    # 4) Compose Signal, Search, EvalSource, Policy.
    judge = SwapAndAgree(PairwiseJudge(model=JUDGE_MODEL))
    # Cheaper Haiku for the first round (no feedback to incorporate yet);
    # Sonnet from round 2 onward when there's judge feedback to act on.
    search = SPO(
        proposer_model=PROPOSER_MODEL,
        first_round_model=AGENT_MODEL,  # = Haiku, same as agent
        rounds=1,  # SPO propose called once per OfflineImprover.trigger_round()
    )
    eval_source = FixedEvalSet(load_eval_set(EVAL_QUESTIONS_PATH))
    # OfflineImprover writes candidates to the archive but does not change
    # what the running agent serves. Promotion (live_champion swap) happens
    # separately via `archive.promote()` once a human reviews the candidate.
    policy = ImproverPolicy(
        schedule=Schedule.MANUAL,
        questions_per_round=QUESTIONS_PER_ROUND,
        promote_threshold_win_rate=0.5,
    )

    # 5) Build the OfflineImprover and attach it to the agent (for dashboard
    #    introspection; structurally a no-op for offline improvement).
    improver = OfflineImprover(
        agent=agent,
        target_artifact_id=PROMPT_ARTIFACT_ID,
        signal=judge,
        search=search,
        archive=archive,
        eval_source=eval_source,
        policy=policy,
        seed_fallback=seed,
    )
    agent.attach_improver(improver)

    # 6) Start the background loop and drive rounds explicitly. With
    #    Schedule.MANUAL the background task stays idle until trigger_round().
    await improver.start()
    try:
        for _ in range(ROUNDS_TO_DRIVE):
            await improver.trigger_round()
    finally:
        await improver.stop()

    # 7) Status summary.
    s = improver.status
    print()
    print("=" * 70)
    print(f"OfflineImprover {s.improver_id} stopped.")
    print(f"  rounds completed: {s.rounds_completed}")
    if s.last_round_result is not None:
        r = s.last_round_result
        print(f"  last round: cand v{r.candidate_version} score={r.candidate_score:.3f}  "
              f"ref score={r.reference_score:.3f}  promoted={r.promoted}")
    print(f"  archive: {ARCHIVE_PATH}")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
