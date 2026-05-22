"""Chapter 2 demo: improving HelixAgent's system prompt with an Improver.

The user-facing surface for self-improvement, in one chapter script. The
orchestration math, per-question judging, aggregation, archive recording,
and event emission all live in helix.improvement.* — this script is how a
reader operates that platform.

Recipe:

  1. Build an Agent and identify which Artifact to improve.
  2. Construct an Improver with a Signal, a Search, an EvalSource, an
     Archive, and a Policy.
  3. Attach the Improver to the Agent. Start it.
  4. Drive rounds (manually, on an interval, or continuously).
  5. The console renderer prints every step as it happens.

Re-run the script and the next round picks up the current champion from the
archive. No state to hand around: the Archive is the contract.
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
from helix.improvement import Improver, ImproverMode, ImproverPolicy, Schedule
from helix.observability import attach_console_renderer
from helix.search.spo import SPO
from helix.signals.pairwise_judge import PairwiseJudge, SwapAndAgree

from chapters.ch02.helixagent_v1 import (
    ARCHIVE_PATH,
    PROMPT_ARTIFACT_ID,
    get_or_create_seed,
    open_archive,
)
from chapters.ch02.helixagent_v0 import build_retrieve_tool

load_env()

# Configuration -----------------------------------------------------------
AGENT_MODEL = "claude-haiku-4-5"
PROPOSER_MODEL = "claude-sonnet-4-6"
JUDGE_MODEL = "claude-sonnet-4-6"
QUESTIONS_PER_ROUND: int | None = None  # None = all 20
ROUNDS_TO_DRIVE = 3

EVAL_QUESTIONS_PATH = Path(__file__).parent / "eval_questions.json"


async def build_agent_with_prompt(prompt, *, max_iterations: int = 10, max_tool_calls: int = 20) -> Agent:
    """Construct an Agent backed by a specific system prompt artifact.

    Caps are surfaced as parameters so the Improver can clamp them per-policy
    without the chapter script needing to know about it.
    """
    return Agent(
        system_prompt=prompt,
        tools=[build_retrieve_tool()],
        model=AGENT_MODEL,
        max_iterations=max_iterations,
        max_tool_calls=max_tool_calls,
    )


async def main_async() -> None:
    # 1) Console subscriber so we see every step stream to the terminal.
    attach_console_renderer(verbose=True)

    # 2) Persistent archive + genesis seed if empty.
    archive = open_archive()
    seed = await get_or_create_seed(archive)

    # 3) Compose Signal, Search, EvalSource, Policy.
    judge = SwapAndAgree(PairwiseJudge(model=JUDGE_MODEL))
    # Cheaper Haiku for the first round (no feedback to incorporate yet);
    # Sonnet from round 2 onward when there's judge feedback to act on.
    search = SPO(
        proposer_model=PROPOSER_MODEL,
        first_round_model=AGENT_MODEL,  # = Haiku, same as agent
        rounds=1,  # SPO propose called once per Improver.trigger_round()
    )
    eval_source = FixedEvalSet(load_eval_set(EVAL_QUESTIONS_PATH))
    # mode=OFFLINE is the default, made explicit here for teaching: offline
    # improvers write candidates to the archive but do not change what the
    # running agent serves. Promotion (live_champion swap) happens separately
    # via `archive.promote()` once a human reviews the candidate.
    policy = ImproverPolicy(
        schedule=Schedule.MANUAL,
        mode=ImproverMode.OFFLINE,
        questions_per_round=QUESTIONS_PER_ROUND,
        promote_threshold_win_rate=0.5,
    )

    # 4) Build the Improver, attach it to a representative Agent.
    agent = await build_agent_with_prompt(seed)
    improver = Improver(
        target_artifact_id=PROMPT_ARTIFACT_ID,
        signal=judge,
        search=search,
        archive=archive,
        eval_source=eval_source,
        build_agent_with_prompt=build_agent_with_prompt,
        policy=policy,
        seed_fallback=seed,
    )
    agent.attach_improver(improver)

    # 5) Start the background loop and drive rounds explicitly. With
    #    Schedule.MANUAL the background task stays idle until trigger_round().
    await improver.start()
    try:
        for _ in range(ROUNDS_TO_DRIVE):
            await improver.trigger_round()
    finally:
        await improver.stop()

    # 6) Status summary.
    s = improver.status
    print()
    print("=" * 70)
    print(f"Improver {s.improver_id} stopped.")
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
