"""Two Improvers on the same Agent, strict alternation.

Demonstrates the platform's multi-Improver capability: an SPO Improver and a
GEPA Improver both target the same artifact (the system prompt) and share the
same Archive. Strict alternation drives one SPO round, then one GEPA round,
then SPO again, etc. The Archive arbitrates by score regardless of which
Search produced each candidate.

Pedagogical point: two Search methods can target the same artifact in parallel
without coordinating. The Archive is the single source of truth, and
`archive.best()` returns the highest-scoring candidate across both Searches.
That candidate is not yet what the running agent serves; promotion (the
live champion swap via `archive.promote()`) is a separate step. SPEC
section 5: 'A Search proposes-and-selects against any artifact kind it is
compatible with.' DESIGN_NOTES.md section 10.

Re-run the script to drive more alternating rounds.
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
from helix.search.gepa import GEPA
from helix.search.spo import SPO
from helix.signals.pairwise_judge import PairwiseJudge, SwapAndAgree
from helix.signals.reflection import Reflection

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
QUESTIONS_PER_ROUND: int | None = None  # None = all questions in eval_questions_v2.json
TOTAL_ROUNDS = 4  # 2 SPO + 2 GEPA in strict alternation

# GEPA configuration: population=4, generations=2 gives a real genetic
# algorithm (8 candidates per round, ~3x SPO cost per GEPA round).
GEPA_POPULATION = 4
GEPA_GENERATIONS = 2

EVAL_QUESTIONS_PATH = Path(__file__).parent / "eval_questions_v2.json"


async def main_async() -> None:
    attach_console_renderer(verbose=True)

    archive = open_archive()
    seed = await get_or_create_seed(archive)

    # Define the Agent once. Both improvers clone it via `agent.with_artifacts`
    # to test their candidates. SPEC §15.2.
    agent = Agent(
        system_prompt=seed,
        tools=[build_retrieve_tool()],
        model=AGENT_MODEL,
    )

    judge = SwapAndAgree(PairwiseJudge(model=JUDGE_MODEL))
    eval_source = FixedEvalSet(load_eval_set(EVAL_QUESTIONS_PATH))
    policy = ImproverPolicy(
        schedule=Schedule.MANUAL,
        questions_per_round=QUESTIONS_PER_ROUND,
        promote_threshold_win_rate=0.5,
    )

    spo_improver = OfflineImprover(
        agent=agent,
        target_artifact_id=PROMPT_ARTIFACT_ID,
        signal=judge,
        search=SPO(
            proposer_model=PROPOSER_MODEL,
            first_round_model=AGENT_MODEL,
            rounds=1,
        ),
        archive=archive,
        eval_source=eval_source,
        policy=policy,
        seed_fallback=seed,
        improver_id="imp-spo",
    )

    gepa_improver = OfflineImprover(
        agent=agent,
        target_artifact_id=PROMPT_ARTIFACT_ID,
        signal=judge,
        search=GEPA(
            proposer_model=PROPOSER_MODEL,
            reflector=Reflection(model=PROPOSER_MODEL),
            agent=agent,
            eval_source=eval_source,
            population_size=GEPA_POPULATION,
            generations=GEPA_GENERATIONS,
        ),
        archive=archive,
        eval_source=eval_source,
        policy=policy,
        seed_fallback=seed,
        improver_id="imp-gepa",
    )

    agent.attach_improver(spo_improver)
    agent.attach_improver(gepa_improver)
    print(f"\nAttached {len(agent.improvers)} improvers to agent (SPO + GEPA)")

    await spo_improver.start()
    await gepa_improver.start()
    try:
        # Strict alternation: SPO, GEPA, SPO, GEPA, ...
        improvers = [spo_improver, gepa_improver]
        for r in range(TOTAL_ROUNDS):
            active = improvers[r % len(improvers)]
            print(f"\n========== Round {r+1}/{TOTAL_ROUNDS}: {active.improver_id} ==========")
            await active.trigger_round()
    finally:
        await spo_improver.stop()
        await gepa_improver.stop()

    print()
    print("=" * 70)
    print("Both improvers stopped.")
    for imp in (spo_improver, gepa_improver):
        s = imp.status
        last = s.last_round_result
        print(f"\n  {s.improver_id}: rounds={s.rounds_completed}")
        if last is not None:
            print(f"    last round: cand v{last.candidate_version} score={last.candidate_score:.3f}  "
                  f"ref score={last.reference_score:.3f}  promoted={last.promoted}")
    top = await archive.best(k=3)
    print(f"\n  Archive top 3 (overall champions across both improvers):")
    for v in top:
        print(f"    v{v.artifact.version} by {v.artifact.created_by}  score={v.measurement.score:.3f}")
    print(f"\n  Archive: {ARCHIVE_PATH}")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
