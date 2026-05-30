"""Chapter 3 §3.4 — escalating strategy chain: SPO -> GEPA -> DGM.

Demonstrates the StrategyChain Search: an ordered list of Search methods
with a failure budget per method. Cheap methods (SPO) try first; expensive
methods escalate only when the cheaper ones stop producing wins. Promotion
resets the active strategy's failure count; failures past the budget rotate
to the next strategy.

The escalation order is a cost ladder:
  - SPO: one mutation per round, cheapest.
  - GEPA: population evolution, ~Nx the candidates per round.
  - DGM: archive-evolutionary, samples from full history.

The artifact under improvement is the retrieve tool's description (the same
target as dual_improver.py). Compare the two scripts: dual_improver runs all
three methods every round in round-robin; this one runs ONE method until it
plateaus, then escalates.

Re-run the script to drive more rounds against whatever strategy is
currently active in the chain.

Cost: ~$1 with the default knobs.
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
from helix.artifact import Artifact, ArtifactKind, genesis, Subtype
from helix.env import load_env
from helix.eval import FixedEvalSet, load_eval_set
from helix.improvement import OfflineImprover, ImproverPolicy, Schedule
from helix.observability import attach_console_renderer
from helix.search import StrategyChain
from helix.search.dgm import BlindLLMMutator, DGMSearch
from helix.search.gepa import GEPA
from helix.search.spo import SPO
from helix.signals.pairwise_judge import PairwiseJudge, SwapAndAgree
from helix.signals.reflection import Reflection

from chapter_appendices.getting_started.helixagent_v0 import (
    DEFAULT_MODEL,
    RETRIEVE_DESCRIPTION_ID,
    SYSTEM_PROMPT_V0,
    build_retrieve_description_artifact,
    build_retrieve_tool_with_artifact_description,
)

load_env()

# Configuration -----------------------------------------------------------
AGENT_MODEL = DEFAULT_MODEL
PROPOSER_MODEL = "claude-sonnet-4-6"
JUDGE_MODEL = "claude-sonnet-4-6"
QUESTIONS_PER_ROUND: int | None = None
ROUNDS_TO_DRIVE = 5
MAX_FAILURES_PER_STRATEGY = 2  # rotate after 2 consecutive failures

GEPA_POPULATION = 4
GEPA_GENERATIONS = 2

ARCHIVE_PATH = Path(__file__).parent / "runs" / "ch03_archive.sqlite"
EVAL_QUESTIONS_PATH = Path(__file__).parent / "eval_questions_v2.json"
PROMPT_ARTIFACT_ID = "prompt.helixagent.system"


def open_archive(path: Path = ARCHIVE_PATH) -> SQLiteArchive:
    path.parent.mkdir(parents=True, exist_ok=True)
    return SQLiteArchive(path)


async def get_or_create_description(archive: SQLiteArchive) -> Artifact:
    live = await archive.live_champion(RETRIEVE_DESCRIPTION_ID)
    if live is not None:
        return live
    existing = await archive.by_id(RETRIEVE_DESCRIPTION_ID, version=1)
    if existing is not None:
        return existing.artifact
    seed = build_retrieve_description_artifact()
    archive._store_artifact(seed)  # type: ignore[attr-defined]
    archive._conn.commit()  # type: ignore[attr-defined]
    return seed


def _system_prompt_artifact() -> Artifact:
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

    agent = Agent(
        system_prompt=_system_prompt_artifact(),
        tools=[build_retrieve_tool_with_artifact_description(description)],
        model=AGENT_MODEL,
    )

    judge = SwapAndAgree(PairwiseJudge(model=JUDGE_MODEL))
    eval_source = FixedEvalSet(load_eval_set(EVAL_QUESTIONS_PATH))
    policy = ImproverPolicy(
        schedule=Schedule.MANUAL,
        questions_per_round=QUESTIONS_PER_ROUND,
        promote_threshold_win_rate=0.5,
    )

    # The cost ladder: SPO (cheapest) -> GEPA -> DGM (most thorough). Each
    # method gets MAX_FAILURES_PER_STRATEGY consecutive failures before the
    # chain rotates to the next.
    chain = StrategyChain(
        strategies=[
            SPO(
                proposer_model=PROPOSER_MODEL,
                first_round_model=AGENT_MODEL,
                rounds=1,
            ),
            GEPA(
                proposer_model=PROPOSER_MODEL,
                reflector=Reflection(model=PROPOSER_MODEL),
                agent=agent,
                eval_source=eval_source,
                population_size=GEPA_POPULATION,
                generations=GEPA_GENERATIONS,
            ),
            DGMSearch(
                mutator=BlindLLMMutator(model=PROPOSER_MODEL),
                rounds=1,
            ),
        ],
        max_failures_per_strategy=MAX_FAILURES_PER_STRATEGY,
    )

    improver = OfflineImprover(
        agent=agent,
        target_artifact_id=RETRIEVE_DESCRIPTION_ID,
        signal=judge,
        search=chain,
        archive=archive,
        eval_source=eval_source,
        policy=policy,
        seed_fallback=description,
        improver_id="imp-chain",
    )

    agent.attach_improver(improver)
    print(f"\nStrategyChain configured: {[s.kind.value for s in chain.strategies]}")
    print(f"Max failures per strategy: {MAX_FAILURES_PER_STRATEGY}")

    await improver.start()
    try:
        for r in range(ROUNDS_TO_DRIVE):
            print(f"\n========== Round {r+1}/{ROUNDS_TO_DRIVE} (active: {chain.active_kind}) ==========")
            if chain.all_retired:
                print("All strategies retired; halting.")
                break
            await improver.trigger_round()
    finally:
        await improver.stop()

    print()
    print("=" * 70)
    print(f"Chain status: {chain.status()}")
    s = improver.status
    last = s.last_round_result
    print(f"  rounds completed: {s.rounds_completed}")
    if last is not None:
        print(f"  last round: cand v{last.candidate_version} score={last.candidate_score:.3f}  "
              f"ref score={last.reference_score:.3f}  promoted={last.promoted}")
    top = await archive.best(k=3)
    print(f"\n  Archive top 3:")
    for v in top:
        print(f"    v{v.artifact.version} by {v.artifact.created_by}  score={v.measurement.score:.3f}")
    print(f"\n  Archive: {ARCHIVE_PATH}")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
