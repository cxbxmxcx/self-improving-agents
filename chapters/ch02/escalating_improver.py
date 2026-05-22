"""Single Improver with a StrategyChain that escalates from SPO to GEPA.

Demonstrates the platform's StrategyChain Search: an ordered list of Search
methods with a failure budget per method. Cheap methods (SPO) try first;
expensive methods (GEPA) escalate only when the cheap methods stop producing
wins. Promotion resets the active strategy's failure count; failures past
the budget rotate to the next strategy.

Pedagogical point: this is the cost-aware version of multi-method search.
The user sets the ordering and budget; the framework arbitrates. Compare
with dual_improver.py, which runs all methods always.

Re-run the script to drive more rounds against whatever strategy is
currently active in the chain.
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
from helix.search import StrategyChain
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
QUESTIONS_PER_ROUND: int | None = None
ROUNDS_TO_DRIVE = 5
MAX_FAILURES_PER_STRATEGY = 2  # rotate after 2 consecutive failures (gives each method room to recover)

# GEPA configuration: population=4, generations=2 (8 candidates per GEPA round).
GEPA_POPULATION = 4
GEPA_GENERATIONS = 2

EVAL_QUESTIONS_PATH = Path(__file__).parent / "eval_questions_v2.json"


async def build_agent_with_prompt(prompt, *, max_iterations: int = 10, max_tool_calls: int = 20) -> Agent:
    return Agent(
        system_prompt=prompt,
        tools=[build_retrieve_tool()],
        model=AGENT_MODEL,
        max_iterations=max_iterations,
        max_tool_calls=max_tool_calls,
    )


async def main_async() -> None:
    attach_console_renderer(verbose=True)

    archive = open_archive()
    seed = await get_or_create_seed(archive)

    judge = SwapAndAgree(PairwiseJudge(model=JUDGE_MODEL))
    eval_source = FixedEvalSet(load_eval_set(EVAL_QUESTIONS_PATH))
    policy = ImproverPolicy(
        schedule=Schedule.MANUAL,
        mode=ImproverMode.OFFLINE,
        questions_per_round=QUESTIONS_PER_ROUND,
        promote_threshold_win_rate=0.5,
    )

    # The StrategyChain: SPO first (cheap), GEPA second (expensive). One
    # failure per strategy is the budget; rotate to the next on overflow.
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
                agent_factory=build_agent_with_prompt,
                eval_source=eval_source,
                population_size=GEPA_POPULATION,
                generations=GEPA_GENERATIONS,
            ),
        ],
        max_failures_per_strategy=MAX_FAILURES_PER_STRATEGY,
    )

    improver = Improver(
        target_artifact_id=PROMPT_ARTIFACT_ID,
        signal=judge,
        search=chain,
        archive=archive,
        eval_source=eval_source,
        build_agent_with_prompt=build_agent_with_prompt,
        policy=policy,
        seed_fallback=seed,
        improver_id="imp-chain",
    )

    agent = await build_agent_with_prompt(seed)
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
