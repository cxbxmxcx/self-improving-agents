"""Chapter 3 §3.4 (Path B) — an escalating strategy chain on the travel agent.

The framework's StrategyChain runs SPO until it stalls, rotates to GEPA, then to
DGM: a cost ladder from cheap to thorough that retires a method once it plateaus.
Here it evolves the `search_flights` description, graded by TravelTaskJudge. Same
ground-truth signal and task agent as travel_tool_optimization.py; the difference
is sequential escalation instead of three improvers in parallel.

Run:
    python chapters/ch03/escalating_travel.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from helix.env import load_env
from helix.eval import FixedEvalSet
from helix.improvement import ImproverPolicy, OfflineImprover, Schedule
from helix.observability import attach_console_renderer
from helix.search.dgm import BlindLLMMutator, DGMSearch
from helix.search.gepa import GEPA
from helix.search.spo import SPO
from helix.search.strategy_chain import StrategyChain
from helix.signals.reflection import Reflection

from agents.travel import TravelTaskJudge, build_travel_agent, load_travel_eval_set
from chapters.ch03.travel_tool_optimization import (
    SCENARIOS_PATH,
    TARGET_DESCRIPTION_ID,
    get_or_create_descriptions,
    open_archive,
)

load_env()

# Configuration -----------------------------------------------------------
AGENT_MODEL = "claude-haiku-4-5"
PROPOSER_MODEL = "claude-sonnet-4-6"
ROUNDS_TO_DRIVE = 5
GEPA_POPULATION = 4
GEPA_GENERATIONS = 2
MAX_FAILURES_PER_STRATEGY = 2


async def main_async() -> None:
    attach_console_renderer(verbose=True)

    archive = open_archive()
    descriptions = await get_or_create_descriptions(archive)
    agent = build_travel_agent(descriptions, model=AGENT_MODEL)

    signal = TravelTaskJudge()
    eval_source = FixedEvalSet(load_travel_eval_set(SCENARIOS_PATH))
    policy = ImproverPolicy(schedule=Schedule.MANUAL, promote_threshold_win_rate=0.5)

    # Cost ladder: SPO (cheapest) -> GEPA -> DGM (most thorough). The chain
    # rotates after MAX_FAILURES_PER_STRATEGY consecutive non-improving rounds.
    chain = StrategyChain(
        strategies=[
            SPO(proposer_model=PROPOSER_MODEL, first_round_model=AGENT_MODEL, rounds=1),
            GEPA(
                proposer_model=PROPOSER_MODEL,
                reflector=Reflection(model=PROPOSER_MODEL),
                agent=agent,
                eval_source=eval_source,
                population_size=GEPA_POPULATION,
                generations=GEPA_GENERATIONS,
            ),
            DGMSearch(mutator=BlindLLMMutator(model=PROPOSER_MODEL), rounds=1),
        ],
        max_failures_per_strategy=MAX_FAILURES_PER_STRATEGY,
    )

    improver = OfflineImprover(
        agent=agent,
        target_artifact_id=TARGET_DESCRIPTION_ID,
        signal=signal,
        search=chain,
        archive=archive,
        eval_source=eval_source,
        policy=policy,
        seed_fallback=descriptions[TARGET_DESCRIPTION_ID],
        improver_id="imp-travel-chain",
    )
    agent.attach_improver(improver)
    print(f"\nStrategyChain: {[s.kind.value for s in chain.strategies]}")

    await improver.start()
    try:
        for r in range(ROUNDS_TO_DRIVE):
            print(f"\n===== round {r + 1}/{ROUNDS_TO_DRIVE} (active: {chain.active_kind}) =====")
            if chain.all_retired:
                print("All strategies retired; halting.")
                break
            await improver.trigger_round()
    finally:
        await improver.stop()

    print()
    print("=" * 70)
    print(f"chain status: {chain.status()}")
    top = await archive.best(k=3, signal_id=signal.signal_id)
    for v in top:
        s = f"{v.measurement.score:.2f}" if v.measurement else "n/a"
        print(f"  v{v.artifact.version} by {v.artifact.created_by}  task success {s}")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
