"""Chapter 3 (Path B, DGM variant) — archive-evolutionary tool-description search.

The companion to travel_tool_optimization.py, identical in every way except the
search method: where that script evolves a GEPA population, this one runs the
framework DGMSearch (the Darwin Goedel Machine pattern) over an archive. Both
share the same task agent, the same isolated per-tool scenarios, the same
deterministic TravelTaskJudge, and the same grounded, feedback-conditioned
mutator, so the only variable is GEPA-population versus DGM-archive search.

Two wiring differences from the GEPA script, both forced by how DGM works:
  - Per-tool archives. DGM samples its next seed from the whole archive with
    quality-diversity weighting, and archive.sample() has no artifact-id filter,
    so a shared archive would let the hotel improver fork from a flight
    candidate. One archive per tool keeps each search within its own lineage.
  - Rounds, not generations. Each trigger_round() proposes and evaluates exactly
    one DGM candidate (sample a seed, mutate, score it). Driving DGM_ROUNDS
    rounds per tool gives the archive time to grow and the quality-diversity
    sampler something to explore, and matches the ~9-candidate budget the GEPA
    run spends per tool (population 3 x 3 generations).

Run:
    python chapters/ch03/travel_tool_optimization_dgm.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from helix.archive import SQLiteArchive
from helix.artifact import Artifact
from helix.env import load_env
from helix.eval import FixedEvalSet
from helix.improvement import ImproverPolicy, OfflineImprover, Schedule
from helix.observability import attach_console_renderer
from helix.search.dgm import BlindLLMMutator, DGMSearch

from agents.travel import (
    TravelTaskJudge,
    build_travel_agent,
    genesis_descriptions,
    grounded_mutation_prompt,
    isolated_eval_set,
    load_isolated_scenarios,
)
from agents.travel_sim import SEARCHABLE_TOOL_DESCRIPTION_IDS, reconstruct_trip

load_env()

# Configuration -----------------------------------------------------------
AGENT_MODEL = "claude-sonnet-4-6"
PROPOSER_MODEL = "claude-sonnet-4-6"
# Each round proposes and scores one DGM candidate; nine rounds per tool matches
# the GEPA run's per-tool candidate budget (population 3 x 3 generations).
DGM_ROUNDS = 9
QUESTIONS_PER_ROUND: int | None = None  # None = all isolated scenarios for the tool

RUNS_DIR = Path(__file__).parent / "runs"
SCENARIOS_PATH = Path(__file__).parent / "travel_scenarios_isolated.json"


def open_archive(tool_name: str) -> SQLiteArchive:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    return SQLiteArchive(RUNS_DIR / f"travel_dgm_{tool_name}.sqlite")


async def tool_success(descriptions, scenarios, model) -> float:
    """Absolute task success on one tool's isolated scenarios (see GEPA script)."""
    agent = build_travel_agent(descriptions, model=model)
    total = 0.0
    for sc in scenarios:
        _, trajectory = await agent.run(sc.request)
        total += sc.score(reconstruct_trip(trajectory))
    return total / len(scenarios) if scenarios else 0.0


async def main_async() -> None:
    attach_console_renderer(verbose=True)

    descriptions = genesis_descriptions()
    isolated = load_isolated_scenarios(SCENARIOS_PATH)  # {tool_name: [scenario, ...]}
    signal = TravelTaskJudge()
    policy = ImproverPolicy(
        schedule=Schedule.MANUAL,
        questions_per_round=QUESTIONS_PER_ROUND,
        promote_threshold_win_rate=0.5,
    )

    # Per-tool baseline on the genesis description.
    baselines: dict[str, float] = {}
    for tool_name in SEARCHABLE_TOOL_DESCRIPTION_IDS:
        baselines[tool_name] = await tool_success(descriptions, isolated[tool_name], AGENT_MODEL)
    print("\nBASELINE per-tool task success (genesis descriptions):")
    for tool_name in SEARCHABLE_TOOL_DESCRIPTION_IDS:
        print(f"  {tool_name:<18} {baselines[tool_name]:.3f}")
    print()

    # One DGM improver per searchable tool, each on its own archive.
    archives: dict[str, SQLiteArchive] = {}
    improvers = []
    for tool_name, desc_id in SEARCHABLE_TOOL_DESCRIPTION_IDS.items():
        archive = open_archive(tool_name)
        await archive.put_artifact(descriptions[desc_id])  # seed genesis v1
        archives[tool_name] = archive
        dgm = DGMSearch(
            mutator=BlindLLMMutator(model=PROPOSER_MODEL, prompt=grounded_mutation_prompt(tool_name)),
            rounds=1,  # one candidate per trigger_round; we drive DGM_ROUNDS rounds
        )
        imp = OfflineImprover(
            agent=build_travel_agent(descriptions, model=AGENT_MODEL),
            target_artifact_id=desc_id,
            signal=signal,
            search=dgm,
            archive=archive,
            eval_source=FixedEvalSet(isolated_eval_set(isolated[tool_name])),
            policy=policy,
            seed_fallback=descriptions[desc_id],
            improver_id=f"dgm-{tool_name}",
        )
        improvers.append((tool_name, desc_id, imp))

    for _, _, imp in improvers:
        await imp.start()
    try:
        for _, _, imp in improvers:
            for _ in range(DGM_ROUNDS):
                await imp.trigger_round()
    finally:
        for _, _, imp in improvers:
            await imp.stop()

    # Best description per tool, re-measured absolutely on its isolated set.
    print()
    print("=" * 70)
    best_desc: dict[str, Artifact] = {}
    for tool_name, desc_id, _ in improvers:
        ranked = await archives[tool_name].best(k=200)
        best_desc[tool_name] = ranked[0].artifact if ranked else descriptions[desc_id]

    print("PER-TOOL task success  (isolated, deterministic, absolute) -- DGM")
    print(f"  {'tool':<18} {'BEFORE':>8} {'AFTER':>8}")
    for tool_name, desc_id, _ in improvers:
        one_swap = dict(descriptions)
        one_swap[desc_id] = best_desc[tool_name]
        after = await tool_success(one_swap, isolated[tool_name], AGENT_MODEL)
        print(f"  {tool_name:<18} {baselines[tool_name]:>8.3f} {after:>8.3f}")
    print("-" * 70)
    for tool_name, desc_id, _ in improvers:
        a = best_desc[tool_name]
        print(f"\n{tool_name}  v{a.version} ({a.created_by}):")
        print(f"  {a.content[:200]}")
    print(f"\narchives: {RUNS_DIR}/travel_dgm_<tool>.sqlite")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
