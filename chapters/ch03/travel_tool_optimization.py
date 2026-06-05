"""Chapter 3 (Path B) — grounded, multi-tool description search on the task agent.

Three tool descriptions improve at once: one GEPA improver per searchable tool
(search_flights, search_hotels, search_activities), the §16.1 multi-improver
pattern. The tools are independent, so each improver targets its own
TOOL_DESCRIPTION artifact while sharing the archive.

Each tool hides one NON-INFERABLE gotcha: information a capable agent cannot get
from the request or the search results, only from the description.
  - search_flights: the fare is multiplied by `cabin`, which defaults to first
    class (3x). A budget request fails unless the description says to pass
    cabin='economy'.
  - search_hotels: the nightly rate is multiplied by `rate_plan`, which defaults
    to flexible (1.5x). A four-star hotel never fits a tight nightly budget
    unless the description says to pass rate_plan='advance'.
  - search_activities: `city` is the destination airport code, so "New York"
    returns nothing unless the description reveals the code (JFK).

Three things make the lift attributable and visible:
  - The mutator is grounded in each tool's real parameter schema
    (grounded_mutation_prompt), so it names the parameters that exist.
  - Each improver is evaluated on ISOLATED scenarios that exercise only its tool,
    so a flight-description candidate is scored purely on flight tasks. This
    removes the conjunction across tools and the variance of judging whole trips.
  - The signal is TravelTaskJudge over deterministic ground truth, and the
    before/after is reported per tool as an absolute task-success fraction.

The agent is a capable model that completes the bookings; because the gotchas are
non-inferable it still fails with the vague genesis descriptions, so the search
has real headroom to recover by making each gotcha explicit.

Run:
    python chapters/ch03/travel_tool_optimization.py
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
from helix.search.gepa import GEPA
from helix.signals.reflection import Reflection

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
# The agent must be capable enough to complete multi-step bookings; a weaker
# model stops after searching, which masks any description effect. The gotchas
# are non-inferable, so a capable agent still fails without a good description.
AGENT_MODEL = "claude-sonnet-4-6"
PROPOSER_MODEL = "claude-sonnet-4-6"
ROUNDS_TO_DRIVE = 2
GEPA_POPULATION = 3
GEPA_GENERATIONS = 2
QUESTIONS_PER_ROUND: int | None = None  # None = all scenarios

ARCHIVE_PATH = Path(__file__).parent / "runs" / "travel_archive.sqlite"
SCENARIOS_PATH = Path(__file__).parent / "travel_scenarios_isolated.json"


def open_archive(path: Path = ARCHIVE_PATH) -> SQLiteArchive:
    path.parent.mkdir(parents=True, exist_ok=True)
    return SQLiteArchive(path)


async def get_or_create_descriptions(archive: SQLiteArchive) -> dict[str, Artifact]:
    """Resolve each searchable tool description to its live champion, seeding the
    genesis (vague) version on first run via the public put_artifact API."""
    out: dict[str, Artifact] = {}
    for desc_id, seed in genesis_descriptions().items():
        live = await archive.live_champion(desc_id)
        if live is not None:
            out[desc_id] = live
            continue
        existing = await archive.by_id(desc_id, version=1)
        out[desc_id] = existing.artifact if existing is not None else seed
        if existing is None:
            await archive.put_artifact(seed)
    return out


async def tool_success(descriptions, scenarios, model) -> float:
    """Absolute task success on one tool's isolated scenarios.

    The agent is built with `descriptions` (one tool overridden, the rest at
    genesis) and run on that tool's scenarios; each booked trip is scored
    deterministically against its constraints. Because the scenarios touch only
    one tool, the number is attributable to that tool's description alone.
    """
    agent = build_travel_agent(descriptions, model=model)
    total = 0.0
    for sc in scenarios:
        _, trajectory = await agent.run(sc.request)
        total += sc.score(reconstruct_trip(trajectory))
    return total / len(scenarios) if scenarios else 0.0


async def main_async() -> None:
    attach_console_renderer(verbose=True)

    archive = open_archive()
    descriptions = await get_or_create_descriptions(archive)
    isolated = load_isolated_scenarios(SCENARIOS_PATH)  # {tool_name: [scenario, ...]}

    signal = TravelTaskJudge()
    policy = ImproverPolicy(
        schedule=Schedule.MANUAL,
        questions_per_round=QUESTIONS_PER_ROUND,
        promote_threshold_win_rate=0.5,
    )

    # Per-tool baseline: each tool description scored on its own scenarios only.
    baselines: dict[str, float] = {}
    for tool_name, desc_id in SEARCHABLE_TOOL_DESCRIPTION_IDS.items():
        baselines[tool_name] = await tool_success(descriptions, isolated[tool_name], AGENT_MODEL)
    print("\nBASELINE per-tool task success (genesis descriptions):")
    for tool_name in SEARCHABLE_TOOL_DESCRIPTION_IDS:
        print(f"  {tool_name:<18} {baselines[tool_name]:.3f}")
    print()

    # One grounded GEPA improver per searchable tool, each on its isolated set.
    improvers = []
    for tool_name, desc_id in SEARCHABLE_TOOL_DESCRIPTION_IDS.items():
        eval_source = FixedEvalSet(isolated_eval_set(isolated[tool_name]))
        gepa = GEPA(
            proposer_model=PROPOSER_MODEL,
            reflector=Reflection(model=PROPOSER_MODEL),
            agent=build_travel_agent(descriptions, model=AGENT_MODEL),
            eval_source=eval_source,
            population_size=GEPA_POPULATION,
            generations=GEPA_GENERATIONS,
            mutation_prompt=grounded_mutation_prompt(tool_name),
            crossover_prompt=grounded_mutation_prompt(tool_name),
        )
        imp = OfflineImprover(
            agent=build_travel_agent(descriptions, model=AGENT_MODEL),
            target_artifact_id=desc_id,
            signal=signal,
            search=gepa,
            archive=archive,
            eval_source=eval_source,
            policy=policy,
            seed_fallback=descriptions[desc_id],
            improver_id=f"imp-{tool_name}",
        )
        improvers.append((tool_name, desc_id, imp))

    for _, _, imp in improvers:
        await imp.start()
    try:
        for _ in range(ROUNDS_TO_DRIVE):
            for _, _, imp in improvers:
                await imp.trigger_round()
    finally:
        for _, _, imp in improvers:
            await imp.stop()

    # Best description per tool, re-measured absolutely on its isolated set.
    # Rank by raw score across every recorded version (the score field already
    # holds each candidate's absolute task-success fraction); for each tool the
    # first match is its highest-scoring version, the genesis seed included.
    print()
    print("=" * 70)
    ranked = await archive.best(k=200)
    best_desc = dict(descriptions)
    for _, desc_id, _ in improvers:
        for v in ranked:
            if v.artifact.id == desc_id:
                best_desc[desc_id] = v.artifact
                break

    print("PER-TOOL task success  (isolated, deterministic, absolute)")
    print(f"  {'tool':<18} {'BEFORE':>8} {'AFTER':>8}")
    for tool_name, desc_id, _ in improvers:
        # Score the winner against this tool's scenarios with only its description swapped.
        one_swap = dict(descriptions)
        one_swap[desc_id] = best_desc[desc_id]
        after = await tool_success(one_swap, isolated[tool_name], AGENT_MODEL)
        print(f"  {tool_name:<18} {baselines[tool_name]:>8.3f} {after:>8.3f}")
    print("-" * 70)
    for tool_name, desc_id, _ in improvers:
        a = best_desc[desc_id]
        print(f"\n{tool_name}  v{a.version} ({a.created_by}):")
        print(f"  {a.content[:200]}")
    print(f"\narchive: {ARCHIVE_PATH}")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
