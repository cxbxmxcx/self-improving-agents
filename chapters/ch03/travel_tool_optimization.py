"""Chapter 3 (Path B) — grounded, multi-tool description search on the task agent.

Three tool descriptions improve at once: one GEPA improver per searchable tool
(search_flights, search_hotels, search_activities), the §16.1 multi-improver
pattern. The tools are independent, so each improver targets its own
TOOL_DESCRIPTION artifact while sharing the agent and the archive.

Two things make the search actually move the score:
  - The mutator is grounded in each tool's real parameter schema
    (grounded_mutation_prompt), so it stops inventing parameters that do not
    exist and starts naming the ones that do (nonstop, max_price, min_rating).
  - The signal is TravelTaskJudge, deterministic ground truth.

The script prints whole-agent task success before and after so the lift from
better descriptions is visible.

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
    load_travel_eval_set,
    load_travel_scenarios,
)
from agents.travel_sim import SEARCHABLE_TOOL_DESCRIPTION_IDS, reconstruct_trip

load_env()

# Configuration -----------------------------------------------------------
AGENT_MODEL = "claude-haiku-4-5"
PROPOSER_MODEL = "claude-sonnet-4-6"
ROUNDS_TO_DRIVE = 2
GEPA_POPULATION = 3
GEPA_GENERATIONS = 2
QUESTIONS_PER_ROUND: int | None = None  # None = all scenarios

ARCHIVE_PATH = Path(__file__).parent / "runs" / "travel_archive.sqlite"
SCENARIOS_PATH = Path(__file__).parent / "travel_scenarios.json"


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


async def whole_agent_success(descriptions, scenarios, model) -> float:
    """Average task success of the agent built with this description set."""
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
    scenarios = load_travel_scenarios(SCENARIOS_PATH)

    baseline = await whole_agent_success(descriptions, scenarios, AGENT_MODEL)
    print(f"\nBASELINE whole-agent task success (genesis descriptions): {baseline:.3f}\n")

    signal = TravelTaskJudge()
    eval_source = FixedEvalSet(load_travel_eval_set(SCENARIOS_PATH))
    policy = ImproverPolicy(
        schedule=Schedule.MANUAL,
        questions_per_round=QUESTIONS_PER_ROUND,
        promote_threshold_win_rate=0.5,
    )

    # One grounded GEPA improver per searchable tool description.
    improvers = []
    for tool_name, desc_id in SEARCHABLE_TOOL_DESCRIPTION_IDS.items():
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

    # Assemble the best description per tool and re-measure the whole agent.
    print()
    print("=" * 70)
    ranked = await archive.best(k=50, signal_id=signal.signal_id)
    best_desc = dict(descriptions)
    for _, desc_id, _ in improvers:
        for v in ranked:
            if v.artifact.id == desc_id:
                best_desc[desc_id] = v.artifact
                break

    after = await whole_agent_success(best_desc, scenarios, AGENT_MODEL)
    print(f"AFTER  whole-agent task success (best descriptions): {after:.3f}")
    print(f"BEFORE whole-agent task success (genesis):           {baseline:.3f}")
    print("-" * 70)
    for tool_name, desc_id, _ in improvers:
        a = best_desc[desc_id]
        print(f"\n{tool_name}  v{a.version} ({a.created_by}):")
        print(f"  {a.content[:160]}")
    print(f"\narchive: {ARCHIVE_PATH}")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
