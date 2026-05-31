"""Chapter 3 (Path B) — framework search over a travel tool description.

The travel task agent has several tools, so the natural-language description of
`search_flights` is what tells the model to pass `nonstop` and `max_price`. The
genesis description ("Search for flights.") omits them, so the agent misses
constraints and books the wrong flight. Here the framework's SPO, GEPA, and DGM
evolve that description, graded by TravelTaskJudge: a deterministic ground-truth
signal that reconstructs the booked trip from each trajectory and prefers the one
that better satisfies the scenario constraints.

This is the §3.4 multi-improver pattern on the task agent. Each OfflineImprover
clones the same agent via with_artifacts to test its candidate description; the
shared archive arbitrates by score regardless of which method produced the
winner.

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
from helix.search.dgm import BlindLLMMutator, DGMSearch
from helix.search.gepa import GEPA
from helix.search.spo import SPO
from helix.signals.reflection import Reflection

from agents.travel import (
    TravelTaskJudge,
    build_travel_agent,
    genesis_descriptions,
    load_travel_eval_set,
)

load_env()

# Configuration -----------------------------------------------------------
AGENT_MODEL = "claude-haiku-4-5"
PROPOSER_MODEL = "claude-sonnet-4-6"
TARGET_DESCRIPTION_ID = "prompt.tool.search_flights.description"
ROUNDS_TO_DRIVE = 3
GEPA_POPULATION = 4
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
        if existing is not None:
            out[desc_id] = existing.artifact
            continue
        await archive.put_artifact(seed)
        out[desc_id] = seed
    return out


async def main_async() -> None:
    attach_console_renderer(verbose=True)

    archive = open_archive()
    descriptions = await get_or_create_descriptions(archive)

    # One travel-agent definition. Both improvers clone it via
    # agent.with_artifacts({TARGET_DESCRIPTION_ID: candidate}) to test a
    # candidate search_flights description; the other descriptions stay fixed.
    agent = build_travel_agent(descriptions, model=AGENT_MODEL)

    signal = TravelTaskJudge()
    eval_source = FixedEvalSet(load_travel_eval_set(SCENARIOS_PATH))
    policy = ImproverPolicy(
        schedule=Schedule.MANUAL,
        questions_per_round=QUESTIONS_PER_ROUND,
        promote_threshold_win_rate=0.5,
    )

    spo_improver = OfflineImprover(
        agent=agent,
        target_artifact_id=TARGET_DESCRIPTION_ID,
        signal=signal,
        search=SPO(proposer_model=PROPOSER_MODEL, first_round_model=AGENT_MODEL, rounds=1),
        archive=archive,
        eval_source=eval_source,
        policy=policy,
        seed_fallback=descriptions[TARGET_DESCRIPTION_ID],
        improver_id="imp-travel-spo",
    )
    gepa_improver = OfflineImprover(
        agent=agent,
        target_artifact_id=TARGET_DESCRIPTION_ID,
        signal=signal,
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
        seed_fallback=descriptions[TARGET_DESCRIPTION_ID],
        improver_id="imp-travel-gepa",
    )
    dgm_improver = OfflineImprover(
        agent=agent,
        target_artifact_id=TARGET_DESCRIPTION_ID,
        signal=signal,
        search=DGMSearch(mutator=BlindLLMMutator(model=PROPOSER_MODEL), rounds=1),
        archive=archive,
        eval_source=eval_source,
        policy=policy,
        seed_fallback=descriptions[TARGET_DESCRIPTION_ID],
        improver_id="imp-travel-dgm",
    )

    improvers = (spo_improver, gepa_improver, dgm_improver)
    for imp in improvers:
        agent.attach_improver(imp)
        await imp.start()

    try:
        for _ in range(ROUNDS_TO_DRIVE):
            for imp in improvers:
                await imp.trigger_round()
    finally:
        for imp in improvers:
            await imp.stop()

    print()
    print("=" * 70)
    best = await archive.best(k=1, signal_id=signal.signal_id)
    if best:
        v = best[0]
        print(f"Best search_flights description: v{v.artifact.version} "
              f"(task success {v.measurement.score:.2f})" if v.measurement else "")
        print("-" * 70)
        print(v.artifact.content)
    print(f"\narchive: {ARCHIVE_PATH}")
    print("Promote a winner with the dashboard to make the better description live.")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
