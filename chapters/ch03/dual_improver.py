"""Chapter 3 §3.4 — three search methods on one artifact (parallel).

Demonstrates the multi-improver pattern (SPEC §16.1): an SPO improver, a
GEPA improver, and a DGM improver all target the SAME artifact — the
retrieve tool's description — and share one Archive. The script drives
them in round-robin alternation. The Archive arbitrates by score
regardless of which Search produced each candidate.

The pedagogical point: three different search strategies (hill-climb,
population-evolution, archive-evolution) compose without coordinating.
Each writes candidates to the shared archive; `archive.best()` returns
the overall winner across all three. That winner is not yet what the
running agent serves — promotion is a separate, human-gated step (the
Ch 2 deploy gate).

(The filename predates the chapter's expansion from two methods to three;
it now runs SPO + GEPA + DGM.)

Re-run the script to drive more rounds.

Cost: ~$2 with the default knobs.
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
from helix.artifact import Artifact, ArtifactKind, genesis
from helix.env import load_env
from helix.eval import FixedEvalSet, load_eval_set
from helix.improvement import OfflineImprover, ImproverPolicy, Schedule
from helix.observability import attach_console_renderer
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
TOTAL_ROUNDS = 6  # 2 each of SPO / GEPA / DGM in round-robin

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
        kind=ArtifactKind.PROMPT,
        content=SYSTEM_PROMPT_V0,
        created_by="human",
    )


async def main_async() -> None:
    attach_console_renderer(verbose=True)

    archive = open_archive()
    description = await get_or_create_description(archive)

    # One agent definition. All three improvers clone it via
    # agent.with_artifacts({RETRIEVE_DESCRIPTION_ID: candidate}) to test
    # their candidate descriptions.
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

    spo_improver = OfflineImprover(
        agent=agent,
        target_artifact_id=RETRIEVE_DESCRIPTION_ID,
        signal=judge,
        search=SPO(proposer_model=PROPOSER_MODEL, first_round_model=AGENT_MODEL, rounds=1),
        archive=archive,
        eval_source=eval_source,
        policy=policy,
        seed_fallback=description,
        improver_id="imp-spo",
    )

    gepa_improver = OfflineImprover(
        agent=agent,
        target_artifact_id=RETRIEVE_DESCRIPTION_ID,
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
        seed_fallback=description,
        improver_id="imp-gepa",
    )

    dgm_improver = OfflineImprover(
        agent=agent,
        target_artifact_id=RETRIEVE_DESCRIPTION_ID,
        signal=judge,
        search=DGMSearch(
            mutator=BlindLLMMutator(model=PROPOSER_MODEL),
            rounds=1,
        ),
        archive=archive,
        eval_source=eval_source,
        policy=policy,
        seed_fallback=description,
        improver_id="imp-dgm",
    )

    for imp in (spo_improver, gepa_improver, dgm_improver):
        agent.attach_improver(imp)
    print(f"\nAttached {len(agent.improvers)} improvers to agent (SPO + GEPA + DGM)")

    improvers = [spo_improver, gepa_improver, dgm_improver]
    for imp in improvers:
        await imp.start()
    try:
        # Round-robin: SPO, GEPA, DGM, SPO, GEPA, DGM, ...
        for r in range(TOTAL_ROUNDS):
            active = improvers[r % len(improvers)]
            print(f"\n========== Round {r+1}/{TOTAL_ROUNDS}: {active.improver_id} ==========")
            await active.trigger_round()
    finally:
        for imp in improvers:
            await imp.stop()

    print()
    print("=" * 70)
    print("All three improvers stopped.")
    for imp in improvers:
        s = imp.status
        last = s.last_round_result
        print(f"\n  {s.improver_id}: rounds={s.rounds_completed}")
        if last is not None:
            cscore = f"{last.candidate_score:.3f}" if last.candidate_score is not None else "n/a"
            print(f"    last round: cand v{last.candidate_version} score={cscore}  promoted={last.promoted}")
    top = await archive.best(k=3, signal_id=judge.signal_id)
    print(f"\n  Archive top 3 (overall, across all three search methods):")
    for v in top:
        sc = f"{v.measurement.score:.3f}" if v.measurement and v.measurement.score is not None else "n/a"
        print(f"    v{v.artifact.version} by {v.artifact.created_by}  score={sc}")
    print(f"\n  Archive: {ARCHIVE_PATH}")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
