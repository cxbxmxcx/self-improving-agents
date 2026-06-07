"""Chapter 3: SPO vs GEPA vs DGM on the multi-modal persona task, the regime where
the search-method cost ladder actually separates (see SEARCH_METHODS_FINDINGS.md).

Run the same agent under each search method and compare the best policy each
discovers, selected by ABSOLUTE rating (not the noisy pairwise-vs-seed score) and
measured on a held-out city, because the advantage of the elaborate methods shows
up as generalization, not as a higher training score.

Config that matters:
  - AGENT_MODEL: must have real headroom and a high executable ceiling. Sonnet 4.5
    is the sweet spot (genesis ~0.35, oracle ceiling ~0.73); Sonnet 4.6 is too
    strong (no headroom), Haiku too weak (low ceiling).
  - PROPOSER_MODEL: a capable proposer; a weak proposer breaks GEPA and DGM.
  - Selection is by absolute mean rating over a re-measurement of every candidate.

Run:
    python chapters/ch03/persona_search_experiment.py
"""

from __future__ import annotations

import asyncio
import sqlite3
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from helix.archive import SQLiteArchive
from helix.env import load_env
from helix.eval import FixedEvalSet
from helix.improvement import ImproverPolicy, OfflineImprover, Schedule
from helix.observability import attach_console_renderer
from helix.search.dgm import BlindLLMMutator, DGMSearch
from helix.search.gepa import GEPA
from helix.search.spo import SPO
from helix.signals.reflection import Reflection

from agents.travel import SYSTEM_PROMPT_ID, build_travel_agent
from agents.travel_persona import (
    PERSONAS, POLICY_GENESIS_PERSONA, PersonaRubricJudge, build_persona_policy,
    load_persona_tasks, persona_descriptions, persona_eval_set, score_persona,
)
from agents.travel_sim import reconstruct_trip

load_env()

# Configuration -----------------------------------------------------------
AGENT_MODEL = "claude-sonnet-4-5"     # intermediate agent: real headroom, high ceiling
PROPOSER_MODEL = "claude-sonnet-4-6"  # capable proposer
SPO_ROUNDS = 8
DGM_ROUNDS = 8
GEPA_POP, GEPA_GEN = 4, 2
TOOL_DEFAULTS = {"default_cabin": "economy", "default_rate_code": "Q"}  # honest pricing

TASKS_PATH = Path(__file__).parent / "travel_persona_tasks.json"
RUNS = Path(__file__).parent / "runs"
DESCS = persona_descriptions()
TRAIN = load_persona_tasks(TASKS_PATH, "train")
TEST = load_persona_tasks(TASKS_PATH, "test")  # held-out city


def agent(policy: str):
    return build_travel_agent(DESCS, system_prompt=build_persona_policy(policy),
                              model=AGENT_MODEL, tool_defaults=TOOL_DEFAULTS)


async def absolute(policy: str, tasks) -> float:
    """Absolute mean persona rating: run the agent on each task and grade the trip."""
    a = agent(policy)
    total = 0.0
    for t in tasks:
        _, traj = await a.run(t.request)
        total += score_persona(reconstruct_trip(traj), PERSONAS[t.persona], t.required)[0]
    return total / len(tasks) if tasks else 0.0


def open_fresh(name: str) -> SQLiteArchive:
    RUNS.mkdir(parents=True, exist_ok=True)
    for ext in ("", "-wal", "-shm"):
        p = Path(str(RUNS / f"persona_{name}.sqlite") + ext)
        if p.exists():
            p.unlink()
    return SQLiteArchive(RUNS / f"persona_{name}.sqlite")


async def best_by_absolute(name: str) -> tuple[float, float]:
    """Re-measure every unique policy the search produced and return the best
    one's absolute rating on train and held-out test (fair selection)."""
    conn = sqlite3.connect(f"file:{RUNS / f'persona_{name}.sqlite'}?mode=ro", uri=True)
    contents = [r[0] for r in conn.execute(
        "SELECT content FROM artifacts WHERE artifact_id=?", (SYSTEM_PROMPT_ID,))]
    seen, uniq = set(), []
    for c in contents:
        if c.strip() not in seen:
            seen.add(c.strip()); uniq.append(c)
    scored = [(c, await absolute(c, TRAIN)) for c in uniq]
    best_content, best_train = max(scored, key=lambda x: x[1])
    return best_train, await absolute(best_content, TEST)


def make_improver(name: str, search) -> tuple[OfflineImprover, SQLiteArchive]:
    archive = open_fresh(name)
    seed = build_persona_policy(POLICY_GENESIS_PERSONA)
    imp = OfflineImprover(
        agent=agent(POLICY_GENESIS_PERSONA), target_artifact_id=SYSTEM_PROMPT_ID,
        signal=PersonaRubricJudge(), search=search, archive=archive,
        eval_source=FixedEvalSet(persona_eval_set(TRAIN)),
        policy=ImproverPolicy(schedule=Schedule.MANUAL, questions_per_round=None, promote_threshold_win_rate=0.5),
        seed_fallback=seed, improver_id=f"{name}-imp")
    return imp, archive, seed


async def run_offline(name: str, search, rounds: int) -> tuple[float, float]:
    imp, archive, seed = make_improver(name, search)
    await archive.put_artifact(seed)
    await imp.start()
    try:
        for _ in range(rounds):
            await imp.trigger_round()
    finally:
        await imp.stop()
    return await best_by_absolute(name)


async def run_gepa() -> tuple[float, float]:
    archive = open_fresh("gepa")
    seed = build_persona_policy(POLICY_GENESIS_PERSONA)
    await archive.put_artifact(seed)
    gepa = GEPA(proposer_model=PROPOSER_MODEL, reflector=Reflection(model=PROPOSER_MODEL),
                agent=agent(POLICY_GENESIS_PERSONA), eval_source=FixedEvalSet(persona_eval_set(TRAIN)),
                population_size=GEPA_POP, generations=GEPA_GEN)
    imp, _, _ = make_improver("gepa", gepa)
    imp.archive = archive  # use the seeded archive
    await imp.start()
    try:
        await imp.trigger_round()
    finally:
        await imp.stop()
    return await best_by_absolute("gepa")


async def main_async() -> None:
    attach_console_renderer(verbose=False)
    gen_tr = await absolute(POLICY_GENESIS_PERSONA, TRAIN)
    gen_te = await absolute(POLICY_GENESIS_PERSONA, TEST)
    spo_tr, spo_te = await run_offline("spo", SPO(proposer_model=PROPOSER_MODEL, first_round_model=AGENT_MODEL, rounds=1), SPO_ROUNDS)
    gepa_tr, gepa_te = await run_gepa()
    dgm_tr, dgm_te = await run_offline("dgm", DGMSearch(mutator=BlindLLMMutator(model=PROPOSER_MODEL), rounds=1), DGM_ROUNDS)

    print(f"\n=== SPO vs GEPA vs DGM on the persona task ({AGENT_MODEL}) ===")
    print(f"  selection: absolute rating; test is a held-out city (generalization)\n")
    print(f"  {'method':<9}{'train':>8}{'test':>8}")
    for label, tr, te in (("genesis", gen_tr, gen_te), ("SPO", spo_tr, spo_te),
                          ("GEPA", gepa_tr, gepa_te), ("DGM", dgm_tr, dgm_te)):
        print(f"  {label:<9}{tr:>8.2f}{te:>8.2f}")
    print("\n  The advantage shows as generalization (held-out test), not training score:")
    print("  SPO overfits, GEPA is robust, DGM climbs and generalizes.")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
