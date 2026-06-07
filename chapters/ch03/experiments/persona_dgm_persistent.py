"""DGM evaluated the way its archive intends: one PERSISTENT archive across
batches, so it accumulates and its best ratchets (the archive never discards a
good variant). Re-measures DGM's best after each batch on train and held-out
test, and contrasts with GEPA run fresh each batch (an i.i.d. draw, since GEPA
regenerates its population). The point: DGM should hold or climb across batches
where GEPA oscillates.

Run:
    python chapters/ch03/experiments/persona_dgm_persistent.py
"""
import asyncio
import sqlite3
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from helix.archive import SQLiteArchive
from helix.env import load_env
from helix.eval import FixedEvalSet
from helix.improvement import ImproverPolicy, OfflineImprover, Schedule
from helix.observability import attach_console_renderer
from helix.search.dgm import BlindLLMMutator, DGMSearch
from helix.search.gepa import GEPA
from helix.signals.reflection import Reflection

from agents.travel import SYSTEM_PROMPT_ID, build_travel_agent
from agents.travel_persona import (
    PERSONAS, POLICY_GENESIS_PERSONA, PersonaRubricJudge, build_persona_policy,
    load_persona_tasks, persona_descriptions, persona_eval_set, score_persona,
)
from agents.travel_sim import reconstruct_trip

load_env()

AGENT_MODEL = "claude-sonnet-4-5"
PROPOSER_MODEL = "claude-sonnet-4-6"
BATCHES = 2
ROUNDS_PER_BATCH = 8
GEPA_POP, GEPA_GEN = 4, 2
TOOL_DEFAULTS = {"default_cabin": "economy", "default_rate_code": "Q"}
DESCS = persona_descriptions()
TASKS = Path(__file__).parent.parent / "travel_persona_tasks.json"
TRAIN = load_persona_tasks(TASKS, "train")
TEST = load_persona_tasks(TASKS, "test")
RUNS = Path(__file__).parent.parent / "runs"


def out(s):
    sys.stdout.buffer.write((str(s) + "\n").encode("ascii", "replace")); sys.stdout.flush()


def agent(policy):
    return build_travel_agent(DESCS, system_prompt=build_persona_policy(policy),
                              model=AGENT_MODEL, tool_defaults=TOOL_DEFAULTS)


async def absolute(policy, tasks):
    a = agent(policy); total = 0.0
    for t in tasks:
        _, traj = await a.run(t.request)
        total += score_persona(reconstruct_trip(traj), PERSONAS[t.persona], t.required)[0]
    return total / len(tasks)


async def best_by_absolute(path):
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    contents = [r[0] for r in conn.execute("SELECT content FROM artifacts WHERE artifact_id=?", (SYSTEM_PROMPT_ID,))]
    seen, uniq = set(), []
    for c in contents:
        if c.strip() not in seen:
            seen.add(c.strip()); uniq.append(c)
    scored = [(c, await absolute(c, TRAIN)) for c in uniq]
    best, btr = max(scored, key=lambda x: x[1])
    return len(uniq), btr, await absolute(best, TEST)


def improver(search, archive):
    return OfflineImprover(
        agent=agent(POLICY_GENESIS_PERSONA), target_artifact_id=SYSTEM_PROMPT_ID,
        signal=PersonaRubricJudge(), search=search, archive=archive,
        eval_source=FixedEvalSet(persona_eval_set(TRAIN)),
        policy=ImproverPolicy(schedule=Schedule.MANUAL, questions_per_round=None, promote_threshold_win_rate=0.5),
        seed_fallback=build_persona_policy(POLICY_GENESIS_PERSONA))


async def main_async():
    attach_console_renderer(verbose=False)
    RUNS.mkdir(parents=True, exist_ok=True)

    # DGM: one persistent archive that accumulates across batches.
    dgm_path = RUNS / "persona_dgm_persistent.sqlite"
    for ext in ("", "-wal", "-shm"):
        p = Path(str(dgm_path) + ext)
        if p.exists():
            p.unlink()
    dgm_archive = SQLiteArchive(dgm_path)
    await dgm_archive.put_artifact(build_persona_policy(POLICY_GENESIS_PERSONA))
    dgm_imp = improver(DGMSearch(mutator=BlindLLMMutator(model=PROPOSER_MODEL), rounds=1), dgm_archive)

    out("\n=== DGM (persistent archive) vs GEPA (fresh each batch) on sonnet-4-5 ===\n")
    out(f"{'batch':<7}{'rounds':>7}{'DGM cands':>11}{'DGM train':>11}{'DGM test':>10}{'GEPA train':>12}{'GEPA test':>11}")
    await dgm_imp.start()
    try:
        for b in range(1, BATCHES + 1):
            for _ in range(ROUNDS_PER_BATCH):
                await dgm_imp.trigger_round()
            n, dtr, dte = await best_by_absolute(dgm_path)  # best over the GROWING archive

            # GEPA fresh: a brand new archive and population this batch (i.i.d.).
            gpath = RUNS / f"persona_gepa_fresh_{b}.sqlite"
            for ext in ("", "-wal", "-shm"):
                p = Path(str(gpath) + ext)
                if p.exists():
                    p.unlink()
            garchive = SQLiteArchive(gpath)
            await garchive.put_artifact(build_persona_policy(POLICY_GENESIS_PERSONA))
            gepa = GEPA(proposer_model=PROPOSER_MODEL, reflector=Reflection(model=PROPOSER_MODEL),
                        agent=agent(POLICY_GENESIS_PERSONA), eval_source=FixedEvalSet(persona_eval_set(TRAIN)),
                        population_size=GEPA_POP, generations=GEPA_GEN)
            gimp = improver(gepa, garchive)
            await gimp.start()
            try:
                await gimp.trigger_round()
            finally:
                await gimp.stop()
            _, gtr, gte = await best_by_absolute(gpath)
            out(f"{b:<7}{b * ROUNDS_PER_BATCH:>7}{n:>11}{dtr:>11.2f}{dte:>10.2f}{gtr:>12.2f}{gte:>11.2f}")
    finally:
        await dgm_imp.stop()
    out("\n  DGM's archive persists and accumulates (cands grow); GEPA is fresh each batch.")
    out("  Expect DGM's best to hold or climb; GEPA to be an independent draw each batch.")


if __name__ == "__main__":
    asyncio.run(main_async())
