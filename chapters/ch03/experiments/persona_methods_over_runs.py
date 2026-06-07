"""The book table: SPO, GEPA, and DGM compared over two successive runs, each in
its natural mode. SPO and GEPA start fresh each run (independent draws, since they
do not carry state). DGM keeps ONE persistent archive that accumulates across runs,
which is its defining feature: its best is a ratchet that holds or climbs.

For each run, every method's best policy is selected by ABSOLUTE rating and scored
on train and a held-out city. The expectation: SPO overfits (test below train),
GEPA generalizes but is an independent draw each run, DGM holds or climbs across
runs because the archive never discards a good variant.

Run:
    python chapters/ch03/experiments/persona_methods_over_runs.py
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
from helix.search.spo import SPO
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
RUNS_N = 2
ROUNDS = 8
GEPA_POP, GEPA_GEN = 4, 2
SAMPLES = 4        # rollouts averaged per candidate per task to denoise selection/scoring
TOPK = 4           # finalists from the single-sample first pass that get multi-sampled
CONCURRENCY = 4    # bounded parallel rollouts during re-measurement (search stays sequential)
SEM = asyncio.Semaphore(CONCURRENCY)
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


async def _rollout(policy, task):
    """One agent rollout on one task, returning its persona rating. Bounded by the
    semaphore so multi-sampling parallelizes without overrunning rate limits."""
    async with SEM:
        _, traj = await agent(policy).run(task.request)
    return score_persona(reconstruct_trip(traj), PERSONAS[task.persona], task.required)[0]


async def absolute(policy, tasks, samples=1):
    """Mean persona rating, averaging `samples` rollouts per task to denoise. Each
    task's samples are averaged first, then averaged across tasks (equal task weight)."""
    coros, idxs = [], []
    for i, t in enumerate(tasks):
        for _ in range(samples):
            coros.append(_rollout(policy, t)); idxs.append(i)
    results = await asyncio.gather(*coros)
    by_task = {}
    for score, i in zip(results, idxs):
        by_task.setdefault(i, []).append(score)
    means = [sum(v) / len(v) for v in by_task.values()]
    return sum(means) / len(means)


async def best_by_absolute(path):
    """Two-stage denoised selection. Stage 1 ranks every unique candidate on a single
    cheap rollout; stage 2 multi-samples only the TOPK finalists, so the wobbly
    single-sample best becomes a stable estimate without re-measuring all at depth."""
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    contents = [r[0] for r in conn.execute("SELECT content FROM artifacts WHERE artifact_id=?", (SYSTEM_PROMPT_ID,))]
    seen, uniq = set(), []
    for c in contents:
        if c.strip() not in seen:
            seen.add(c.strip()); uniq.append(c)
    if len(uniq) > TOPK:
        first = [(c, await absolute(c, TRAIN, 1)) for c in uniq]
        first.sort(key=lambda x: x[1], reverse=True)
        finalists = [c for c, _ in first[:TOPK]]
    else:
        finalists = uniq
    scored = [(c, await absolute(c, TRAIN, SAMPLES)) for c in finalists]
    best, btr = max(scored, key=lambda x: x[1])
    return btr, await absolute(best, TEST, SAMPLES)


def reset(path):
    for ext in ("", "-wal", "-shm"):
        p = Path(str(path) + ext)
        if p.exists():
            p.unlink()


def improver(search, archive):
    return OfflineImprover(
        agent=agent(POLICY_GENESIS_PERSONA), target_artifact_id=SYSTEM_PROMPT_ID,
        signal=PersonaRubricJudge(), search=search, archive=archive,
        eval_source=FixedEvalSet(persona_eval_set(TRAIN)),
        policy=ImproverPolicy(schedule=Schedule.MANUAL, questions_per_round=None, promote_threshold_win_rate=0.5),
        seed_fallback=build_persona_policy(POLICY_GENESIS_PERSONA))


async def fresh_run(name, search, run_idx):
    path = RUNS / f"persona_{name}_run{run_idx}.sqlite"
    reset(path)
    archive = SQLiteArchive(path)
    await archive.put_artifact(build_persona_policy(POLICY_GENESIS_PERSONA))
    imp = improver(search, archive)
    await imp.start()
    try:
        for _ in range(ROUNDS):
            await imp.trigger_round()
    finally:
        await imp.stop()
    return await best_by_absolute(path)


async def gepa_fresh_run(run_idx):
    path = RUNS / f"persona_gepa_run{run_idx}.sqlite"
    reset(path)
    archive = SQLiteArchive(path)
    await archive.put_artifact(build_persona_policy(POLICY_GENESIS_PERSONA))
    gepa = GEPA(proposer_model=PROPOSER_MODEL, reflector=Reflection(model=PROPOSER_MODEL),
                agent=agent(POLICY_GENESIS_PERSONA), eval_source=FixedEvalSet(persona_eval_set(TRAIN)),
                population_size=GEPA_POP, generations=GEPA_GEN)
    imp = improver(gepa, archive)
    await imp.start()
    try:
        await imp.trigger_round()
    finally:
        await imp.stop()
    return await best_by_absolute(path)


async def main_async():
    attach_console_renderer(verbose=False)
    RUNS.mkdir(parents=True, exist_ok=True)

    # DGM: one persistent archive for all runs.
    dgm_path = RUNS / "persona_dgm_persistent.sqlite"
    reset(dgm_path)
    dgm_archive = SQLiteArchive(dgm_path)
    await dgm_archive.put_artifact(build_persona_policy(POLICY_GENESIS_PERSONA))
    dgm_imp = improver(DGMSearch(mutator=BlindLLMMutator(model=PROPOSER_MODEL), rounds=1), dgm_archive)

    rows = []
    await dgm_imp.start()
    try:
        for r in range(1, RUNS_N + 1):
            spo = await fresh_run("spo", SPO(proposer_model=PROPOSER_MODEL, first_round_model=AGENT_MODEL, rounds=1), r)
            gepa = await gepa_fresh_run(r)
            for _ in range(ROUNDS):       # DGM continues its archive (does NOT reset)
                await dgm_imp.trigger_round()
            dgm = await best_by_absolute(dgm_path)
            rows.append((r, spo, gepa, dgm))
            out(f"  run {r} done: SPO {spo[1]:.2f}  GEPA {gepa[1]:.2f}  DGM {dgm[1]:.2f} (test)")
    finally:
        await dgm_imp.stop()

    out(f"\n=== SPO vs GEPA vs DGM over {RUNS_N} runs ({AGENT_MODEL}, held-out test) ===")
    out(f"  selection denoised: top-{TOPK} finalists re-measured at {SAMPLES} rollouts/candidate.")
    out("  SPO/GEPA start fresh each run; DGM keeps a persistent, accumulating archive.\n")
    out(f"  {'method':<22}" + "".join(f"{'run'+str(r)+' tr/te':>14}" for r in range(1, RUNS_N + 1)))
    for idx, label in ((1, "SPO (fresh)"), (2, "GEPA (fresh)"), (3, "DGM (persistent)")):
        cells = "".join(f"{rows[r-1][idx][0]:>6.2f}/{rows[r-1][idx][1]:<7.2f}" for r in range(1, RUNS_N + 1))
        out(f"  {label:<22}{cells}")
    out("\n  DGM's archive accumulates across runs (its best is a ratchet); SPO/GEPA are independent draws.")


if __name__ == "__main__":
    asyncio.run(main_async())
