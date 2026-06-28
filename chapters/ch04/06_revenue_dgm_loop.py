"""Chapter 3: the improvement loop with DGM on the revenue task.

DGM (a Darwin-Godel-Machine-style search) keeps an ACCUMULATING archive that never
discards a candidate, and samples parents by quality AND diversity rather than only
the current best. Where GEPA's small population converged at 0.86 and lost the
diversity to recover the last rule, DGM keeps exploring lower-scoring and
structurally different candidates, so a variant that happens to phrase the status
filter as 'status == ok' (excluding cancelled AND returned) survives and can be
built on. The reflection and proposer are the same mid-tier sonnet as SPO and GEPA;
the difference is purely the search structure.

DGM does more total work (more iterations, a growing archive) than SPO or GEPA, so
this is the top of the cost ladder. Run:

    python chapters/ch04/06_revenue_dgm_loop.py
"""
from __future__ import annotations

import asyncio
import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from helix.archive import SQLiteArchive
from helix.env import load_env
from helix.llm_call import acompletion
from helix.search.base import Variant
from helix.signal import GapMeasurement

from agents.revenue import (
    POLICY_GENESIS, TASK_REQUEST, build_revenue_agent, build_revenue_policy,
    reconstruct_answer, score_smooth_with_feedback,
)

load_env()

AGENT_MODEL = "claude-sonnet-4-5"
REFLECTOR_MODEL = "claude-sonnet-4-5"
PROPOSER_MODEL = "claude-sonnet-4-5"
ITERS = 14            # more total work than SPO/GEPA: DGM is the top of the cost ladder
EXPLOIT_P = 0.6       # probability of exploiting a top candidate vs exploring a diverse one
SEM = asyncio.Semaphore(4)
ARCHIVE_PATH = Path(__file__).parent / "runs" / "revenue_dgm.sqlite"

REFLECT_SYS = (
    "You diagnose why a data-analyst agent produced wrong revenue totals. You see "
    "the agent's system prompt and the exact sequence of tool calls it made. Work "
    "out which filters or computations it omitted: did it filter order status, "
    "channel, and payment? did it subtract the refund column? did it pick the right "
    "quarter and group by rep? Output a short, concrete list of corrections to add "
    "to the prompt. Do not output a full prompt, just the list of fixes."
)
MUTATE_SYS = (
    "You improve a data-analyst agent's system prompt. Given the current prompt and "
    "a diagnosis of the rules it is missing, rewrite it to state those rules "
    "explicitly and precisely. Keep what already works. Do NOT hard-code answer "
    "numbers. Output only the new prompt."
)


def _digest(trajectory) -> str:
    lines = []
    for step in getattr(trajectory, "steps", []):
        kind = getattr(getattr(step, "kind", None), "value", getattr(step, "kind", None))
        if kind == "tool_call":
            p = step.payload or {}
            lines.append(f"{p.get('name')}({json.dumps(p.get('arguments', {}))})")
    return "\n".join(lines) or "(no tool calls)"


async def _llm(system: str, user: str, model: str) -> str:
    resp = await acompletion(model=model, messages=[
        {"role": "system", "content": system}, {"role": "user", "content": user}])
    return (resp.choices[0].message.content or "").strip()


async def measure_full(content: str):
    async with SEM:
        agent = build_revenue_agent(content, model=AGENT_MODEL, temperature=0.0)
        _, trajectory = await agent.run(TASK_REQUEST)
    score, feedback = score_smooth_with_feedback(reconstruct_answer(trajectory))
    return score, feedback, trajectory


async def reflect(content: str, trajectory) -> str:
    user = (f"System prompt:\n{content}\n\nThe agent made these tool calls:\n"
            f"{_digest(trajectory)}\n\nThe reported totals are wrong. Diagnose the missing rules.")
    return await _llm(REFLECT_SYS, user, REFLECTOR_MODEL)


async def mutate(content: str, diagnosis: str) -> str:
    user = f"Current prompt:\n{content}\n\nMissing rules to add:\n{diagnosis}\n\nRewrite the prompt."
    return await _llm(MUTATE_SYS, user, PROPOSER_MODEL)


def sample_parent(archive: list):
    """Quality-diversity sampling: usually exploit a top candidate, but with
    probability 1-EXPLOIT_P explore a random archive member (including lower-scoring,
    structurally different ones). This is what keeps DGM from collapsing onto one
    line the way a small population does."""
    ranked = sorted(archive, key=lambda e: e["score"], reverse=True)
    if random.random() < EXPLOIT_P:
        return random.choice(ranked[: min(3, len(ranked))])
    return random.choice(archive)


def _fresh_archive() -> SQLiteArchive:
    for ext in ("", "-wal", "-shm"):
        p = Path(str(ARCHIVE_PATH) + ext)
        if p.exists():
            p.unlink()
    ARCHIVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    return SQLiteArchive(ARCHIVE_PATH)


async def improve_dgm(iters: int):
    random.seed(0)  # reproducible exploration for the book
    store = _fresh_archive()
    seed = build_revenue_policy(POLICY_GENESIS)

    s0, _, traj0 = await measure_full(POLICY_GENESIS)
    print(f"genesis: score={s0:.2f}")
    # Record the genesis with its score so the lineage tree has a ranked root.
    await store.record(
        Variant(artifact=seed, parent=seed, search_method="human"),
        GapMeasurement(score=s0),
    )
    archive = [{"art": seed, "content": POLICY_GENESIS, "score": s0, "traj": traj0}]
    best = s0

    for it in range(iters):
        parent = sample_parent(archive)
        diagnosis = await reflect(parent["content"], parent["traj"])
        child_content = await mutate(parent["content"], diagnosis)
        sc, _, traj = await measure_full(child_content)
        nv = await store.next_version(parent["art"].id) if hasattr(store, "next_version") else parent["art"].version + 1
        child_art = parent["art"].mutate(new_content=child_content, created_by="dgm", version=nv)
        # Record with lineage + measurement so the dashboard can color and rank it.
        await store.record(
            Variant(artifact=child_art, parent=parent["art"], search_method="dgm"),
            GapMeasurement(score=sc),
        )
        archive.append({"art": child_art, "content": child_content, "score": sc, "traj": traj})  # never discard
        best = max(best, sc)
        print(f"iter {it}: candidate={sc:.2f}  best={best:.2f}  archive={len(archive)}  (from parent {parent['score']:.2f})")

    winner = max(archive, key=lambda e: e["score"])
    print(f"\nDGM best score={winner['score']:.2f}\nbest prompt:\n{winner['content']}")
    return winner["content"], winner["score"]


async def main_async():
    await improve_dgm(ITERS)


if __name__ == "__main__":
    asyncio.run(main_async())
