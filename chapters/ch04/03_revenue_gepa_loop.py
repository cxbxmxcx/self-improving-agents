"""Chapter 4: the improvement loop with GEPA on the revenue task.

Same loop, different Search. Where SPO mutates blindly from the outcome score and
plateaus at 0.63, GEPA does three things SPO does not:

  1. Reflection. It reads the agent's actual execution trace (the tool calls it
     made) and an LLM diagnoses which rules the agent omitted. That is far richer
     than "the total is too high", so the mutation can target the real cause.
  2. Population. It keeps several candidates, not one line, so partial wins on
     different rules survive instead of being overwritten.
  3. Crossover. It combines two candidates, merging a prompt good at one rule with
     a prompt good at another.

The reflector and proposer are the SAME mid-tier sonnet SPO used, so any gain comes
from the mechanism, not a stronger model. Run:

    python chapters/ch04/03_revenue_gepa_loop.py
"""
from __future__ import annotations

import asyncio
import json
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

# Proposed prompts may contain characters a Windows console cannot encode;
# printing a result must never be able to crash a finished run.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

AGENT_MODEL = "claude-sonnet-4-5"      # runs the task; temperature 0 -> deterministic scoring
REFLECTOR_MODEL = "claude-sonnet-4-5"  # diagnoses the trajectory (same mid-tier as SPO's proposer)
PROPOSER_MODEL = "claude-sonnet-4-5"   # mutation and crossover
ROUNDS = 5
POP = 3
CONCURRENCY = 4
SEM = asyncio.Semaphore(CONCURRENCY)
ARCHIVE_PATH = Path(__file__).parent / "runs" / "revenue_gepa.sqlite"

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
CROSSOVER_SYS = (
    "Combine two data-analyst system prompts into one that keeps every correct rule "
    "from both and drops redundancy. Output only the new prompt."
)


def _digest(trajectory) -> str:
    """The agent's tool calls, in order, with arguments. This is what reflection
    reads to see what the agent actually did (and did not) filter or compute."""
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
    """Run the agent and return (score, feedback, trajectory). GEPA needs the
    trajectory; SPO never looks at it."""
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


async def crossover(a: str, b: str) -> str:
    return await _llm(CROSSOVER_SYS, f"Prompt A:\n{a}\n\nPrompt B:\n{b}", PROPOSER_MODEL)


ROUNDS_LOG = ARCHIVE_PATH.parent / f"{ARCHIVE_PATH.stem}_rounds.jsonl"


def _fresh_archive() -> SQLiteArchive:
    for ext in ("", "-wal", "-shm"):
        p = Path(str(ARCHIVE_PATH) + ext)
        if p.exists():
            p.unlink()
    if ROUNDS_LOG.exists():
        ROUNDS_LOG.unlink()
    ARCHIVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    return SQLiteArchive(ARCHIVE_PATH)


def _log_round(row: dict) -> None:
    """One JSON line per generation, so the dashboard can replay the population."""
    with ROUNDS_LOG.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")


async def improve_gepa(rounds: int, pop: int):
    archive = _fresh_archive()
    seed = build_revenue_policy(POLICY_GENESIS)

    s0, fb0, traj0 = await measure_full(POLICY_GENESIS)
    print(f"genesis: score={s0:.2f}")
    # Record the genesis with its score so the lineage tree has a ranked root.
    # Feedback and the trajectory persist too, so the score stays auditable.
    await archive.record(
        Variant(artifact=seed, parent=seed, search_method="human"),
        GapMeasurement(score=s0, feedback=fb0),
    )
    await archive.store_trajectory(
        seed, "revenue", traj0, answer=getattr(traj0, "final_output", None)
    )
    # population entries: (artifact, content, score, trajectory)
    population = [(seed, POLICY_GENESIS, s0, traj0)]

    for r in range(rounds):
        parents = sorted(population, key=lambda x: x[2], reverse=True)[: max(2, pop // 2)]
        # The living population at breeding time, stamped onto every child so
        # the lineage view can show what the search could see when it bred.
        pool_versions = [art.version for art, _, _, _ in population]

        # Reflective mutation: diagnose each parent's trajectory, then mutate.
        # Each proposal carries its second parent, None except for crossover.
        async def make_child(par):
            diagnosis = await reflect(par[1], par[3])
            return par[0], await mutate(par[1], diagnosis), None

        jobs = [make_child(p) for p in parents]
        if len(parents) >= 2:  # one crossover of the top two
            async def make_cross():
                merged = await crossover(parents[0][1], parents[1][1])
                return parents[0][0], merged, parents[1][0]
            jobs.append(make_cross())
        proposed = await asyncio.gather(*jobs)

        # Evaluate offspring (concurrently), record to the archive with lineage.
        async def eval_child(parent_art, content, other_parent):
            sc, fb, traj = await measure_full(content)
            nv = await archive.next_version(parent_art.id) if hasattr(archive, "next_version") else parent_art.version + 1
            # parent_id holds a single ref, so a crossover child files its
            # second parent in metadata; the lineage tree draws it as a
            # dashed edge from that breadcrumb.
            meta = (
                {"other_parent_id": other_parent.id,
                 "other_parent_version": other_parent.version}
                if other_parent is not None else None
            )
            created_by = "gepa_crossover" if other_parent is not None else "gepa_mutation"
            child = parent_art.mutate(
                new_content=content, created_by=created_by, metadata=meta, version=nv
            )
            # Record with lineage + measurement so the dashboard can color and rank it.
            await archive.record(
                Variant(
                    artifact=child,
                    parent=parent_art,
                    search_method="gepa",
                    metadata={"round": r, "bred_from_pool": pool_versions},
                ),
                GapMeasurement(score=sc, feedback=fb),
            )
            await archive.store_trajectory(
                child, "revenue", traj, answer=getattr(traj, "final_output", None)
            )
            return child, content, sc, traj

        evaluated = await asyncio.gather(*[eval_child(pa, c, op) for pa, c, op in proposed])
        population.extend(evaluated)
        population = sorted(population, key=lambda x: x[2], reverse=True)[:pop]
        print(f"gen {r}: best={population[0][2]:.2f}   (population: {[round(x[2], 2) for x in population]})")
        _log_round({
            "round": r,
            "best_score": population[0][2],
            "mean_score": sum(x[2] for x in population) / len(population),
            "population": [art.version for art, _, _, _ in population],
        })
        # A ground-truth signal knows its own ceiling: a perfect check means the
        # task is solved, so every call after it would be pure waste.
        if population[0][2] >= 1.0:
            print(f"ceiling reached at gen {r}: ground truth reports a perfect answer, stopping")
            break

    best = max(population, key=lambda x: x[2])
    print(f"\nGEPA best score={best[2]:.2f}\nbest prompt:\n{best[1]}")
    return best[1], best[2]


async def main_async():
    await improve_gepa(ROUNDS, POP)


if __name__ == "__main__":
    asyncio.run(main_async())
