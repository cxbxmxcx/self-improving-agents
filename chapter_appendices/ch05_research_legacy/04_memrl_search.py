"""Chapter 5, section 5.3: fixed value-based learning in the memory harness.

The harness applies memory through its configured recall and write-back
hooks. MemoryLearner uses q_new = q + alpha * (credit - q) to update stored
values, without changing that application mechanism or its own algorithm.

UtilitySignal supplies position-weighted credit under a fixed configuration.
The final replays compare different saved stores and fixed recall settings;
they do not isolate a causal effect of utility learning on unseen questions.

Run:

    python chapter_appendices/ch05_research_legacy/04_memrl_search.py
    python chapter_appendices/ch05_research_legacy/04_memrl_search.py --smoke
"""
from __future__ import annotations

import asyncio
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from helix.artifact import ArtifactKind
from helix.env import load_env
from helix.memory.episodic import EpisodicMemory
from helix.memory.hooks import RECALLED_KEY
from helix.memory.learning import MemoryLearner, ieu
from helix.signals.utility import UtilitySignal

from chapter_appendices.ch05_research_legacy.common import (
    RUNS_DIR,
    build_agent,
    fresh_archive,
    fresh_tiers,
    maybe_smoke,
    print_band_table,
    run_asking,
    session,
    setup_console,
    write_jsonl,
)
from chapter_appendices.ch05_research_legacy.traffic import load_traffic

load_env()
setup_console()

STEM = "memrl_search"
ALPHA = 0.5
FLOOR = 0.25
# A fixed harness setting chosen before learning; the learner never tunes it.
UTILITY_WEIGHT = 0.35


def snapshot(tier: EpisodicMemory, name: str, archive, utility_weight: float = 0.0) -> EpisodicMemory:
    """Copy a tier's SQLite file so a later phase can read it unchanged."""
    tier._conn.commit()
    src = Path(tier.path)
    dst = src.with_name(f"{src.stem}_{name}.sqlite")
    shutil.copyfile(src, dst)
    return EpisodicMemory(dst, check_same_thread=False, archive=archive, utility_weight=utility_weight)


async def replay(askings, tiers, label: str) -> list[tuple[int, float]]:
    """Session two again, recall only, no write-back, no learning."""
    slot: dict = {}
    agent = build_agent(tiers, question_id_slot=slot, read_only=True)
    rows = []
    for asking in askings:
        _, _, score = await run_asking(agent, asking, slot, session_suffix=f"-{label}")
        rows.append((asking.band, score))
    return rows


async def main() -> None:
    archive = fresh_archive(STEM)
    tiers = fresh_tiers(STEM, archive, utility_weight=UTILITY_WEIGHT)
    epi = tiers["episodic"]
    slot: dict = {}
    agent = build_agent(tiers, question_id_slot=slot)
    traffic = load_traffic()
    s1, s2 = session(traffic, "-s1"), session(traffic, "-s2")

    # Learn in the same tier the harness reads. Position credit is a configured
    # heuristic; it does not establish which recalled entry caused success.
    utility = UtilitySignal(attribution="position")
    learner = MemoryLearner(agent.memory["episodic"], archive, utility, alpha=ALPHA,
                            downvote_threshold=0.5, upvote_threshold=0.5, created_by="memrl")

    print("Session 1: the agent retains experience (utility updates are not yet enabled).\n")
    for asking in s1:
        _, _, score = await run_asking(agent, asking, slot)
        print(f"  [{asking.question_id}] {asking.user_id:<6} checker {score:.2f}")
    untouched = snapshot(epi, "after_s1", archive)

    print("\nSession 2: fixed recall rule, then one value update per run.\n")
    q_log = []
    s2_live = []
    for asking in s2:
        _, traj, reward = await run_asking(agent, asking, slot)
        s2_live.append((asking.band, reward))
        moved = await learner.update(traj, reward=reward)
        for e in moved:
            q_log.append({"question_id": asking.question_id, "entry": e.id, "version": e.artifact_ref[1], "q": e.score, "reward": reward})
        arrows = ", ".join(f"v{e.artifact_ref[1]}:{e.score:.2f}" for e in moved)
        print(f"  [{asking.question_id}] {asking.user_id:<6} reward {reward:.2f}  recalled {len(traj.metadata.get(RECALLED_KEY, []))}  moved {len(moved)}  {arrows}")
    pruned = await learner.prune(FLOOR)
    print(f"\nPruned {pruned} entr{'y' if pruned == 1 else 'ies'} below utility {FLOOR}.")
    learned = snapshot(epi, "after_memrl", archive, utility_weight=UTILITY_WEIGHT)

    print("\nThe best entries by utility, in Intent-Experience-Utility form:\n")
    for v in await learner.best_entries(k=5):
        row = ieu(v.artifact)
        print(f"  u={row['utility']:.2f}  v{v.artifact.version}  intent: {str(row['intent'])[:50]}")
        print(f"          experience: {str(row['experience'])[:90]}")

    print("\nReplaying session 2 three ways (recall only, no learning):")
    no_memory = await replay(s2, None, "nomem")
    unlearned = await replay(s2, {"episodic": untouched}, "unlearned")
    reinforced = await replay(s2, {"episodic": learned}, "memrl")
    print_band_table("Checker means on session 2's questions", {
        "no memory": no_memory,
        "memory, unlearned": unlearned,
        "learned memory": reinforced,
    })

    n_mem = sum(1 for v in await archive.all_artifacts_with_measurements() if v.artifact.kind is ArtifactKind.MEMORY_ENTRY)
    write_jsonl(RUNS_DIR / f"{STEM}_q.jsonl", q_log)
    write_jsonl(RUNS_DIR / f"{STEM}_comparison.jsonl", [
        {"condition": name, "band": b, "score": s}
        for name, rows in (("no_memory", no_memory), ("unlearned", unlearned), ("memrl", reinforced))
        for b, s in rows
    ])
    print(f"\n{n_mem} memory artifact versions in {archive.path}")
    print("The Observatory's Memory panel shows every version and its utility history.")


if __name__ == "__main__":
    with maybe_smoke():
        asyncio.run(main())
