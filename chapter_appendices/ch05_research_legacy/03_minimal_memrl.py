"""Chapter 5, section 5.3: a fixed one-step value update written directly.

Session one retains experience and session two assigns equal outcome credit
to recalled entries. The update q_new = q + alpha * (reward - q) changes
stored values while the agent harness's memory hooks remain unchanged.

This diagnostic keeps utility_weight=0.0, so utility updates alone do not
change recall ranking. Script 04 configures a nonzero weight before learning
and uses the same update formula through MemoryLearner.

Run:

    python chapter_appendices/ch05_research_legacy/03_minimal_memrl.py
    python chapter_appendices/ch05_research_legacy/03_minimal_memrl.py --smoke
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from helix.env import load_env
from helix.memory import operators as ops
from helix.memory.hooks import RECALLED_KEY

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

STEM = "minimal_memrl"
ALPHA = 0.5        # how far utility moves toward each outcome
FLOOR = 0.25       # entries below this are pruned from the live index


async def main() -> None:
    archive = fresh_archive(STEM)
    tiers = fresh_tiers(STEM, archive)
    epi = tiers["episodic"]
    slot: dict = {}
    agent = build_agent(tiers, question_id_slot=slot)
    traffic = load_traffic()

    # Session one: the agent answers and remembers. Every entry starts at
    # utility 0.5, unproven; the checker's verdicts are noted, not yet used.
    print("Session 1: filling memory with the agent's own answers.\n")
    s1_rows = []
    for asking in session(traffic, "-s1"):
        answer, traj, score = await run_asking(agent, asking, slot)
        s1_rows.append((asking.band, score))
        print(f"  [{asking.question_id}] {asking.user_id:<6} checker {score:.2f}")

    # Session two: recall, act, score, and credit. The eighty lines that
    # matter are the ones inside this loop.
    print("\nSession 2: recall, score, reinforce.\n")
    s2_rows = []
    log = []
    for asking in session(traffic, "-s2"):
        answer, traj, reward = await run_asking(agent, asking, slot)
        s2_rows.append((asking.band, reward))
        recalled = traj.metadata.get(RECALLED_KEY, [])
        for r in recalled:
            if not r["artifact_ref"]:
                continue
            entry = await epi.get(r["entry_id"])
            if entry is None:
                continue
            q = entry.score
            q_new = q + ALPHA * (reward - q)          # the law of effect, as a number
            if q_new > q:
                entry = await ops.upvote(epi, entry, delta=q_new - q, created_by="minimal_memrl", reason=f"reward {reward:.2f} on {asking.question_id}")
            elif q_new < q:
                entry = await ops.downvote(epi, entry, delta=q - q_new, created_by="minimal_memrl", reason=f"reward {reward:.2f} on {asking.question_id}")
            log.append({"question_id": asking.question_id, "entry": entry.id, "q_before": q, "q_after": entry.score, "reward": reward})
        print(f"  [{asking.question_id}] {asking.user_id:<6} reward {reward:.2f}  recalled {len(recalled)}  "
              + ", ".join(f"{l['q_before']:.2f}->{l['q_after']:.2f}" for l in log[-len(recalled):]))

    # Prune: whatever the outcomes kept pushing down leaves the live index.
    # The archive keeps every version, so nothing is lost, only unrecalled.
    dropped = await ops.evict_below(epi, FLOOR)
    print(f"\nPruned {dropped} entr{'y' if dropped == 1 else 'ies'} below utility {FLOOR}.")

    print("\nThe store after one pass of reinforcement (live index, by utility):\n")
    rows = sorted(await epi.list_all(), key=lambda e: e.score, reverse=True)
    for e in rows:
        payload = e.content
        print(f"  {e.score:.2f}  v{e.artifact_ref[1]}  {e.scope_key.key:<6} {str(payload.get('user_message', ''))[:60]}")

    print_band_table("Checker means, session 1 (memory filling) vs session 2 (recall of session 1)", {
        "session 1": s1_rows,
        "session 2": s2_rows,
    })
    write_jsonl(RUNS_DIR / f"{STEM}_updates.jsonl", log)
    print(f"\nArchive: {archive.path}")


if __name__ == "__main__":
    with maybe_smoke():
        asyncio.run(main())
