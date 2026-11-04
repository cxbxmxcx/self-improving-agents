"""Chapter 5, section 5.4: attach value-based learning to the memory harness.

The same harness reads and writes memory throughout the traffic stream.
OnlineMemoryImprover feeds each completed run to a fixed MemoryLearner;
learned utility changes while hook code and configured recall settings do not.

The script replays traffic and then runs one configured consolidation pass.
The saved pass pruned and merged entries but made no scope promotions, and
it did not evaluate answer quality again after consolidation.

Watch it live: open the Observatory's Memory panel on this script's archive
before running, and the store grows, votes, and consolidates in place.

Run:

    python chapter_appendices/ch05_research_legacy/06_online_memory_loop.py
    python chapter_appendices/ch05_research_legacy/06_online_memory_loop.py --smoke
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from helix.env import load_env
from helix.improvement.online_memory import OnlineMemoryImprover
from helix.memory.consolidation import ConsolidationPolicy, consolidate_episodic
from helix.memory.learning import MemoryLearner
from helix.signals.utility import UtilitySignal

from chapter_appendices.ch05_research_legacy.common import (
    RUNS_DIR,
    build_agent,
    fresh_archive,
    fresh_tiers,
    maybe_smoke,
    print_band_table,
    reward_for,
    run_asking,
    session,
    setup_console,
    write_jsonl,
)
from chapter_appendices.ch05_research_legacy.traffic import load_traffic

load_env()
setup_console()

STEM = "online_memory"
ALPHA = 0.5
FLOOR = 0.25
PROMOTE = 0.7
UTILITY_WEIGHT = 0.35


async def main() -> None:
    archive = fresh_archive(STEM)
    tiers = fresh_tiers(STEM, archive, utility_weight=UTILITY_WEIGHT)
    epi = tiers["episodic"]
    slot: dict = {}
    agent = build_agent(tiers, question_id_slot=slot)

    # The harness and learner share this exact live tier. Learning changes
    # its stored values, not the recall formula, its weight, or hook timing.
    learner = MemoryLearner(
        agent.memory["episodic"], archive, UtilitySignal(attribution="position"),
        alpha=ALPHA, downvote_threshold=0.5,
        upvote_threshold=0.5, created_by="memrl",
    )
    improver = OnlineMemoryImprover(agent, learner=learner, reward_fn=reward_for)
    agent.attach_improver(improver)
    await improver.start()

    print(f"Replaying {len(load_traffic())} requests through the fixed memory harness.\n")
    rows = []
    for asking in load_traffic():
        _, traj, score = await run_asking(agent, asking, slot)
        rows.append({"session": asking.session_id, "question_id": asking.question_id, "band": asking.band, "score": score})
        status = improver.status()
        print(f"  [{asking.session_id:>9}] {asking.question_id:>4} reward {score:.2f}   "
              f"updates {status.updates_applied}, entries touched {status.entries_touched}")

    status = improver.status()
    print(f"\nAfter the stream: {status.trajectories_seen} trajectories, "
          f"{status.updates_applied} updates, {status.entries_touched} entry versions minted, "
          f"{status.skipped_no_reward} skipped.")

    print("\nThe 3 AM pass:\n")
    report = await consolidate_episodic(
        epi,
        ConsolidationPolicy(min_utility=FLOOR, promote_utility=PROMOTE),
        bus=agent.bus,
    )
    print(f"  {report.entries_before} entries before, {report.entries_after} after")
    print(f"  {report.notes}")

    print("\nThe store the next morning (top of the live index):\n")
    for e in sorted(await epi.list_all(), key=lambda e: e.score, reverse=True)[:10]:
        print(f"  u={e.score:.2f}  {e.scope_key.to_str():<14} {str(e.content.get('user_message', ''))[:55]}")

    s1 = [(r["band"], r["score"]) for r in rows if r["session"].endswith("-s1")]
    s2 = [(r["band"], r["score"]) for r in rows if r["session"].endswith("-s2")]
    print_band_table("Checker means as the stream ran", {"first sessions": s1, "second sessions": s2})
    write_jsonl(RUNS_DIR / f"{STEM}_stream.jsonl", rows)
    print(f"\nArchive: {archive.path}")
    print("Reproduce the Observatory view: python helix_web/run.py, Memory panel, archive chapters.ch05.runs.online_memory")


if __name__ == "__main__":
    with maybe_smoke():
        asyncio.run(main())
