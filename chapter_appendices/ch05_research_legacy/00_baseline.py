"""Chapter 5, step zero: the traffic set and a checker we can look in the eye.

Two jobs before any memory learns anything. First, build the replayable
traffic: the twenty helpdesk questions, reworded by hand so the second
session's wording differs from the first, laid out as three users over two
sessions each. Second, run the memoryless helpdesk over the
original wording and print the checker's verdict beside every answer, so
the reward the chapter learns from has been read by a human before a loop
trusts it (chapter 3's rule: validate the checker before believing it).

Run:

    python chapter_appendices/ch05_research_legacy/00_baseline.py            # build traffic.json if missing, then baseline
    python chapter_appendices/ch05_research_legacy/00_baseline.py --rebuild  # rewrite traffic.json from traffic.py first
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from helix.env import load_env

from agents import helpdesk
from chapter_appendices.ch05_research_legacy.traffic import (
    TRAFFIC_PATH,
    build_traffic,
    check,
    load_traffic,
    save_traffic,
)

load_env()
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

RUNS_DIR = Path(__file__).parent / "runs"
BASELINE_PATH = RUNS_DIR / "baseline_no_memory.jsonl"
SEM = asyncio.Semaphore(4)


def build_traffic_file() -> None:
    askings = build_traffic()
    save_traffic(askings)
    print(f"Wrote {len(askings)} askings to {TRAFFIC_PATH}")


async def run_one(asking, agent_factory):
    async with SEM:
        agent = agent_factory()
        t0 = time.perf_counter()
        answer, trajectory = await agent.run(asking.text)
        return {
            "question_id": asking.question_id,
            "band": asking.band,
            "user_id": asking.user_id,
            "text": asking.text,
            "answer": answer,
            "score": check(asking.question_id, answer),
            "elapsed_sec": round(time.perf_counter() - t0, 1),
            "n_tool_calls": sum(1 for s in trajectory.steps if s.kind.value == "tool_call"),
        }


async def baseline() -> None:
    askings = [a for a in load_traffic() if a.session_id.endswith("-s1")]
    print(f"\nBaseline: {len(askings)} questions, no memory, {helpdesk.DEFAULT_MODEL}")
    records = await asyncio.gather(*(run_one(a, lambda: helpdesk.build(skip_memory=True)) for a in askings))
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    with BASELINE_PATH.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    print("\nChecker verdicts (read these before trusting the reward):\n")
    for r in sorted(records, key=lambda r: int(r["question_id"][1:])):
        flat = " ".join((r["answer"] or "").split())
        print(f"{r['question_id']:>3} band {r['band']}  score {r['score']:.2f}  {flat[:150]}")
    by_band: dict[int, list[float]] = {}
    for r in records:
        by_band.setdefault(r["band"], []).append(r["score"])
    print("\nPer-band mean score:")
    for band in sorted(by_band):
        scores = by_band[band]
        print(f"  band {band}: {sum(scores) / len(scores):.2f}  (n={len(scores)})")
    overall = sum(r["score"] for r in records) / len(records)
    print(f"  overall: {overall:.2f}")
    print(f"\nWrote {BASELINE_PATH}")


async def main() -> None:
    if "--rebuild" in sys.argv or not TRAFFIC_PATH.exists():
        build_traffic_file()
    await baseline()


if __name__ == "__main__":
    asyncio.run(main())
