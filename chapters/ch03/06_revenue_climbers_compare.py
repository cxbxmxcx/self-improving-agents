"""Chapter 3: two single-candidate climbers, one wall (no LLM, reads the archives).

Run the two loops first so their archives exist:

    python chapters/ch03/04_revenue_hillclimb_loop.py
    python chapters/ch03/05_revenue_spo_loop.py

then run this to read both archives and put the result side by side:

    python chapters/ch03/06_revenue_climbers_compare.py

Hill-climbing ratchets on the absolute ground-truth score; SPO accepts on a
pairwise judge's preference. They consume different signals, but both keep only
one candidate at a time, and on a signal that cannot say which rule is missing,
a single climber gets stuck either way. Both stall at the genesis score. That
shared wall is what chapter 4 climbs.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from helix.archive import SQLiteArchive

RUNS = Path(__file__).parent / "runs"
ARCHIVES = [
    ("hill-climb", RUNS / "revenue_hillclimb.sqlite"),
    ("SPO", RUNS / "revenue_spo.sqlite"),
]


async def summarize(path: Path) -> tuple[float, float, int] | None:
    """Return (genesis_score, best_score, rounds) read from one run archive, or
    None if the archive does not exist yet."""
    if not path.exists():
        return None
    variants = await SQLiteArchive(path).all_artifacts_with_measurements()
    scored = [(v.artifact.created_by, v.measurement.score) for v in variants
              if v.measurement is not None and v.measurement.score is not None]
    genesis = next((s for who, s in scored if who == "human"), None)
    candidates = [s for who, s in scored if who != "human"]
    best = max((s for _, s in scored), default=genesis)
    return genesis, best, len(candidates)


async def main_async() -> None:
    print("Two single-candidate climbers on the revenue task\n")
    print(f"{'method':<12}{'genesis':<10}{'best':<8}{'rounds':<8}{'net gain':<10}")
    for name, path in ARCHIVES:
        result = await summarize(path)
        if result is None:
            print(f"{name:<12}(no archive yet; run its loop first: {path.name})")
            continue
        genesis, best, rounds = result
        print(f"{name:<12}{genesis:<10.2f}{best:<8.2f}{rounds:<8}{best - genesis:+.2f}")
    print()
    print("Both stall at the genesis score. Hill-climb ratchets on an absolute score")
    print("and SPO on a pairwise preference, but a single climber on a signal that")
    print("cannot localize the missing rule gets stuck either way. That is the wall.")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
