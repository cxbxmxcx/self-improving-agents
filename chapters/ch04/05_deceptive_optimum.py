"""Chapter 4: a deceptive landscape where greedy sticks and an archive escapes.

This is the no-cost mirror of the revenue cost ladder. A greedy hill-climber,
the shape of SPO, walks uphill and stops at the first peak it finds. An archive
search, the shape of DGM, keeps every point it has seen and explores broadly, so
it crosses the valley and finds the taller peak the climber never reached.

The landscape has a tempting local maximum at x=3 and the true maximum at x=9,
with a low valley between them. Both searches start from the same place. Run:

    python chapters/ch04/05_deceptive_optimum.py
"""
from __future__ import annotations

# A 1-D fitness landscape. The local max (x=3, height 6) traps a greedy climber;
# the global max (x=9, height 10) sits past a valley it will not cross.
LANDSCAPE = {0: 1, 1: 3, 2: 5, 3: 6, 4: 4, 5: 2, 6: 3, 7: 5, 8: 8, 9: 10, 10: 7}
START = 0


def neighbors(x: int) -> list[int]:
    return [n for n in (x - 1, x + 1) if n in LANDSCAPE]


def greedy_hill_climb() -> tuple[int, int]:
    """SPO-shaped: step to a better neighbor, stop when none is better."""
    x = START
    while True:
        uphill = [n for n in neighbors(x) if LANDSCAPE[n] > LANDSCAPE[x]]
        if not uphill:
            return x, LANDSCAPE[x]            # trapped at the first peak
        x = max(uphill, key=lambda n: LANDSCAPE[n])


def qd_weight(entry: dict) -> float:
    return entry["score"] / (2 ** entry["used"])


def archive_search(iterations: int = 30) -> tuple[int, int]:
    """DGM-shaped: keep every point ever seen, sample by quality-diversity, and
    expand the frontier. Nothing is discarded, so the valley does not block it."""
    archive = {START: {"score": LANDSCAPE[START], "used": 0}}
    for _ in range(iterations):
        parent = max(archive, key=lambda x: qd_weight(archive[x]))  # sample the archive
        archive[parent]["used"] += 1
        for n in neighbors(parent):                                 # explore, keep everything
            if n not in archive:
                archive[n] = {"score": LANDSCAPE[n], "used": 0}
    best = max(archive, key=lambda x: archive[x]["score"])
    return best, LANDSCAPE[best]


def main() -> None:
    gx, gscore = greedy_hill_climb()
    ax, ascore = archive_search()
    print(f"greedy hill-climb (SPO-shaped): stuck at x={gx}, height {gscore}")
    print(f"archive search   (DGM-shaped): reached x={ax}, height {ascore}")
    print()
    print("The climber stops at the first peak because every step toward the taller one")
    print("looks worse first. The archive keeps the low points too, crosses the valley,")
    print("and finds the global max. That is 0.63 versus 1.00 with no model in the loop.")


if __name__ == "__main__":
    main()
