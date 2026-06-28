"""Chapter 4: quality-diversity sampling in ten lines (no LLM, no cost).

An archive that never deletes is only useful if you go back into it. Quality-
diversity sampling is what sends you back: it weights each entry by how good it
is AND how rarely it has been used, so the search cannot fall in love with the
current best and forget everything else.

This demo runs the QD rule next to plain greedy on the same three-entry archive.
Greedy picks the top score every single time; QD picks it first, then the weight
decays and a strong-but-neglected entry gets its turn. Run:

    python chapters/ch04/04_qd_sampler.py
"""
from __future__ import annotations


def qd_weight(entry: dict) -> float:
    """High score is attractive, but each use halves the weight, so a neglected
    entry keeps resurfacing instead of the search fixating on one peak."""
    return entry["score"] / (2 ** entry["used"])


def main() -> None:
    archive = {
        "hi": {"score": 0.9, "used": 0},
        "mid": {"score": 0.7, "used": 0},
        "lo": {"score": 0.5, "used": 0},
    }

    print("draw   greedy   quality-diversity")
    for draw in range(8):
        greedy = max(archive, key=lambda k: archive[k]["score"])  # always the top score
        qd = max(archive, key=lambda k: qd_weight(archive[k]))    # decays with reuse
        archive[qd]["used"] += 1
        print(f"  {draw}      {greedy:<7}  {qd}")

    print()
    print("Greedy never leaves 'hi'. QD samples 'hi', then its weight decays and 'mid'")
    print("and even 'lo' get a turn, so every branch in the archive stays alive. That is")
    print("how DGM revisits an old branch a converging population would have thrown away.")


if __name__ == "__main__":
    main()
