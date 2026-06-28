"""Chapter 4: the Pareto front in eight lines (no LLM, no cost).

When a candidate is judged on more than one objective, there is rarely a single
winner. The honest thing to keep is the Pareto front: every candidate that
nothing else beats on all objectives at once, so each survivor is best at
something.

Here the objectives are quality (higher is better) and cost (lower is better).
The front is the trade-off staircase, a high-quality expensive option next to a
cheap mediocre one, with the genuinely worse options dropped. Run:

    python chapters/ch04/02_pareto_demo.py
"""
from __future__ import annotations

# (name, quality: higher better, cost: lower better)
CANDIDATES = [
    ("A", 0.90, 9),
    ("B", 0.80, 4),
    ("C", 0.60, 2),
    ("D", 0.70, 9),  # dominated by B (lower quality, higher cost)
    ("E", 0.50, 5),  # dominated by C (lower quality, higher cost)
]


def dominates(x, y) -> bool:
    """x dominates y if x is at least as good on BOTH objectives and strictly
    better on at least one. Higher quality is better; lower cost is better."""
    at_least_as_good = x[1] >= y[1] and x[2] <= y[2]
    strictly_better = x[1] > y[1] or x[2] < y[2]
    return at_least_as_good and strictly_better


def pareto_front(candidates):
    """Keep every candidate that nothing else dominates."""
    return [c for c in candidates if not any(dominates(o, c) for o in candidates if o is not c)]


def main() -> None:
    front = pareto_front(CANDIDATES)
    front_names = {c[0] for c in front}
    print("candidate   quality   cost   verdict")
    for name, q, cost in CANDIDATES:
        verdict = "on the front" if name in front_names else "dominated, dropped"
        print(f"    {name}        {q:.2f}     {cost:>2}    {verdict}")
    print()
    print("The front is the trade-off staircase: A buys quality with cost, C buys")
    print("cheapness with quality, B sits between. None of the three beats another on")
    print("both axes, so all three survive while D and E are dropped.")


if __name__ == "__main__":
    main()
