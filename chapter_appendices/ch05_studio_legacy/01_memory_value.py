"""Listing 5.1: learn expected usefulness from outcomes, then use it in recall.

Illustrative arithmetic only. The harness selects a case; the support agent
chooses its repair action. The shared learner applies this same value update
to earlier cases supplied as context, without changing the model or harness.
"""

def value_update(q: float, target: float, alpha: float) -> float:
    return q + alpha * (target - q)



def main() -> None:
    print("Same rewards [1, 0], different learning rates")
    for alpha in (0.1, 0.5, 1.0):
        q = 0.5
        values = [q]
        for reward in (1.0, 0.0):
            q = value_update(q, reward, alpha)
            values.append(q)
        path = " -> ".join(f"{value:.3f}" for value in values)
        print(f"  alpha={alpha:.1f}: {path}")

    print("\nFixed recall: same query, entries, relevance, and weight")
    for name, relevance, before, after in (
        ("E1", 0.60, 0.75, value_update(0.75, 0.0, 0.5)),
        ("E2", 0.55, 0.50, 0.50),
    ):
        old_score = 0.65 * relevance + 0.35 * before
        new_score = 0.65 * relevance + 0.35 * after
        print(f"  {name}: recall score {old_score:.5f} -> {new_score:.5f}")


if __name__ == "__main__":
    main()
