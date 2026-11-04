"""Listing 5.2: value updates and a controlled recall calculation.

These are illustrative arithmetic, not measured support-agent results.
The harness uses the observed-outcome update, not the Q-learning contrast.
"""

def value_update(q: float, target: float, alpha: float) -> float:
    return q + alpha * (target - q)


def q_learning_target(
    reward: float, next_values: list[float], gamma: float, terminal: bool
) -> float:
    # A terminal transition has no future return to bootstrap from.
    future = 0.0 if terminal else max(next_values)
    return reward + gamma * future


def main() -> None:
    print("Same rewards [1, 1, 0], different learning rates")
    for alpha in (0.1, 0.5, 1.0):
        q = 0.5
        values = [q]
        for reward in (1.0, 1.0, 0.0):
            q = value_update(q, reward, alpha)
            values.append(q)
        path = " -> ".join(f"{value:.3f}" for value in values)
        print(f"  alpha={alpha:.1f}: {path}")

    print("\nQ-learning targets (initial Q=0.2, alpha=0.5)")
    for terminal, reward in ((False, -0.1), (True, 1.0)):
        target = q_learning_target(reward, [0.8, 0.3], 0.9, terminal)
        updated = value_update(0.2, target, 0.5)
        print(f"  terminal={terminal}: target={target:.3f}; Q_new={updated:.3f}")

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
