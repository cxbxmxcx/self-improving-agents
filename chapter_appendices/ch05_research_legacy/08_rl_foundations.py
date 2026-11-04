"""Reproduce Chapter 5's return, step-size, and Q-target calculations.

These are deterministic arithmetic examples, separate from the studio trace.
They use no model, memory store, or external dependency. The harness keeps
its existing one-step memory update; the Q-learning target here is a contrast.

Run from the repository root:
    python chapters/ch05/08_rl_foundations.py
"""

from __future__ import annotations


def discounted_returns(rewards: list[float], gamma: float) -> list[float]:
    returns = []
    total = 0.0
    for reward in reversed(rewards):
        total = reward + gamma * total
        returns.append(total)
    return list(reversed(returns))


def value_update(q: float, target: float, alpha: float) -> float:
    return q + alpha * (target - q)


def q_learning_target(
    reward: float, next_values: list[float], gamma: float, terminal: bool
) -> float:
    # A terminal transition has no future return to bootstrap from.
    future = 0.0 if terminal else max(next_values)
    return reward + gamma * future


def main() -> None:
    rewards = [-0.1, -0.1, 1.0]
    returns = discounted_returns(rewards, gamma=0.9)
    print("Illustrative arithmetic; not support-agent measurements")
    print("Reward and return (gamma=0.9)")
    for step, (reward, total) in enumerate(zip(rewards, returns)):
        print(f"  t={step}: reward={reward:+.2f}; return={total:.3f}")

    print("\nSame rewards [1, 1, 0], different learning rates")
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
