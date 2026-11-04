"""Listing 5.1: discounted return, with no external dependencies."""

def discounted_returns(rewards: list[float], gamma: float) -> list[float]:
    returns = []
    total = 0.0
    for reward in reversed(rewards):
        total = reward + gamma * total
        returns.append(total)
    return list(reversed(returns))


def main() -> None:
    rewards = [-0.1, -0.1, 1.0]
    returns = discounted_returns(rewards, gamma=0.9)
    print("Reward and return (gamma=0.9)")
    for step, (reward, total) in enumerate(zip(rewards, returns)):
        print(f"  t={step}: reward={reward:+.2f}; return={total:.3f}")


if __name__ == "__main__":
    main()
