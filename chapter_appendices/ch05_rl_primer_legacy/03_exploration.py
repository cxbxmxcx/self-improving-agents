"""Chapter 5, section 5.2: one-step value learning on a bandit, no LLM.

Thorndike's law of effect gives the direction: strengthen what worked.
Value learning adds the bookkeeping: a number per action that tracks how much
reward followed it, updated by q <- q + alpha * (r - q) every time the
action runs. This primer plays that arithmetic on the smallest problem
that matters to this chapter: three memory entries, each with a hidden
true helpfulness, and an agent that must learn which one deserves to be
recalled.

Recalling an entry is pulling an arm. The reward is whether the run it
shaped succeeded, which the agent only observes noisily. Epsilon-greedy
selection recalls the best-known entry most of the time and a random one
occasionally, because an estimate you never revisit can never be fixed.

Deterministic (seeded), instant, and model-free in both senses of the
word: there is no LLM here, and the value update needs no model of the
environment either. General Q-learning's next-state term is omitted.

Run:

    python chapters/ch05/03_exploration.py
"""
from __future__ import annotations

import random

# Three past interactions sit in memory. Their true helpfulness is hidden
# from the agent; only the outcomes of runs that recalled them show it.
ENTRIES = {
    "wired-profile answer": 0.9,   # a past answer that genuinely solves the problem
    "reboot-the-laptop answer": 0.5,  # generic advice; sometimes coincides with success
    "wrong-VPN answer": 0.1,       # confidently wrong; runs that recall it tend to fail
}

ALPHA = 0.1      # learning rate: how far each outcome moves the estimate
EPSILON = 0.1    # probability of choosing randomly, possibly the best entry
ROUNDS = 300
SEED = 7


def main() -> None:
    rng = random.Random(SEED)
    q = {name: 0.5 for name in ENTRIES}      # every entry starts unproven, not distrusted
    recalls = {name: 0 for name in ENTRIES}

    for step in range(1, ROUNDS + 1):
        # The recall policy: usually the best-known entry, sometimes a random one.
        if rng.random() < EPSILON:
            name = rng.choice(list(ENTRIES))
        else:
            name = max(q, key=q.get)
        recalls[name] += 1

        # The environment answers: did the run that recalled this entry succeed?
        reward = 1.0 if rng.random() < ENTRIES[name] else 0.0

        # The law of effect as arithmetic: move the estimate toward the outcome.
        q[name] = q[name] + ALPHA * (reward - q[name])

        if step in (1, 10, 50, 150, ROUNDS):
            row = "  ".join(f"{n.split()[0]:>14} {q[n]:.2f}" for n in ENTRIES)
            print(f"round {step:>3}:  {row}")

    print("\nAfter 300 rounds:")
    for name, truth in ENTRIES.items():
        print(
            f"  {name:<26} true {truth:.1f}   learned {q[name]:.2f}"
            f"   recalled {recalls[name]:>3}x"
        )
    best = max(q, key=q.get)
    print(f"\nHighest final estimate in this run: '{best}'.")
    print("Compare values with recall counts; a high estimate can rest on little evidence.")


if __name__ == "__main__":
    main()
