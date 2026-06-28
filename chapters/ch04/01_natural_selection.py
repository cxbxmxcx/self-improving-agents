"""Chapter 4: natural selection in twelve lines (no LLM, no cost).

Evolution climbs with three dumb moves repeated: vary, select, and inherit. This
loop evolves integers toward a hidden target using nothing else, so you can watch
adaptation appear from variation and selection alone.

Map it onto the framework and the biology turns into code you already have:
variation is mutate(), heredity is the parent each child copies, selection is the
Signal (here, fitness), and a generation is one search round. Run:

    python chapters/ch04/01_natural_selection.py
"""
from __future__ import annotations

import random

TARGET = 73  # the hidden "environment" the population must adapt to


def fitness(genome: int) -> int:
    """Closeness to the target; higher is better, 0 is perfect. This is the Signal."""
    return -abs(genome - TARGET)


def mutate(genome: int) -> int:
    """Variation: a child differs from its parent by one step."""
    return genome + random.choice((-1, 1))


def main() -> None:
    random.seed(0)  # reproducible for the book
    population = [random.randint(0, 100) for _ in range(8)]

    for generation in range(40):
        population.sort(key=fitness, reverse=True)
        survivors = population[:4]                       # selection: keep the top half
        children = [mutate(g) for g in survivors]        # variation + heredity
        population = survivors + children                # the next generation
        best = max(population, key=fitness)
        print(f"gen {generation:>2}: best={best:>3}  fitness={fitness(best)}")
        if fitness(best) == 0:
            print(f"\nReached the target {TARGET} by variation and selection alone.")
            break


if __name__ == "__main__":
    main()
