"""Chapter 3 §3.2 (Path A) — GEPA from scratch on a travel tool description.

The same GEPA you would build by hand: a population of candidate descriptions, a
reflective mutation that reads what failed and proposes a fix, and a Pareto front
over two objectives. Here the artifact is the `search_flights` tool description
and the quality objective is ground truth: TaskSuccessSignal runs the travel
agent against each scenario and scores the booked trip deterministically. The
second objective is description length (a stand-in for prompt cost), so the front
keeps the shortest description at each quality level.

Nothing here imports helix.search.gepa: the point is to see the algorithm naked
before the framework wrapper (chapters/ch03/travel_tool_optimization.py runs the
framework GEPA on the same artifact).

Run:
    python chapters/ch03/gepa_travel_from_scratch.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import litellm

from helix.artifact import Artifact
from helix.env import load_env
from helix.signals.task_success import TaskSuccessSignal

from agents.travel import (
    genesis_descriptions,
    load_travel_scenarios,
    make_score_scenario,
)

load_env()

# Configuration -----------------------------------------------------------
PROPOSER_MODEL = "claude-sonnet-4-6"
AGENT_MODEL = "claude-haiku-4-5"
TARGET_DESCRIPTION_ID = "prompt.tool.search_flights.description"
POPULATION = 4
GENERATIONS = 3
MAX_SCENARIOS = 5
SCENARIOS_PATH = Path(__file__).parent / "travel_scenarios.json"


def _cost(desc: Artifact) -> int:
    """Cheap second objective: prompt length stands in for token cost."""
    return len(desc.content) if isinstance(desc.content, str) else 0


async def _reflective_mutate(parent: Artifact, feedback: str) -> Artifact:
    """One LLM call: critique what the description let the agent miss, rewrite it."""
    system = (
        "You write the description for a `search_flights` tool that an LLM agent "
        "reads to decide how to call it. The tool accepts origin, destination, "
        "date, nonstop (bool), and max_price (int). Rewrite the description so the "
        "agent reliably passes every constraint the user states. Return only the "
        "new description text, one or two sentences, no preamble."
    )
    user = f"Current description:\n{parent.content}\n\nWhat went wrong:\n{feedback}"
    resp = await litellm.acompletion(
        model=PROPOSER_MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
    )
    new_text = (resp.choices[0].message.content or "").strip()
    return parent.mutate(new_text or parent.content, created_by="gepa_reflective")


def _failure_feedback(measurement) -> str:
    """Summarize the lowest-scoring scenarios for the reflective mutator."""
    per = sorted(measurement.metadata.get("per_scenario", []), key=lambda p: p["score"])
    lines = []
    for p in per[:3]:
        booked = p.get("booked", {})
        flight = (booked.get("flight") or {}).get("id", "none")
        lines.append(f"- scenario {p['scenario']} scored {p['score']:.2f}; booked flight {flight}")
    return "\n".join(lines) or "no detail"


def _pareto_front(scored: list[tuple[Artifact, float, int]]) -> list[tuple[Artifact, float, int]]:
    """Keep candidates not dominated on (task_success up, cost down)."""
    front = []
    for cand in scored:
        _, q, c = cand
        dominated = any(
            (oq >= q and oc <= c) and (oq > q or oc < c)
            for _, oq, oc in scored
        )
        if not dominated:
            front.append(cand)
    return front


async def main_async() -> None:
    scenarios = load_travel_scenarios(SCENARIOS_PATH)
    base = genesis_descriptions()
    score_scenario = make_score_scenario(base, model=AGENT_MODEL)
    signal = TaskSuccessSignal(scenarios, score_scenario, max_scenarios=MAX_SCENARIOS)

    async def score(desc: Artifact) -> tuple[Artifact, float, int]:
        m = await signal.measure(desc)
        return desc, m.score or 0.0, _cost(desc)

    # Generation 0: the genesis description plus reflective variants of it.
    seed = base[TARGET_DESCRIPTION_ID]
    seed_scored = await score(seed)
    feedback = _failure_feedback(await signal.measure(seed))
    population = [seed]
    for _ in range(POPULATION - 1):
        population.append(await _reflective_mutate(seed, feedback))

    for gen in range(GENERATIONS):
        scored = [await score(d) for d in population]
        front = _pareto_front(scored)
        print(f"\n=== generation {gen} : Pareto front ({len(front)} of {len(scored)}) ===")
        for desc, q, c in sorted(front, key=lambda x: -x[1]):
            print(f"  v{desc.version}  task_success={q:.2f}  cost={c:4d}  {desc.content[:70]!r}")

        # Next generation: reflectively mutate each front member.
        if gen < GENERATIONS - 1:
            next_pop: list[Artifact] = [d for d, _, _ in front]
            for desc, _, _ in front:
                fb = _failure_feedback(await signal.measure(desc))
                next_pop.append(await _reflective_mutate(desc, fb))
            population = next_pop[: POPULATION * 2]

    print("\nThe front trades task success against description length. Pick the")
    print("shortest description that hits the success you need, then promote it.")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
