"""Chapter 3 §3.3 (Path A) — DGM from scratch on a travel tool description.

The Darwin Goedel Machine keeps an archive instead of a population: every
candidate ever produced stays, and the next seed is sampled by quality and by
how rarely its branch has been picked, so a late round can fork from an early
high-scorer. Here the artifact is the `search_flights` description and quality is
ground truth: TaskSuccessSignal runs the travel agent on the scenarios and scores
the booked trips deterministically.

Nothing here imports helix.search.dgm; the framework DGMSearch (with a pluggable
mutator) is what travel_tool_optimization.py uses. This script shows the archive
and the quality-diversity sampler naked.

Run:
    python chapters/ch03/dgm_travel_from_scratch.py
"""

from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass
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
ROUNDS = 8
MAX_SCENARIOS = 5
SCENARIOS_PATH = Path(__file__).parent / "travel_scenarios.json"


@dataclass
class Entry:
    artifact: Artifact
    score: float
    parent_version: int | None
    times_sampled: int = 0


class Archive:
    """Every variant ever produced, kept forever with its lineage."""

    def __init__(self) -> None:
        self.entries: list[Entry] = []

    def add(self, entry: Entry) -> None:
        self.entries.append(entry)

    def best(self) -> Entry:
        return max(self.entries, key=lambda e: e.score)

    def sample_quality_diversity(self) -> Entry:
        """Pick a seed weighted by score, halved each time it is re-sampled, so
        the search keeps returning to good-but-neglected branches."""
        weights = [max(e.score, 0.01) / (2 ** e.times_sampled) for e in self.entries]
        # Deterministic argmax of the weight (no RNG so runs are reproducible).
        idx = max(range(len(self.entries)), key=lambda i: weights[i])
        self.entries[idx].times_sampled += 1
        return self.entries[idx]


MUTATE_SYSTEM = (
    "You write the description for a `search_flights` tool that an LLM agent "
    "reads to decide how to call it. It accepts origin, destination, date, "
    "nonstop (bool), and max_price (int). Rewrite the sampled description so the "
    "agent reliably passes every constraint a user states. The archive keeps "
    "every variant, so a bold rewrite that explores a new angle is welcome. "
    "Return only the new description text, one or two sentences."
)


async def main_async() -> None:
    scenarios = load_travel_scenarios(SCENARIOS_PATH)
    base = genesis_descriptions()
    score_scenario = make_score_scenario(base, model=AGENT_MODEL)
    signal = TaskSuccessSignal(scenarios, score_scenario, max_scenarios=MAX_SCENARIOS)
    version = 1  # genesis is v1; each candidate gets a unique next version

    async def evaluate(desc: Artifact) -> float:
        m = await signal.measure(desc)
        return m.score or 0.0

    async def mutate(seed: Entry) -> Artifact:
        nonlocal version
        resp = await litellm.acompletion(
            model=PROPOSER_MODEL,
            messages=[
                {"role": "system", "content": MUTATE_SYSTEM},
                {"role": "user", "content": f"Sampled description (task success "
                 f"{seed.score:.2f}):\n{seed.artifact.content}"},
            ],
        )
        text = (resp.choices[0].message.content or "").strip()
        version += 1
        return seed.artifact.mutate(text or seed.artifact.content,
                                    created_by="dgm_mutation", version=version)

    archive = Archive()
    seed = base[TARGET_DESCRIPTION_ID]
    archive.add(Entry(artifact=seed, score=await evaluate(seed), parent_version=None))
    print(f"genesis v1 task success: {archive.entries[0].score:.2f}")

    for r in range(1, ROUNDS + 1):
        picked = archive.sample_quality_diversity()
        candidate = await mutate(picked)
        cand_score = await evaluate(candidate)
        archive.add(Entry(artifact=candidate, score=cand_score,
                          parent_version=picked.artifact.version))
        print(f"round {r:>2}: forked from v{picked.artifact.version} "
              f"(score {picked.score:.2f}) -> v{candidate.version} score {cand_score:.2f}")

    best = archive.best()
    print(f"\nbest: v{best.artifact.version}  task success {best.score:.2f}")
    print("-" * 70)
    print(best.artifact.content)
    print("\nNote which round forked from an early high-scorer rather than the")
    print("latest winner: that is the archive keeping history alive.")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
