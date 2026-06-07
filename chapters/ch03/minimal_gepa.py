"""Chapter 3 §3.2 (RAG agent example) — GEPA from scratch.

GEPA (Agrawal et al., ICLR 2026) is population-based reflective mutation
with Pareto selection. This file builds the three distinctive ideas by
hand so you see them before the framework's helix.search.gepa.GEPA wraps
them:

  1. A population of N candidates evolves together.
  2. Reflective mutation: the proposer reads what the agent DID with a
     prompt (the trajectory) and edits to fix the observed failure, rather
     than rewriting blind.
  3. Pareto selection over two objectives (quality + token cost): a
     candidate that is better on one axis and worse on another is not
     dominated; both survive.

Run:
    python chapters/ch03/minimal_gepa.py

You'll watch three generations. Each generation's Pareto front holds the
candidates that are best at *something* — high quality, low cost, or a
balance. Single-objective search keeps one winner; GEPA keeps the front.

Cost: ~$1 on Haiku + Sonnet with the default knobs.
"""

from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass
from pathlib import Path

import litellm

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from helix.env import load_env

load_env()


AGENT_MODEL = "claude-haiku-4-5"
PROPOSER_MODEL = "claude-sonnet-4-6"
POPULATION_SIZE = 4
GENERATIONS = 3
QUESTIONS_PER_EVAL = 2  # small slice; raise for a stronger signal at higher cost

GENESIS_PROMPT = (
    "You are a research assistant. Answer the user's question accurately "
    "and cite sources when possible."
)

# A tiny eval slice, same shape as Ch 2's 04_minimal_spo. Hardcoded here so the
# script is self-contained.
EVAL = [
    {
        "question": "What does the 2017 'Attention Is All You Need' paper introduce?",
        "reference": "The Transformer architecture, based entirely on attention, "
                     "replacing recurrence and convolutions.",
    },
    {
        "question": "What four memory operators does ExpeL use?",
        "reference": "ADD, UPVOTE, DOWNVOTE, EDIT.",
    },
]


# --------------------------- the agent ---------------------------


async def answer(prompt: str, question: str) -> tuple[str, int]:
    """Return (answer_text, total_tokens). The token count is the cost
    objective GEPA's Pareto front trades against quality."""
    response = await litellm.acompletion(
        model=AGENT_MODEL,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": question},
        ],
    )
    text = response.choices[0].message.content or ""
    usage = getattr(response, "usage", None)
    tokens = getattr(usage, "total_tokens", 0) if usage else 0
    return text, tokens


# --------------------------- two-objective scoring ---------------------------


@dataclass
class Scored:
    prompt: str
    quality: float       # 0-1, higher is better
    tokens: int          # lower is better
    trajectory: str      # the answer text, used for reflective mutation


async def score_quality(question: str, reference: str, answer_text: str) -> float:
    """A 0-1 absolute quality judge. One LLM call."""
    response = await litellm.acompletion(
        model=PROPOSER_MODEL,
        messages=[
            {"role": "system", "content": "Rate the answer 0.0 to 1.0 against the "
             "reference. Reply with only a number."},
            {"role": "user", "content": f"Question: {question}\nReference: {reference}\n"
             f"Answer: {answer_text}\n\nScore (0.0-1.0):"},
        ],
    )
    raw = (response.choices[0].message.content or "0.5").strip()
    try:
        return max(0.0, min(1.0, float(raw.split()[0])))
    except (ValueError, IndexError):
        return 0.5


async def evaluate(prompt: str) -> Scored:
    """Run the prompt over the eval slice, average quality and tokens."""
    qualities: list[float] = []
    total_tokens = 0
    last_answer = ""
    for item in EVAL:
        text, tokens = await answer(prompt, item["question"])
        q = await score_quality(item["question"], item["reference"], text)
        qualities.append(q)
        total_tokens += tokens
        last_answer = text
    return Scored(
        prompt=prompt,
        quality=sum(qualities) / len(qualities),
        tokens=total_tokens,
        trajectory=last_answer,
    )


# --------------------------- reflective mutation ---------------------------


REFLECT_PROMPT = """You are improving a research-assistant system prompt.

You will see the current prompt and the answer it produced on a sample
question. Critique what the answer did poorly (vague? unsourced? verbose?
inaccurate?), then rewrite the prompt to fix that specific failure mode.

Keep the prompt concise — a verbose prompt costs tokens on every request.
Output only the new prompt text."""


async def reflective_mutate(scored: Scored) -> str:
    """Mutate by reflecting on the trajectory, not by blind rewrite.

    This is GEPA's distinctive operator: the proposer reads what the prompt
    actually produced and targets the observed weakness.
    """
    response = await litellm.acompletion(
        model=PROPOSER_MODEL,
        messages=[
            {"role": "system", "content": REFLECT_PROMPT},
            {"role": "user", "content": f"Current prompt:\n{scored.prompt}\n\n"
             f"An answer it produced:\n{scored.trajectory}\n\n"
             f"Critique and rewrite the prompt. Output only the new prompt."},
        ],
    )
    return (response.choices[0].message.content or scored.prompt).strip()


# --------------------------- Pareto selection ---------------------------


def dominates(a: Scored, b: Scored) -> bool:
    """a dominates b if a is at least as good on both objectives and
    strictly better on at least one. Objectives: maximize quality,
    minimize tokens."""
    at_least_as_good = a.quality >= b.quality and a.tokens <= b.tokens
    strictly_better = a.quality > b.quality or a.tokens < b.tokens
    return at_least_as_good and strictly_better


def pareto_front(population: list[Scored]) -> list[Scored]:
    """The non-dominated candidates: every member is best at something."""
    front = []
    for candidate in population:
        if not any(dominates(other, candidate) for other in population if other is not candidate):
            front.append(candidate)
    return front


# --------------------------- the GEPA loop ---------------------------


async def main_async() -> None:
    # Generation 0: seed the population from the genesis prompt by mutating
    # it POPULATION_SIZE times.
    print("Seeding initial population ...")
    seed_scored = await evaluate(GENESIS_PROMPT)
    population: list[Scored] = [seed_scored]
    for _ in range(POPULATION_SIZE - 1):
        mutated = await reflective_mutate(seed_scored)
        population.append(await evaluate(mutated))

    for gen in range(1, GENERATIONS + 1):
        print("=" * 70)
        print(f"Generation {gen}/{GENERATIONS}")
        print("=" * 70)
        for s in population:
            print(f"  quality={s.quality:.2f}  tokens={s.tokens:<5}  "
                  f"{s.prompt[:60]}{'...' if len(s.prompt) > 60 else ''}")

        front = pareto_front(population)
        print(f"\n  Pareto front ({len(front)} non-dominated):")
        for s in front:
            print(f"    quality={s.quality:.2f}  tokens={s.tokens}")

        if gen == GENERATIONS:
            break

        # Next generation: each front member spawns offspring by reflective
        # mutation. Refill the population to POPULATION_SIZE from the front.
        print(f"\n  Spawning generation {gen + 1} from the front ...")
        offspring: list[Scored] = list(front)  # carry the front forward (elitism)
        i = 0
        while len(offspring) < POPULATION_SIZE:
            parent = front[i % len(front)]
            child_prompt = await reflective_mutate(parent)
            offspring.append(await evaluate(child_prompt))
            i += 1
        population = offspring[:POPULATION_SIZE]
        print()

    print()
    print("=" * 70)
    print("Final Pareto front — pick the candidate that fits your cost budget:")
    print("=" * 70)
    for s in pareto_front(population):
        print(f"  quality={s.quality:.2f}  tokens={s.tokens}")
        print(f"    {s.prompt[:120]}{'...' if len(s.prompt) > 120 else ''}")
        print()


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
