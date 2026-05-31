"""Chapter 3 §3.3 (RAG agent example) — DGM from scratch.

The Darwin Goedel Machine (Zhang et al., 2025) is archive-evolutionary
search. Where GEPA evolves a fixed-size population that gets replaced each
generation, DGM keeps an *archive* — every variant ever produced stays —
and samples the next seed from the whole history with quality-diversity
weighting.

Three distinctive ideas, built by hand here:

  1. Archive, not population. Nothing is discarded.
  2. Quality-diversity sampling. The next seed is chosen by score AND by
     how long it's been since that branch was tried. A round-10 mutation
     may fork from a round-3 candidate.
  3. Lineage. Every entry records its parent, so the archive is a tree you
     can replay.

Run:
    python chapters/ch03/minimal_dgm.py

You'll watch ~8 rounds. Each round prints which archived candidate the
sampler picked as the seed — and you'll see it isn't always the latest
winner. That's the DGM dynamic: old high-scoring branches stay alive.

The paper applies this to code, costing ~$22K. This runs on a text prompt
for ~$1. The pattern is the same; the benchmark numbers are not.

Cost: ~$1 on Haiku + Sonnet with the default knobs.
"""

from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass, field
from pathlib import Path

import litellm

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from helix.env import load_env

load_env()


AGENT_MODEL = "claude-haiku-4-5"
PROPOSER_MODEL = "claude-sonnet-4-6"
ROUNDS = 8
QUESTIONS_PER_EVAL = 2

GENESIS_PROMPT = (
    "You are a research assistant. Answer the user's question accurately "
    "and cite sources when possible."
)

EVAL = [
    {
        "question": "What does the 2017 'Attention Is All You Need' paper introduce?",
        "reference": "The Transformer architecture, based entirely on attention.",
    },
    {
        "question": "What four memory operators does ExpeL use?",
        "reference": "ADD, UPVOTE, DOWNVOTE, EDIT.",
    },
]


# --------------------------- the archive ---------------------------


@dataclass
class Entry:
    round_added: int
    prompt: str
    score: float
    parent_round: int | None
    times_sampled: int = 0  # diversity penalty: prefer branches not picked recently


@dataclass
class Archive:
    """Every variant ever produced, kept forever. This is the DGM core."""
    entries: list[Entry] = field(default_factory=list)

    def add(self, entry: Entry) -> None:
        self.entries.append(entry)

    def sample_quality_diversity(self) -> Entry:
        """Pick a seed weighted by score, penalized by how often it's been
        sampled. High-scoring candidates that haven't been forked from
        recently rise to the top — that's the diversity half of QD.
        """
        def weight(e: Entry) -> float:
            # score dominates; each prior sampling halves the weight so the
            # search doesn't keep forking from the same branch.
            return max(e.score, 0.01) / (2 ** e.times_sampled)

        weights = [weight(e) for e in self.entries]
        import random
        chosen = random.choices(self.entries, weights=weights, k=1)[0]
        chosen.times_sampled += 1
        return chosen


# --------------------------- the agent + scoring ---------------------------


async def answer(prompt: str, question: str) -> str:
    response = await litellm.acompletion(
        model=AGENT_MODEL,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": question},
        ],
    )
    return response.choices[0].message.content or ""


async def score_quality(question: str, reference: str, answer_text: str) -> float:
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


async def evaluate(prompt: str) -> float:
    qualities = []
    for item in EVAL:
        text = await answer(prompt, item["question"])
        qualities.append(await score_quality(item["question"], item["reference"], text))
    return sum(qualities) / len(qualities)


# --------------------------- mutation ---------------------------


MUTATE_PROMPT = """You are exploring variants of a research-assistant system
prompt under archive-evolutionary search. You will see one prompt sampled
from the search history and its score. Propose a single improved variant.

Because the archive keeps every prompt, bold rewrites that explore a new
direction are valuable even if they risk scoring lower — the search can
always fork from a better ancestor later. Keep the prompt concise.

Output only the new prompt text."""


async def mutate(seed: Entry) -> str:
    response = await litellm.acompletion(
        model=PROPOSER_MODEL,
        messages=[
            {"role": "system", "content": MUTATE_PROMPT},
            {"role": "user", "content": f"Sampled prompt (score {seed.score:.2f}):\n"
             f"{seed.prompt}\n\nPropose an improved variant. Output only the new prompt."},
        ],
    )
    return (response.choices[0].message.content or seed.prompt).strip()


# --------------------------- the DGM loop ---------------------------


async def main_async() -> None:
    archive = Archive()

    # Seed the archive with the genesis prompt.
    print("Evaluating genesis prompt ...")
    genesis_score = await evaluate(GENESIS_PROMPT)
    archive.add(Entry(round_added=0, prompt=GENESIS_PROMPT, score=genesis_score, parent_round=None))
    print(f"  genesis score: {genesis_score:.2f}")
    print()

    for round_idx in range(1, ROUNDS + 1):
        # 1. Sample a seed from the whole archive (not just the latest winner).
        seed = archive.sample_quality_diversity()

        # 2. Mutate it.
        candidate_prompt = await mutate(seed)

        # 3. Score and archive the candidate.
        candidate_score = await evaluate(candidate_prompt)
        archive.add(Entry(
            round_added=round_idx,
            prompt=candidate_prompt,
            score=candidate_score,
            parent_round=seed.round_added,
        ))

        marker = "  <- forked from an OLDER branch" if seed.round_added < round_idx - 1 else ""
        print(f"Round {round_idx:2d}: sampled seed from round {seed.round_added} "
              f"(score {seed.score:.2f}) -> candidate scores {candidate_score:.2f}{marker}")

    print()
    print("=" * 70)
    print("Archive contents (note: nothing was ever discarded):")
    print("=" * 70)
    for e in sorted(archive.entries, key=lambda e: e.score, reverse=True):
        parent = f"from r{e.parent_round}" if e.parent_round is not None else "genesis"
        print(f"  r{e.round_added:2d}  score={e.score:.2f}  {parent}  "
              f"sampled {e.times_sampled}x")

    best = max(archive.entries, key=lambda e: e.score)
    print()
    print(f"Best in archive: round {best.round_added}, score {best.score:.2f}")
    print(f"  {best.prompt[:150]}{'...' if len(best.prompt) > 150 else ''}")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
