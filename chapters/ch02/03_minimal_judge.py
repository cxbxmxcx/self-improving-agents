"""Chapter 2 §2.2 — An LLM-as-judge, hand-written.

What you read here is what the framework's `helix.signals.pairwise_judge`
SwapAndAgree(PairwiseJudge(...)) does, in 60 lines. The framework version
adds Pydantic-validated output, prompt caching, observability spans, and
cost accounting. The algorithm is identical.

Run:
    python chapters/ch02/03_minimal_judge.py

What you'll see:
  1. A pairwise verdict on two sample answers.
  2. The same two answers judged again with their positions swapped.
     Often the verdict flips. That's position bias.
  3. A swap-and-agree pass that runs the judge twice and requires
     agreement; on disagreement it returns TIE.

This is what an LLM judge is. Three sections from here SPO will call
something just like this to decide whether a candidate prompt beat the
reference.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

# Bare LiteLLM. No framework imports in this file — the point is to show
# the algorithm naked, before §2.4 introduces the production wrappers.
import litellm

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from helix.env import load_env  # picks up .env so the script runs out-of-the-box

load_env()


JUDGE_MODEL = "claude-sonnet-4-6"

RUBRIC = """You are judging which of two candidate answers to a research
question is better, against a reference answer.

Score on these axes in this order of importance:
1. Factual accuracy against the reference answer.
2. Whether the answer is grounded in retrieved sources, not invented.
3. Clarity and directness.

Reply as JSON:
  "winner": "LEFT" or "RIGHT" or "TIE"
  "feedback": one sentence explaining why
"""


async def judge_once(
    question: str,
    reference: str,
    left: str,
    right: str,
) -> dict:
    """One pairwise judge call. The verdict says which of LEFT or RIGHT wins."""
    user_prompt = f"""Question:
{question}

Reference answer:
{reference}

Candidate LEFT:
{left}

Candidate RIGHT:
{right}

Reply as JSON with two fields: "winner" and "feedback"."""
    response = await litellm.acompletion(
        model=JUDGE_MODEL,
        messages=[
            {"role": "system", "content": RUBRIC},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content or "{}"
    try:
        return json.loads(raw)
    except Exception:
        return {"winner": "TIE", "feedback": "judge output malformed"}


async def judge_with_swap_and_agree(
    question: str,
    reference: str,
    answer_a: str,
    answer_b: str,
) -> dict:
    """Run the judge twice with positions swapped; require agreement.

    On disagreement we return TIE. This suppresses position bias: judges
    that prefer whichever answer is shown first will produce inconsistent
    verdicts when the order changes.
    """
    forward = await judge_once(question, reference, left=answer_a, right=answer_b)
    reverse = await judge_once(question, reference, left=answer_b, right=answer_a)

    # After the swap, LEFT in `reverse` means answer_b won; remap.
    flip = {"LEFT": "RIGHT", "RIGHT": "LEFT", "TIE": "TIE"}
    reverse_remapped = flip.get(reverse.get("winner"), "TIE")

    forward_winner = forward.get("winner", "TIE")
    if forward_winner == reverse_remapped and forward_winner != "TIE":
        return {
            "winner": forward_winner,
            "feedback": forward.get("feedback", ""),
            "agreed": True,
        }
    return {
        "winner": "TIE",
        "feedback": f"swap-disagreement: forward={forward_winner}, reverse={reverse_remapped}",
        "agreed": False,
    }


# --------------------------- demo ---------------------------

QUESTION = "What does the 2017 'Attention Is All You Need' paper introduce?"
REFERENCE = (
    "The Transformer architecture: a sequence-to-sequence model based "
    "entirely on attention mechanisms, replacing recurrence and convolutions."
)
ANSWER_GROUNDED = (
    "The paper introduces the Transformer, a model architecture that "
    "relies entirely on attention mechanisms and dispenses with recurrence "
    "and convolutions. It uses multi-head self-attention in both encoder "
    "and decoder."
)
ANSWER_HALLUCINATED = (
    "The paper introduces BERT, a pretrained masked language model that "
    "outperformed prior recurrent networks on benchmarks like GLUE and SQuAD."
)


async def main_async() -> None:
    print("=" * 70)
    print("Demo 1: a single pairwise verdict")
    print("=" * 70)
    verdict = await judge_once(QUESTION, REFERENCE, ANSWER_GROUNDED, ANSWER_HALLUCINATED)
    print(f"  winner: {verdict.get('winner')}")
    print(f"  feedback: {verdict.get('feedback')}")
    print()

    print("=" * 70)
    print("Demo 2: position bias — same answers, swapped positions")
    print("=" * 70)
    print("Running with grounded=LEFT, hallucinated=RIGHT ...")
    v1 = await judge_once(QUESTION, REFERENCE, ANSWER_GROUNDED, ANSWER_HALLUCINATED)
    print(f"  -> {v1.get('winner')}")
    print("Running with hallucinated=LEFT, grounded=RIGHT ...")
    v2 = await judge_once(QUESTION, REFERENCE, ANSWER_HALLUCINATED, ANSWER_GROUNDED)
    print(f"  -> {v2.get('winner')}")
    # A consistent judge picks the grounded answer both times. A position-
    # biased judge picks whichever was shown first.
    print()

    print("=" * 70)
    print("Demo 3: swap-and-agree filters out the position-biased cases")
    print("=" * 70)
    agreed = await judge_with_swap_and_agree(
        QUESTION, REFERENCE, ANSWER_GROUNDED, ANSWER_HALLUCINATED,
    )
    print(f"  winner: {agreed.get('winner')}")
    print(f"  agreed: {agreed.get('agreed')}")
    print(f"  feedback: {agreed.get('feedback')}")
    print()
    print("LEFT means the grounded answer wins. TIE means the judge disagreed "
          "with itself across the swap; in that case we don't trust either run.")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
