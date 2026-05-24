"""Chapter 2 §2.3 — SPO from scratch.

What you read here is what the framework's `helix.search.spo.SPO` does,
in 80 lines. The framework version adds caching, parent-pointer
bookkeeping, observability, and budget enforcement. The algorithm is the
same three steps you'll see below: mutate, score, accept.

Run:
    python chapters/ch02/minimal_spo.py

What you'll see, for three rounds:
  - The current prompt's answer on a small eval slice.
  - A mutated candidate prompt (with the judge's last feedback used as
    guidance, just like SPO).
  - The candidate's answer on the same slice.
  - The judge's verdict: candidate wins or reference wins.
  - Accept on win (candidate becomes the new reference); reject on loss.

This is SPO: hill-climbing search guided by pairwise judge feedback.
Section §2.4 introduces the framework version that adds production
plumbing around the same loop.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import litellm

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from helix.env import load_env

# Reuse the swap-and-agree judge from §2.2. This is the only `helix`
# import; the rest of the SPO algorithm is written out below.
from chapters.ch02.minimal_judge import judge_with_swap_and_agree

load_env()


AGENT_MODEL = "claude-haiku-4-5"
PROPOSER_MODEL = "claude-sonnet-4-6"
ROUNDS = 3
QUESTIONS_PER_ROUND = 3  # small slice so the demo is cheap

EVAL_QUESTIONS_PATH = Path(__file__).parent / "eval_questions.json"


# --------------------------- the agent ---------------------------


async def answer_with_prompt(system_prompt: str, question: str) -> str:
    """A trivially simple agent: one LLM call with the candidate prompt.

    The real `helixagent_v1` runs ReAct + retrieval; this stand-in keeps
    the SPO loop visually clean. SPO doesn't care what shape the agent is;
    only the answers matter to the judge.
    """
    response = await litellm.acompletion(
        model=AGENT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
    )
    return response.choices[0].message.content or ""


# --------------------------- the SPO mutation step ---------------------------


MUTATION_PROMPT = """You are improving a system prompt for a research assistant.

Current prompt:
---
{current}
---

The pairwise judge most recently said this about the prompt's output:
---
{feedback}
---

Rewrite the system prompt so the agent does better next time. Keep it
concise (one paragraph). Output only the new prompt text, nothing else."""


async def propose_mutation(current_prompt: str, last_feedback: str) -> str:
    """Ask the proposer LLM to rewrite the prompt.

    SPO conditions each mutation on the judge's feedback from the prior
    round. First round has no feedback yet, so we pass a placeholder.
    """
    response = await litellm.acompletion(
        model=PROPOSER_MODEL,
        messages=[
            {
                "role": "user",
                "content": MUTATION_PROMPT.format(
                    current=current_prompt,
                    feedback=last_feedback or "(no feedback yet — this is the first round)",
                ),
            },
        ],
    )
    return (response.choices[0].message.content or "").strip()


# --------------------------- the SPO loop ---------------------------


GENESIS_PROMPT = (
    "You are a research assistant. Answer the user's question accurately "
    "and cite sources when possible."
)


def load_eval_slice() -> list[dict]:
    """Pick a small slice of the eval set for the demo."""
    data = json.loads(EVAL_QUESTIONS_PATH.read_text(encoding="utf-8"))
    return list(data["questions"])[:QUESTIONS_PER_ROUND]


async def score_candidate_vs_reference(
    candidate_prompt: str,
    reference_prompt: str,
    questions: list[dict],
) -> tuple[int, int, int, str]:
    """Run both prompts on the eval slice and judge each pair.

    Returns (wins, losses, ties, last_feedback_string). 'wins' counts
    questions where the candidate beat the reference.
    """
    wins = losses = ties = 0
    last_feedback = ""
    for q in questions:
        cand_answer = await answer_with_prompt(candidate_prompt, q["question"])
        ref_answer = await answer_with_prompt(reference_prompt, q["question"])
        verdict = await judge_with_swap_and_agree(
            question=q["question"],
            reference=q["reference_answer"],
            answer_a=cand_answer,  # LEFT
            answer_b=ref_answer,   # RIGHT
        )
        winner = verdict.get("winner")
        if winner == "LEFT":
            wins += 1
        elif winner == "RIGHT":
            losses += 1
        else:
            ties += 1
        last_feedback = verdict.get("feedback", "") or last_feedback
    return wins, losses, ties, last_feedback


async def main_async() -> None:
    questions = load_eval_slice()
    print(f"Loaded {len(questions)} eval questions.")
    print()

    # The 'current best' starts as the genesis prompt. Each round may
    # replace it with a winning candidate.
    current_prompt = GENESIS_PROMPT
    last_feedback = ""

    for round_idx in range(1, ROUNDS + 1):
        print("=" * 70)
        print(f"Round {round_idx}/{ROUNDS}")
        print("=" * 70)

        # 1. Mutate the current prompt, conditioned on the last judge feedback.
        print("Proposing a mutation ...")
        candidate_prompt = await propose_mutation(current_prompt, last_feedback)
        print(f"Candidate prompt (first 200 chars):")
        print(f"  {candidate_prompt[:200]}{'...' if len(candidate_prompt) > 200 else ''}")
        print()

        # 2. Score the candidate against the reference.
        print(f"Judging on {len(questions)} questions ...")
        wins, losses, ties, last_feedback = await score_candidate_vs_reference(
            candidate_prompt=candidate_prompt,
            reference_prompt=current_prompt,
            questions=questions,
        )
        print(f"  candidate: {wins} wins, {losses} losses, {ties} ties")
        print(f"  judge said: {last_feedback[:200]}")
        print()

        # 3. Accept on win (more wins than losses). Reject otherwise.
        if wins > losses:
            print(f"  [ACCEPT] candidate beat reference; promoting to new current.")
            current_prompt = candidate_prompt
        else:
            print(f"  [REJECT] candidate did not beat reference; keeping current.")
        print()

    print("=" * 70)
    print("Final prompt:")
    print("=" * 70)
    print(current_prompt)


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
