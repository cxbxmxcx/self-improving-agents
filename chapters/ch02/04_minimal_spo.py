"""Chapter 2 §2.3 — SPO from scratch.

What you read here is what the framework's `helix.search.spo.SPO` does,
in 80 lines. The framework version adds caching, parent-pointer
bookkeeping, observability, and budget enforcement. The algorithm is the
same three steps you'll see below: mutate, score, accept.

SPO is self-supervised: there are no reference answers and no labels. The
only signal is a pairwise comparison of two prompts' outputs, judged against
a rubric that says what a good answer looks like. That is what lets SPO
improve a prompt with nothing but a set of questions.

Run:
    python chapters/ch02/04_minimal_spo.py

What you'll see, for three rounds:
  - A mutated candidate prompt (with the judge's last feedback used as
    guidance, just like SPO).
  - The candidate and the current best each answer the same questions.
  - The rubric judge's verdict per question: candidate wins, current best
    wins, or tie.
  - Accept on win (candidate becomes the new current best); reject otherwise.

This is SPO: hill-climbing search guided by a self-supervised pairwise judge.
Section §2.4 introduces the framework version that adds production plumbing
around the same loop.
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

load_env()


AGENT_MODEL = "claude-haiku-4-5"
PROPOSER_MODEL = "claude-sonnet-4-6"
JUDGE_MODEL = "claude-sonnet-4-6"
ROUNDS = 3
QUESTIONS_PER_ROUND = 8  # enough questions that decisive wins separate from ties

# Just questions, no answers. SPO is self-supervised: the signal comes from
# comparing two prompts' outputs against the rubric, never from a labeled key.
# Each question needs a complete, multi-part answer, so a prompt that explains
# fully beats one that answers in a single terse sentence. That is the gap SPO
# climbs.
QUESTIONS = [
    "Explain the difference between weather and climate.",
    "Why does the sky appear blue during the day?",
    "Describe the main stages of the water cycle.",
    "What causes the seasons on Earth?",
    "Explain the difference between a virus and a bacterium.",
    "Why does ice float on water?",
    "How do vaccines help the immune system?",
    "Summarize the plot of Romeo and Juliet in two or three sentences.",
]


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


# --------------------------- the self-supervised judge ---------------------------


# The rubric is the single shared objective: what a good answer looks like. The
# judge uses it to compare two answers; the proposer (below) uses the same text
# to know what to optimize toward. It is a rubric, not an answer key, which is
# what keeps SPO self-supervised. Defining it once means the two uses can never
# drift apart.
RUBRIC = """1. Accuracy: correct, with no invented or misleading claims.
2. Completeness: covers the key points the question calls for.
3. Clarity: well organized and easy to follow.
4. Honesty: acknowledges uncertainty instead of guessing."""

JUDGE_SYSTEM = f"""You are comparing two answers, LEFT and RIGHT, to the same question,
and deciding which one is better.

Judge on this rubric, in order of importance:
{RUBRIC}

If the two answers are equally good, reply TIE.

Reply as JSON: "winner" ("LEFT", "RIGHT", or "TIE") and "feedback" (one sentence)."""


async def judge_once(question: str, left: str, right: str) -> dict:
    """One rubric-based pairwise judge call. No reference answer."""
    user_prompt = f"""Question:
{question}

Answer LEFT:
{left}

Answer RIGHT:
{right}

Which answer is better? Reply as JSON with "winner" and "feedback"."""
    response = await litellm.acompletion(
        model=JUDGE_MODEL,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content or "{}"
    try:
        return json.loads(raw)
    except Exception:
        return {"winner": "TIE", "feedback": "judge output malformed"}


async def judge_with_swap_and_agree(question: str, answer_a: str, answer_b: str) -> dict:
    """Run the rubric judge both ways and require agreement; disagreement = TIE.

    This is the swap-and-agree idea from section 2.2, minus the reference: it
    suppresses position bias without ever consulting a labeled answer.
    """
    forward = await judge_once(question, left=answer_a, right=answer_b)
    reverse = await judge_once(question, left=answer_b, right=answer_a)
    flip = {"LEFT": "RIGHT", "RIGHT": "LEFT", "TIE": "TIE"}
    reverse_remapped = flip.get(reverse.get("winner"), "TIE")
    forward_winner = forward.get("winner", "TIE")
    if forward_winner == reverse_remapped and forward_winner != "TIE":
        return {"winner": forward_winner, "feedback": forward.get("feedback", ""), "agreed": True}
    return {"winner": "TIE",
            "feedback": f"swap-disagreement: forward={forward_winner}, reverse={reverse_remapped}",
            "agreed": False}


# --------------------------- the SPO mutation step ---------------------------


MUTATION_PROMPT = """You are improving the system prompt for an AI assistant.

The assistant's answers are judged on this rubric, in order of importance:
{rubric}

Current prompt:
---
{current}
---

The pairwise judge most recently said this about the prompt's output:
---
{feedback}
---

Rewrite the system prompt so the assistant scores better against that rubric.
Output only the new prompt text, nothing else."""


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
                    rubric=RUBRIC,
                    current=current_prompt,
                    feedback=last_feedback or "(no feedback yet; this is the first round)",
                ),
            },
        ],
    )
    return (response.choices[0].message.content or "").strip()


# --------------------------- the SPO loop ---------------------------


GENESIS_PROMPT = (
    # Deliberately flawed so the first mutation has room to win: it forces
    # terse, one-sentence answers with no explanation, which the rubric judge
    # marks down for incompleteness on questions that need a full answer.
    "You are an assistant. Answer every question in a single short sentence, "
    "and never add explanation, detail, or reasoning."
)


async def score_candidate_vs_current(
    candidate_prompt: str,
    current_prompt: str,
    questions: list[str],
) -> tuple[int, int, int, str]:
    """Run both prompts on the questions and let the rubric judge pick a winner.

    No reference answers anywhere: the candidate's output is compared only
    against the current best's output, judged by the rubric. Returns
    (wins, losses, ties, last_feedback). 'wins' counts questions where the
    candidate beat the current best.
    """
    wins = losses = ties = 0
    last_feedback = ""
    for question in questions:
        cand_answer = await answer_with_prompt(candidate_prompt, question)
        curr_answer = await answer_with_prompt(current_prompt, question)
        verdict = await judge_with_swap_and_agree(
            question,
            answer_a=cand_answer,  # LEFT  = candidate
            answer_b=curr_answer,  # RIGHT = current best
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
    questions = QUESTIONS[:QUESTIONS_PER_ROUND]
    print(f"Loaded {len(questions)} questions (no reference answers).")
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

        # 2. Score the candidate against the current best with the rubric judge.
        print(f"Judging on {len(questions)} questions ...")
        wins, losses, ties, last_feedback = await score_candidate_vs_current(
            candidate_prompt=candidate_prompt,
            current_prompt=current_prompt,
            questions=questions,
        )
        print(f"  candidate: {wins} wins, {losses} losses, {ties} ties")
        print(f"  judge said: {last_feedback[:200]}")
        print()

        # 3. Accept on win (more wins than losses). Reject otherwise.
        if wins > losses:
            print(f"  [ACCEPT] candidate beat the current best; promoting it.")
            current_prompt = candidate_prompt
        else:
            print(f"  [REJECT] candidate did not beat the current best; keeping it.")
        print()

    print("=" * 70)
    print("Final prompt:")
    print("=" * 70)
    print(current_prompt)


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
