"""Chapter 3: the improvement loop with SPO on the revenue task.

One loop, four primitives: measure (Signal) -> propose (Search) -> select -> record
(Archive), repeated. SPO proposes one mutation of the system prompt per round; we
measure it deterministically against the ground-truth leaderboard, record it to the
archive with its parent pointer, and keep it as the new best if it scored higher
(hill climbing). The proposer reads the scorer's feedback each round and gradually
discovers the hidden rules, so the score climbs from the vague genesis prompt
toward the oracle.

The point of the chapter: swapping SPO for GEPA or DGM changes only the `search`
object passed to `improve()`. The loop is identical. Run:

    python chapters/ch03/revenue_spo_loop.py
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from helix.archive import SQLiteArchive
from helix.env import load_env
from helix.search.base import SearchBudget
from helix.search.spo import SPO
from helix.signal import Cost, GapMeasurement, Preference

from agents.revenue import (
    POLICY_GENESIS, TASK_REQUEST, build_revenue_agent, build_revenue_policy,
    reconstruct_answer, score_with_feedback,
)

load_env()

AGENT_MODEL = "claude-sonnet-4-5"      # runs the task; temperature 0 makes scoring deterministic
PROPOSER_MODEL = "claude-opus-4-8"     # rewrites the prompt from the scorer's feedback
ROUNDS = 6
ARCHIVE_PATH = Path(__file__).parent / "runs" / "revenue_spo.sqlite"

# SPO's proposer knows the task and sees the scorer's outcome feedback, but NOT the
# schema and NOT the agent's execution trace. It can see THAT a total is wrong, not
# WHY, so it cannot localize the missing rule and tends to make generic edits. This
# is the limitation GEPA's trajectory reflection and DGM's archive search overcome.
PROPOSER_PROMPT = (
    "You are improving the system prompt of a data-analyst agent that answers "
    "questions by chaining query tools over a company database.\n\n"
    f"The agent must answer: \"{TASK_REQUEST}\"\n\n"
    "You will see the current system prompt and feedback from a scorer that compared "
    "the agent's answer to the ground truth: which leaderboard positions are wrong "
    "and whether each total is too high or too low (never the correct numbers). "
    "Revise the prompt to fix what the feedback indicates. Do NOT hard-code answer "
    "numbers.\n\n"
    "Output only the new system prompt. No preamble, no markdown."
)


async def measure(content: str) -> tuple[float, str]:
    """Run the agent with this system prompt on the task and score its leaderboard
    against the ground truth, returning (score, feedback)."""
    agent = build_revenue_agent(content, model=AGENT_MODEL, temperature=0.0)
    _, trajectory = await agent.run(TASK_REQUEST)
    return score_with_feedback(reconstruct_answer(trajectory))


def _fresh_archive() -> SQLiteArchive:
    for ext in ("", "-wal", "-shm"):
        p = Path(str(ARCHIVE_PATH) + ext)
        if p.exists():
            p.unlink()
    ARCHIVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    return SQLiteArchive(ARCHIVE_PATH)


async def improve(search, rounds: int) -> tuple[str, float]:
    """The improvement loop. `search` is the only thing that changes between the
    SPO, GEPA, and DGM examples."""
    archive = _fresh_archive()
    seed = build_revenue_policy(POLICY_GENESIS)
    await archive.put_artifact(seed)

    best_content, (best_score, best_fb) = POLICY_GENESIS, await measure(POLICY_GENESIS)
    print(f"genesis: score={best_score:.2f}  ({best_fb})")

    budget = SearchBudget(max_candidates=rounds)
    # SPO yields one candidate at a time; we fill its measurement, which SPO reads
    # on the next iteration to keep the winner and to feed the proposer. The signal
    # argument is unused here because we score and judge the candidate ourselves.
    async for variant in search.propose(seed=seed, signal=None, archive=archive, budget=budget):
        score, fb = await measure(variant.artifact.content)
        if score > best_score:
            preference = Preference.LEFT      # candidate beat the current best -> SPO promotes it
        elif score < best_score:
            preference = Preference.RIGHT
        else:
            preference = Preference.TIE
        variant.measurement = GapMeasurement(score=score, preference=preference,
                                              feedback=fb, confidence=score, cost=Cost())
        await archive.put_artifact(variant.artifact)
        marker = ""
        if score > best_score:
            best_content, best_score, best_fb = variant.artifact.content, score, fb
            marker = "   <- new best"
        print(f"round {variant.metadata.get('round', '?')}: candidate={score:.2f}  best={best_score:.2f}{marker}  ({fb})")

    print(f"\nbest score={best_score:.2f}\nbest prompt:\n{best_content}")
    return best_content, best_score


async def main_async() -> None:
    search = SPO(proposer_model=PROPOSER_MODEL, proposer_prompt=PROPOSER_PROMPT,
                 first_round_model=PROPOSER_MODEL, rounds=ROUNDS)
    await improve(search, ROUNDS)


if __name__ == "__main__":
    asyncio.run(main_async())
