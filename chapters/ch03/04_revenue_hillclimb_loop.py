"""Chapter 3: hill-climbing on the revenue task (the Karpathy ratchet).

This is the simplest search that consumes a measurement: rewrite the prompt,
keep the change only if the measured score went up, repeat. It is the Godel
gate from 01_godel_gate.py turned into a loop, with proof relaxed to a number,
and it is the search Karpathy autoresearch made famous (Tobi Lutke's reported
19 percent overnight run is the same single-thread ratchet).

Hill-climbing reads an ABSOLUTE score, which is exactly what the ground-truth
signal gives us. That is the difference from SPO: SPO accepts on a pairwise
judge's preference and needs no ground truth, while hill-climbing accepts on
the score itself. The loop and the task are identical; only the Search and the
accept rule change. Run:

    python chapters/ch03/04_revenue_hillclimb_loop.py
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
from helix.search.base import SearchBudget, Variant
from helix.search.hillclimb import HillClimb
from helix.signal import Cost, GapMeasurement

from agents.revenue import (
    POLICY_GENESIS, TASK_REQUEST, build_revenue_agent, build_revenue_policy,
    reconstruct_answer, score_smooth_with_feedback,
)

load_env()

AGENT_MODEL = "claude-sonnet-4-5"      # runs the task; temperature 0 makes scoring deterministic
PROPOSER_MODEL = "claude-sonnet-4-5"   # mid-tier on purpose: it cannot hold all rules at once
ROUNDS = 10
ARCHIVE_PATH = Path(__file__).parent / "runs" / "revenue_hillclimb.sqlite"

# The proposer sees the current prompt, its score, and the scorer's feedback (which
# rep is too high or too low), but not the schema and not which rule is missing. It
# can see that the totals are wrong, not why, so a single climber cannot localize the
# fix. Hill-climbing consumes an absolute score where SPO consumes a judge's
# preference; the signal type differs, but on this task both are equally thin.
PROPOSER_PROMPT = (
    "You are improving the system prompt of a data-analyst agent that answers "
    "questions by chaining query tools over a company database.\n\n"
    f"The agent must answer: \"{TASK_REQUEST}\"\n\n"
    "You will see the current system prompt, its score, and feedback from a scorer "
    "that compared the agent's answer to the ground truth: which leaderboard "
    "positions are wrong and whether each total is too high or too low (never the "
    "correct numbers). Revise the prompt to raise the score. Do NOT hard-code answer "
    "numbers.\n\n"
    "Output only the new system prompt. No preamble, no markdown."
)


async def measure(content: str) -> tuple[float, str]:
    """Run the agent with this system prompt and score its leaderboard against the
    ground truth, returning (score, feedback). The score is the absolute number the
    hill-climb ratchets on."""
    agent = build_revenue_agent(content, model=AGENT_MODEL, temperature=0.0)
    _, trajectory = await agent.run(TASK_REQUEST)
    return score_smooth_with_feedback(reconstruct_answer(trajectory))


def _fresh_archive() -> SQLiteArchive:
    for ext in ("", "-wal", "-shm"):
        p = Path(str(ARCHIVE_PATH) + ext)
        if p.exists():
            p.unlink()
    ARCHIVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    return SQLiteArchive(ARCHIVE_PATH)


async def improve(search, rounds: int) -> tuple[str, float]:
    """The improvement loop. Swapping HillClimb for SPO, GEPA, or DGM changes only
    the `search` object; the loop is identical."""
    archive = _fresh_archive()
    seed = build_revenue_policy(POLICY_GENESIS)

    best_content, (best_score, best_fb) = POLICY_GENESIS, await measure(POLICY_GENESIS)
    print(f"genesis: score={best_score:.2f}  ({best_fb})")
    # Record the genesis with its score so the lineage tree has a ranked root.
    await archive.record(
        Variant(artifact=seed, parent=seed, search_method="human"),
        GapMeasurement(score=best_score, feedback=best_fb),
    )

    budget = SearchBudget(max_candidates=rounds)
    # HillClimb yields one candidate at a time and reads variant.measurement.score on
    # the next iteration: it keeps the candidate only if the score strictly improved.
    # The absolute score IS the signal, so there is no pairwise judge here.
    async for variant in search.propose(seed=seed, signal=None, archive=archive, budget=budget):
        score, fb = await measure(variant.artifact.content)
        variant.measurement = GapMeasurement(score=score, feedback=fb,
                                              confidence=score, cost=Cost())
        # Record with lineage + measurement so the dashboard can color and rank it.
        await archive.record(variant, variant.measurement)
        marker = "   <- new best" if score > best_score else ""
        if score > best_score:
            best_content, best_score, best_fb = variant.artifact.content, score, fb
        print(f"round {variant.metadata.get('round', '?')}: candidate={score:.2f}  best={best_score:.2f}{marker}  ({fb})")

    print(f"\nbest score={best_score:.2f}\nbest prompt:\n{best_content}")
    return best_content, best_score


async def main_async() -> None:
    search = HillClimb(proposer_model=PROPOSER_MODEL, proposer_prompt=PROPOSER_PROMPT,
                       rounds=ROUNDS)
    await improve(search, ROUNDS)


if __name__ == "__main__":
    asyncio.run(main_async())
