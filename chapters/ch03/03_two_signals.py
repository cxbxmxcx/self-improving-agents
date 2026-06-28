"""Chapter 3: two signals on one trajectory, and the ceiling between them.

A signal reports the gap between what the agent did and good, and every signal
returns the same shape, a GapMeasurement. This script runs the revenue agent
once on the genesis prompt and then measures that single run two ways, so you
can see two members of the signal family fill two different fields of the same
object.

The ground-truth signal fills `score`: a number that says how wrong the answer
is, the symptom. The reflection signal fills `feedback`: a sentence that says
which rule the agent never applied, the diagnosis. Neither one alone is enough
to drive a confident fix, and that gap is the signal ceiling. Run:

    python chapters/ch03/03_two_signals.py
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# A reflection critique can contain characters the Windows console encoding does
# not cover, so print as UTF-8 and never crash on an exotic glyph.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from helix.env import load_env
from helix.signals.reflection import Reflection

from agents.revenue import (
    POLICY_GENESIS, TASK_REQUEST, build_revenue_agent, build_revenue_policy,
    reconstruct_answer, score_smooth,
)

load_env()

AGENT_MODEL = "claude-sonnet-4-5"      # temperature 0 so the run is reproducible
REFLECTOR_MODEL = "claude-sonnet-4-5"

# The default reflector prompt is tuned for a RAG agent (it asks about retrieval
# and citations), so we give the reflection signal a revenue-specific brief. The
# point of reflection is a concrete diagnosis, so we name the rules it should
# check for and forbid vague advice.
REVENUE_REFLECTION_PROMPT = (
    "You are critiquing a data-analyst agent's run on a revenue-leaderboard task. "
    "You see the agent's system prompt and the exact tool calls it made. Work out "
    "which business rules it omitted: did it filter order status (exclude cancelled "
    "and returned), channel (exclude internal), and payment (exclude pending)? Did "
    "it subtract the refund column? Did it pick the quarter relative to today and "
    "group by rep? Reply with a brief plain-text list of only the rules the agent "
    "missed, one per line. Do not add markdown headers, a code block, or any other "
    "commentary; name the missing rules and nothing else."
)


async def main_async() -> None:
    # One run of the agent on the genesis prompt produces one trajectory.
    agent = build_revenue_agent(POLICY_GENESIS, model=AGENT_MODEL, temperature=0.0)
    _, trajectory = await agent.run(TASK_REQUEST)

    # Ground-truth signal: the bare score, with no per-rep notes, so the contrast
    # stays honest. This is the symptom: a number, not a cause.
    gt_score = score_smooth(reconstruct_answer(trajectory))

    # Reflection signal: reads the trajectory and returns a critique in the
    # `feedback` field, with `score` left None.
    reflection = Reflection(model=REFLECTOR_MODEL, prompt=REVENUE_REFLECTION_PROMPT)
    candidate = build_revenue_policy(POLICY_GENESIS)
    refl = await reflection.measure(candidate=candidate, trajectory=trajectory,
                                    ground_truth={"question": TASK_REQUEST})

    print("=" * 70)
    print("ONE trajectory, TWO signals, ONE shape (GapMeasurement)")
    print("=" * 70)
    print()
    print("Ground truth  (fills `score`, leaves `feedback` empty)")
    print(f"  score    = {gt_score:.2f}")
    print(f"  feedback = {refl_or_none(None)}")
    print("  -> the symptom: how wrong the answer is, but not why")
    print()
    print("Reflection    (fills `feedback`, leaves `score` empty)")
    print(f"  score    = {refl_or_none(refl.score)}")
    print(f"  feedback = {refl.feedback.strip()}")
    print("  -> the diagnosis: which rules were missed, but not how wrong")
    print()
    print("Same object, different fields filled. The score has a number and no")
    print("diagnosis; the critique has a diagnosis and no number. Neither alone is")
    print("enough to drive a confident fix, and that gap is the signal ceiling.")


def refl_or_none(value) -> str:
    return "None" if value is None else f"{value:.2f}"


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
