"""Quick validation of the revenue substrate: genesis should miss the gotchas and
score low, the oracle should pin them and score 1.0, and both should be
deterministic at temperature 0 (no LLM in the tool loop)."""
import asyncio
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agents.revenue import (
    POLICY_GENESIS, POLICY_ORACLE, TASK_REQUEST, build_revenue_agent,
    reconstruct_answer, score_revenue,
)


def out(s):
    sys.stdout.buffer.write((str(s) + "\n").encode("ascii", "replace")); sys.stdout.flush()


async def run(policy):
    agent = build_revenue_agent(policy, temperature=0.0)
    _, traj = await agent.run(TASK_REQUEST)
    ans = reconstruct_answer(traj)
    return score_revenue(ans), ans


async def main():
    for label, pol in [("GENESIS", POLICY_GENESIS), ("ORACLE", POLICY_ORACLE)]:
        for i in range(2):
            sc, ans = await run(pol)
            out(f"{label} run{i + 1}: score={sc:.2f} answer={ans}")


if __name__ == "__main__":
    asyncio.run(main())
