"""Calibration: does a searched prompt lower the reasoning floor on chapter 4's revenue task?

Runs the chapter 4 revenue question with gpt-5.4-mini at each reasoning
setting under three prompts: the genesis prompt the searches started from,
the DGM winner from chapter 4's archive (scored 1.0 there with Sonnet), and
chapter 4's hand-written oracle that pins all six rules. Each cell runs
twice. The question it answers before chapter 6 is rebuilt: can reasoning
alone recover the six business rules, and does searched context lower the
setting the task needs?

Replays the recording by default; --record makes the paid calls.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sqlite3
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agents.revenue import (POLICY_GENESIS, POLICY_ORACLE, TASK_REQUEST, build_revenue_agent,
                            reconstruct_answer, score_revenue, score_smooth)
from chapters.ch06.assistant.harness import MODEL, PRICES
from chapters.ch06.assistant.recording import Recording, recorded_seconds, restore_seconds
from helix.eval.effort import EFFORTS, EffortSetting, reasoning_tokens
from helix.hooks import HookPoint
from helix.signals.cost import run_cost

HERE = Path(__file__).parent
RECORDED = HERE / "assistant" / "recorded" / "revenue_calibration.jsonl"
DGM_ARCHIVE = REPO_ROOT / "chapters" / "ch04" / "runs" / "revenue_dgm.sqlite"
TRIALS = 2


def dgm_winner() -> str:
    """The highest-scoring prompt in chapter 4's DGM archive."""
    db = sqlite3.connect(f"file:{DGM_ARCHIVE.as_posix()}?mode=ro", uri=True)
    content, is_json = db.execute(
        "select a.content, a.content_is_json from artifacts a join measurements m "
        "on m.artifact_id = a.artifact_id and m.version = a.version "
        "group by a.artifact_id, a.version order by max(m.score) desc, a.version limit 1").fetchone()
    return json.loads(content) if is_json else content


async def run_once(prompt: str, effort: str, trial: int) -> dict:
    agent = build_revenue_agent(prompt, model=MODEL)
    agent.hooks.register(HookPoint.PRE_MODEL, EffortSetting(effort))
    # The trial number rides in the request so each trial is its own recorded call chain.
    request = TASK_REQUEST if trial == 0 else f"{TASK_REQUEST} (run {trial + 1})"
    with recorded_seconds() as seconds:
        _, trajectory = await agent.run(request)
    restore_seconds(trajectory, seconds)
    answer = reconstruct_answer(trajectory)
    cost = run_cost(trajectory, PRICES)
    return {"exact": score_revenue(answer), "smooth": round(score_smooth(answer), 3),
            "reasoning": reasoning_tokens(trajectory), "dollars": round(cost.dollars, 5),
            "seconds": round(cost.seconds, 1), "answer": answer}


async def main(args: argparse.Namespace) -> None:
    prompts = {"genesis": POLICY_GENESIS, "dgm_winner": dgm_winner(), "oracle": POLICY_ORACLE}
    recording = Recording(args.recording, mode="record" if args.record else "replay")
    gate = asyncio.Semaphore(4)

    async def cell(name, effort, trial):
        async with gate:
            return name, effort, trial, await run_once(prompts[name], effort, trial)

    with recording.active():
        results = await asyncio.gather(*(cell(n, e, t) for n in prompts for e in EFFORTS for t in range(TRIALS)))
    table = {}
    for name, effort, trial, r in results:
        table.setdefault((name, effort), []).append(r)
    print(f"{'prompt':<12} {'effort':<7} {'exact':>11} {'smooth':>13} {'reasoning':>10} {'dollars':>8}")
    for name in prompts:
        for effort in EFFORTS:
            rs = table[(name, effort)]
            print(f"{name:<12} {effort:<7} {' '.join(f'{r['exact']:.2f}' for r in rs):>11} "
                  f"{' '.join(f'{r['smooth']:.2f}' for r in rs):>13} "
                  f"{sum(r['reasoning'] for r in rs) / len(rs):>10.0f} {sum(r['dollars'] for r in rs) / len(rs):>8.4f}")
    if args.record:
        print(f"\n{recording.live_calls} live calls, ${recording.spent(PRICES):.2f}")
    out = HERE / "runs" / "revenue_calibration.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps([{"prompt": n, "effort": e, "trial": t, **r} for n, e, t, r in results], indent=2),
                   encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--recording", type=Path, default=RECORDED)
    args = parser.parse_args()
    if args.record:
        from helix.env import load_env

        load_env()
    asyncio.run(main(args))
