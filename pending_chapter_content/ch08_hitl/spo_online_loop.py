"""Pending Ch 8 demo: online prompt improvement with auto-promote.

This script demonstrates the OnlineImprover pattern and is held here
until Chapter 8 (HITL + live feedback) is drafted, where the auto-
promotion semantics pair naturally with the chapter's HITL gates and
implicit-feedback signals. Ch 2 (the foundational improvement chapter)
covers the offline pattern only.

Where offline runs SPO against a labeled eval set and waits for a
human to promote the winner, this script wires an `OnlineImprover` to
the agent and simulates live traffic by replaying eval questions one
at a time. The improver:

  - Subscribes to the agent's SESSION_END hook.
  - Spot-checks each completed trajectory with `LiveTrajectoryJudge`
    (reference-free; rubric-based).
  - Accumulates scores in a rolling window.
  - When the rolling average drops below threshold, proposes a candidate
    via SPO and shadow-evaluates against the next few real requests.
  - If the candidate beats the reference, fires `CandidateWins(mode="online",
    auto_promote=True)`. The default promotion handler reacts by calling
    `archive.promote()`, and from the next request onward the agent serves
    the new version.

Contrast with offline:

  +----------------+----------------------+------------------------+
  |                | offline              | online                 |
  +----------------+----------------------+------------------------+
  | improver class | OfflineImprover      | OnlineImprover         |
  | trigger        | manual / scheduled   | rolling score drop     |
  | signal         | pairwise judge       | absolute rubric judge  |
  | promotion      | human via dashboard  | auto via default hook  |
  | safety         | any artifact layer   | L1/L2 only (enforced)  |
  +----------------+----------------------+------------------------+

Cost note: every spot-checked request invokes the judge LLM. Each shadow
evaluation runs the agent twice and judges both. Adjust the policy knobs
(sample_rate, rolling_window, shadow_sample) to keep the demo cheap.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from helix.agent import Agent
from helix.env import load_env
from helix.eval import load_eval_set
from helix.improvement import OnlineImprover, ImproverPolicy
from helix.observability import attach_console_renderer
from helix.search.base import SearchBudget
from helix.search.spo import SPO
from helix.signals.live_judge import LiveTrajectoryJudge

# Ch 2 listing 02's module name starts with a digit, so the v1 archive helpers
# are loaded with importlib rather than a plain `import`.
import importlib

_v1 = importlib.import_module("chapters.ch02.02_helixagent_v1")
ARCHIVE_PATH = _v1.ARCHIVE_PATH
PROMPT_ARTIFACT_ID = _v1.PROMPT_ARTIFACT_ID
get_or_create_seed = _v1.get_or_create_seed
open_archive = _v1.open_archive

from chapter_appendices.getting_started.helixagent_v0 import build_retrieve_tool

load_env()

# Configuration -----------------------------------------------------------
AGENT_MODEL = "claude-haiku-4-5"
PROPOSER_MODEL = "claude-sonnet-4-6"
JUDGE_MODEL = "claude-sonnet-4-6"
IMPROVER_ID = "imp-online"

# Each knob can be overridden by an env var for cheap demo runs.
TOTAL_REQUESTS = int(os.environ.get("HELIX_ONLINE_TOTAL_REQUESTS", "12"))
SAMPLE_RATE = float(os.environ.get("HELIX_ONLINE_SAMPLE_RATE", "1.0"))
ROLLING_WINDOW = int(os.environ.get("HELIX_ONLINE_ROLLING_WINDOW", "4"))
ROLLING_THRESHOLD = float(os.environ.get("HELIX_ONLINE_ROLLING_THRESHOLD", "0.70"))
SHADOW_SAMPLE = int(os.environ.get("HELIX_ONLINE_SHADOW_SAMPLE", "3"))

EVAL_QUESTIONS_PATH = Path(__file__).parent / "eval_questions.json"


async def main_async() -> None:
    attach_console_renderer(verbose=True)

    archive = open_archive()
    seed = await get_or_create_seed(archive)
    eval_set = load_eval_set(EVAL_QUESTIONS_PATH)
    questions = list(eval_set.questions)
    if len(questions) < TOTAL_REQUESTS:
        questions = (questions * ((TOTAL_REQUESTS // len(questions)) + 1))[:TOTAL_REQUESTS]
    questions = questions[:TOTAL_REQUESTS]

    # Define the agent once. The OnlineImprover subscribes to its
    # SESSION_END hook and reacts to each completed trajectory.
    agent = Agent(
        system_prompt=seed,
        tools=[build_retrieve_tool()],
        model=AGENT_MODEL,
    )

    policy = ImproverPolicy(
        sample_rate=SAMPLE_RATE,
        rolling_window=ROLLING_WINDOW,
        rolling_threshold=ROLLING_THRESHOLD,
        shadow_sample=SHADOW_SAMPLE,
        auto_promote=True,
        budget_per_round=SearchBudget(max_candidates=1, max_tokens=50_000),
    )

    improver = OnlineImprover(
        agent=agent,
        target_artifact_id=PROMPT_ARTIFACT_ID,
        signal=LiveTrajectoryJudge(model=JUDGE_MODEL),
        search=SPO(
            proposer_model=PROPOSER_MODEL,
            first_round_model=AGENT_MODEL,
            rounds=1,
        ),
        archive=archive,
        policy=policy,
        improver_id=IMPROVER_ID,
    )
    agent.attach_improver(improver)
    await improver.start()

    print()
    print("=" * 70)
    print(f"Online improvement loop  •  {TOTAL_REQUESTS} requests  "
          f"•  sample rate {SAMPLE_RATE:.0%}  •  threshold {ROLLING_THRESHOLD}")
    print("=" * 70)
    print(f"Starting prompt: {seed.id} v{seed.version}  ({seed.created_by})")
    print()

    try:
        for i, q in enumerate(questions, start=1):
            print(f"[req {i:02d}/{TOTAL_REQUESTS}] {q.question[:80]}"
                  f"{'...' if len(q.question) > 80 else ''}")
            try:
                answer, _trajectory = await agent.run(q.question)
            except Exception as exc:
                print(f"  ! agent error: {type(exc).__name__}: {exc}")
                continue

            s = improver.status
            avg = s.rolling_average
            avg_str = f"{avg:.2f}" if avg is not None else "n/a"
            print(f"           rolling avg={avg_str}  spot_checks={s.spot_checks_done}  "
                  f"proposed={s.candidates_proposed}  promoted={s.candidates_promoted}")
    finally:
        await improver.stop()

    print()
    print("=" * 70)
    print("Online loop complete.")
    final_live = await archive.live_champion(PROMPT_ARTIFACT_ID)
    if final_live is not None:
        print(f"  final live champion: v{final_live.version} ({final_live.created_by})")
    else:
        print(f"  no live champion recorded (agent still serves genesis)")
    s = improver.status
    print(f"  improver status: trajectories={s.trajectories_seen} "
          f"spot_checks={s.spot_checks_done} proposed={s.candidates_proposed} "
          f"promoted={s.candidates_promoted}")
    print(f"  archive: {ARCHIVE_PATH}")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
