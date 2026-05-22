"""Chapter 2 demo: improving HelixAgent's system prompt online, with auto-promote.

The ONLINE counterpart to `spo_offline_loop.py`. Where offline runs SPO
against a labeled eval set and waits for a human to promote the winner,
this script simulates the online pattern:

  - Live "traffic" is replayed from the chapter's eval set, one request
    at a time, with no reference answer available to the runtime.
  - A reference-free Signal (`LiveTrajectoryJudge`) spot-checks a sample
    of responses, scoring each trajectory 0-1 against a rubric.
  - A rolling average of recent spot-check scores becomes the gap signal.
  - When the rolling average drops below a threshold, the script triggers
    SPO to propose a candidate prompt and shadow-evaluates it on the next
    few requests.
  - If the candidate's mean spot-check score beats the reference's, the
    script publishes `CandidateWins(mode="online", auto_promote=True)`.
    The default promotion hook (registered globally by the Improver
    package) reacts by calling `archive.promote()`, and from the next
    request onward the agent serves the new version.

The key contrast with offline:

  +----------------+----------------------+------------------------+
  |                | offline              | online                 |
  +----------------+----------------------+------------------------+
  | trigger        | manual / scheduled   | rolling score drop     |
  | signal         | pairwise judge       | absolute rubric judge  |
  | promotion      | human via dashboard  | auto via default hook  |
  | safety         | any artifact layer   | L1/L2 only (enforced)  |
  +----------------+----------------------+------------------------+

The Improver constructor refuses ImproverMode.ONLINE for L3/L4 artifacts.
The system prompt is L1 (ArtifactKind.PROMPT), so this script is safe to
run.

This script intentionally drives orchestration in-line rather than using
the Improver class's round loop. The Improver's `run_improvement_round`
is pairwise by construction (reference pass + candidate pass + pairwise
judge); online improvement uses different primitives in a different
shape. The script is a faithful composition of Signal + Search + Archive
+ Hook from the spec, just wired differently. See DESIGN_NOTES.md
section 10.

Cost note: every spot-checked request invokes the agent plus the judge
LLM. Adjust SAMPLE_RATE and TOTAL_REQUESTS to keep the example cheap
when iterating.
"""

from __future__ import annotations

import asyncio
import os
import sys
from collections import deque
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from helix.agent import Agent
from helix.archive import SQLiteArchive
from helix.artifact import Artifact
from helix.env import load_env
from helix.eval import load_eval_set
from helix.improvement import ImproverMode, ImproverPolicy, Schedule
from helix.improvement.promotion import (
    ensure_default_handler_registered,
    register_improver_archive,
)
from helix.observability import attach_console_renderer
from helix.observability.bus import get_bus
from helix.observability.events import CandidateWins
from helix.search.base import SearchBudget, Variant
from helix.search.spo import SPO
from helix.signal import GapMeasurement
from helix.signals.live_judge import LiveTrajectoryJudge
from helix.trajectory import Trajectory

from chapters.ch02.helixagent_v1 import (
    ARCHIVE_PATH,
    PROMPT_ARTIFACT_ID,
    get_or_create_seed,
    open_archive,
)
from chapters.ch02.helixagent_v0 import build_retrieve_tool

load_env()

# Configuration -----------------------------------------------------------
AGENT_MODEL = "claude-haiku-4-5"
PROPOSER_MODEL = "claude-sonnet-4-6"
JUDGE_MODEL = "claude-sonnet-4-6"
IMPROVER_ID = "imp-online"

# Each knob can be overridden by an env var for cheap demo runs. The
# defaults are tuned for a representative ~$1-2 demo on Haiku + Sonnet.

# How many simulated live requests to replay this run.
TOTAL_REQUESTS = int(os.environ.get("HELIX_ONLINE_TOTAL_REQUESTS", "12"))

# Fraction of requests that get a spot-check by the rubric judge. 1.0 means
# score every response; 0.33 means roughly every third response. Real
# deployments would set this much lower (1-5%) to keep judge cost bounded.
SAMPLE_RATE = float(os.environ.get("HELIX_ONLINE_SAMPLE_RATE", "1.0"))

# Rolling window for the gap signal. Drop below ROLLING_THRESHOLD over the
# last ROLLING_WINDOW spot-checked responses triggers a candidate proposal.
ROLLING_WINDOW = int(os.environ.get("HELIX_ONLINE_ROLLING_WINDOW", "4"))
ROLLING_THRESHOLD = float(os.environ.get("HELIX_ONLINE_ROLLING_THRESHOLD", "0.70"))

# When a candidate is proposed, how many subsequent requests it sees as a
# shadow evaluation before the script compares it to the reference.
SHADOW_SAMPLE = int(os.environ.get("HELIX_ONLINE_SHADOW_SAMPLE", "3"))

EVAL_QUESTIONS_PATH = Path(__file__).parent / "eval_questions.json"


def build_agent(prompt: Artifact, model: str = AGENT_MODEL) -> Agent:
    """Build an Agent that serves a specific prompt artifact."""
    return Agent(
        system_prompt=prompt,
        tools=[build_retrieve_tool()],
        model=model,
        max_iterations=10,
        max_tool_calls=20,
    )


async def serve_one_request(prompt: Artifact, question: str) -> tuple[str | None, Trajectory]:
    """Pretend this is a live request: build an agent on the spot, answer once."""
    agent = build_agent(prompt)
    try:
        answer, trajectory = await agent.run(question)
        return answer, trajectory
    except Exception as exc:
        # In a real deployment the request handler decides what to do here;
        # for the demo we just record the failure as a None answer and let
        # the judge see the (empty) trajectory.
        print(f"  ! agent error: {type(exc).__name__}: {exc}")
        return None, Trajectory(task=question)


async def spot_check(
    prompt: Artifact,
    trajectory: Trajectory,
    judge: LiveTrajectoryJudge,
) -> GapMeasurement:
    """Score one trajectory in isolation with the rubric judge."""
    return await judge.measure(candidate=prompt, trajectory=trajectory)


async def propose_candidate(
    seed: Artifact,
    archive: SQLiteArchive,
    judge: LiveTrajectoryJudge,
) -> Variant | None:
    """Ask SPO for one candidate mutation of the current live prompt.

    SPO normally consumes a pairwise Signal so it can read judge feedback
    from prior rounds. With an absolute rubric judge there's no left/right
    feedback to act on, but SPO's first-round path falls back to mutating
    blind from the seed alone. That's the only path used here.
    """
    search = SPO(
        proposer_model=PROPOSER_MODEL,
        first_round_model=AGENT_MODEL,
        rounds=1,
    )
    budget = SearchBudget(max_candidates=1, max_tokens=50_000)
    async for variant in search.propose(
        seed=seed, signal=judge, archive=archive, budget=budget
    ):
        return variant
    return None


async def main_async() -> None:
    attach_console_renderer(verbose=True)

    archive = open_archive()
    eval_set = load_eval_set(EVAL_QUESTIONS_PATH)
    questions = list(eval_set.questions)
    if len(questions) < TOTAL_REQUESTS:
        # Recycle questions if the eval set is small.
        questions = (questions * ((TOTAL_REQUESTS // len(questions)) + 1))[:TOTAL_REQUESTS]
    questions = questions[:TOTAL_REQUESTS]

    # The genesis (or current live champion if one is already promoted) is
    # what the agent serves at the start. Online improvement reads this
    # and writes back to the same id via archive.promote() if a candidate
    # wins. Layer check happens implicitly: PROMPT is L1, so ONLINE is
    # allowed.
    initial = await get_or_create_seed(archive)

    # ImproverPolicy is used here only to advertise the mode + auto_promote
    # contract that the default handler reads off the CandidateWins event.
    # This script does its own orchestration; the Improver class is not
    # instantiated. The policy still documents the *intent* of the loop.
    policy = ImproverPolicy(
        schedule=Schedule.CONTINUOUS,
        mode=ImproverMode.ONLINE,
        auto_promote=True,
    )
    print(f"policy: mode={policy.mode.value}  auto_promote={policy.auto_promote}")

    # Wire the auto-promote hook. The Improver class normally does this
    # when constructed; since we're not using one, we wire it by hand.
    register_improver_archive(IMPROVER_ID, archive)
    bus = get_bus()
    ensure_default_handler_registered(bus)

    judge = LiveTrajectoryJudge(model=JUDGE_MODEL)

    live_prompt: Artifact = initial
    rolling: deque[float] = deque(maxlen=ROLLING_WINDOW)

    print()
    print("=" * 70)
    print(f"Online improvement loop  •  {TOTAL_REQUESTS} requests  "
          f"•  sample rate {SAMPLE_RATE:.0%}  •  threshold {ROLLING_THRESHOLD}")
    print("=" * 70)
    print(f"Starting live prompt: {live_prompt.id} v{live_prompt.version}  ({live_prompt.created_by})")
    print()

    for i, q in enumerate(questions, start=1):
        print(f"[req {i:02d}/{TOTAL_REQUESTS}] {q.question[:80]}{'...' if len(q.question) > 80 else ''}")

        # Step 1: serve the request with the current live prompt.
        answer, trajectory = await serve_one_request(live_prompt, q.question)

        # Step 2: probabilistic spot-check. Real deployments would gate
        # this by a sample-rate flag and a per-tenant rate budget.
        should_score = (i - 1) % max(1, int(round(1 / max(SAMPLE_RATE, 0.01)))) == 0
        if not should_score or answer is None:
            print(f"           (no spot check)")
            continue

        m = await spot_check(live_prompt, trajectory, judge)
        rolling.append(m.score or 0.0)
        avg = sum(rolling) / len(rolling)
        flags = m.metadata.get("flags") or []
        flags_str = f"  flags={flags}" if flags else ""
        print(f"           judge score={m.score:.2f}  rolling avg={avg:.2f}  "
              f"window={len(rolling)}/{ROLLING_WINDOW}{flags_str}")

        # Step 3: gap test. Only trigger a candidate when the rolling
        # window is full *and* it has dropped below threshold. Triggering
        # off a single bad response would propose a new prompt every
        # request and burn budget.
        if len(rolling) < ROLLING_WINDOW:
            continue
        if avg >= ROLLING_THRESHOLD:
            continue

        print(f"           [!] rolling avg {avg:.2f} < {ROLLING_THRESHOLD}, proposing candidate...")
        candidate_variant = await propose_candidate(live_prompt, archive, judge)
        if candidate_variant is None:
            print(f"           (SPO did not propose; continuing on current prompt)")
            continue

        # Step 4: shadow-evaluate the candidate on the next few questions
        # without serving it to live traffic yet. This is the cheap online
        # equivalent of offline's full reference-vs-candidate pass.
        cand_prompt = candidate_variant.artifact
        ref_scores: list[float] = []
        cand_scores: list[float] = []
        shadow_questions = []
        for j in range(SHADOW_SAMPLE):
            if i + j >= len(questions):
                break
            shadow_questions.append(questions[i + j])

        if not shadow_questions:
            print(f"           (no remaining traffic to shadow-test candidate)")
            continue

        print(f"           shadow eval on {len(shadow_questions)} request(s)...")
        for sq in shadow_questions:
            ref_ans, ref_traj = await serve_one_request(live_prompt, sq.question)
            cand_ans, cand_traj = await serve_one_request(cand_prompt, sq.question)
            if ref_ans is not None:
                ref_m = await spot_check(live_prompt, ref_traj, judge)
                ref_scores.append(ref_m.score or 0.0)
            if cand_ans is not None:
                cand_m = await spot_check(cand_prompt, cand_traj, judge)
                cand_scores.append(cand_m.score or 0.0)

        if not ref_scores or not cand_scores:
            print(f"           (shadow eval inconclusive; both sides had errors)")
            continue

        ref_avg = sum(ref_scores) / len(ref_scores)
        cand_avg = sum(cand_scores) / len(cand_scores)
        print(f"           shadow: ref={ref_avg:.2f}  cand={cand_avg:.2f}")

        # Step 5: persist BOTH variants with their shadow-eval measurements
        # so the dashboard / archive panel can show the loop's reasoning.
        # The Variant for the reference is constructed inline.
        ref_measurement = GapMeasurement(
            score=ref_avg,
            preference=m.preference,  # NONE — absolute scoring
            feedback=f"shadow-eval reference avg over {len(ref_scores)} requests",
            confidence=min(1.0, len(ref_scores) / max(1, SHADOW_SAMPLE)),
            metadata={"role": "reference_shadow", "n_questions": len(ref_scores)},
        )
        cand_measurement = GapMeasurement(
            score=cand_avg,
            preference=m.preference,
            feedback=f"shadow-eval candidate avg over {len(cand_scores)} requests",
            confidence=min(1.0, len(cand_scores) / max(1, SHADOW_SAMPLE)),
            metadata={"role": "candidate_shadow", "n_questions": len(cand_scores)},
        )
        ref_variant = Variant(
            artifact=live_prompt,
            parent=live_prompt,
            search_method=live_prompt.created_by,
            metadata={"role": "reference_in_online_round"},
        )
        await archive.record(candidate_variant, cand_measurement)
        await archive.record(ref_variant, ref_measurement)

        # Step 6: if the candidate won, fire CandidateWins. The default
        # hook handler picks this up because mode == "online" and
        # auto_promote is True, and calls archive.promote() for us. That
        # publishes a Promoted event in turn, which the chat UI / chapter
        # scripts read from `archive.live_champion()` on the next request.
        if cand_avg > ref_avg:
            print(f"           [WIN] candidate wins -- emitting CandidateWins (auto-promote)")
            await bus.publish(CandidateWins(
                improver_id=IMPROVER_ID,
                target_artifact_id=PROMPT_ARTIFACT_ID,
                candidate_version=cand_prompt.version,
                reference_version=live_prompt.version,
                candidate_score=cand_avg,
                reference_score=ref_avg,
                mode="online",
                auto_promote=True,
            ))
            # Confirm the live champion swap happened. live_champion() is
            # the same call helixagent_v1.py uses, so this is exactly what
            # the agent will see on its next request.
            new_live = await archive.live_champion(PROMPT_ARTIFACT_ID)
            if new_live is not None and new_live.version == cand_prompt.version:
                live_prompt = new_live
                rolling.clear()
                print(f"           live prompt now v{live_prompt.version}; rolling window reset")
        else:
            print(f"           candidate did not beat reference; live prompt unchanged")

    print()
    print("=" * 70)
    print(f"Online loop complete.")
    final_live = await archive.live_champion(PROMPT_ARTIFACT_ID)
    if final_live is not None:
        print(f"  final live champion: v{final_live.version} ({final_live.created_by})")
    else:
        print(f"  no live champion recorded (agent still serves genesis)")
    print(f"  archive: {ARCHIVE_PATH}")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
