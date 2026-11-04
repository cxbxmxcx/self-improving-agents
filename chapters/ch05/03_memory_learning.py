"""Listing 5.3: learn which scheduling experiences help a personal calendar agent.

The calendar tools create real events in a local JSON file. Scripted user
reviews accept or reject the first booking; recalled episodes receive value
updates while the model, memory application, and learning rule stay fixed.
Default responses are scripted from visible context, so no API is required.
--live --model MODEL runs a real LLM against the same local calendar tools.
Each invocation requires a new or empty output directory to preserve evidence.
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
from importlib import import_module
import json
from pathlib import Path
import sys
import uuid

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from helix.archive import SQLiteArchive
from helix.improvement.online_memory import OnlineMemoryImprover
from helix.memory import EpisodicMemory, MemoryLearner
from helix.memory.base import MemoryContext
from helix.memory.hooks import RECALLED_KEY, WRITTEN_KEY
from helix.signals.utility import UtilitySignal
from helix.trajectory import StepKind

from chapters.ch05.personal.encoder import ENCODER_ID, embed
# Numbered chapter modules use import_module because import statements cannot
# contain identifiers beginning with a digit.
calendar_agent = import_module("chapters.ch05.02_calendar_agent")
CalendarEnvironment, build_agent, load_requests = (
    calendar_agent.CalendarEnvironment, calendar_agent.build_agent, calendar_agent.load_requests,
)
from chapters.ch05.personal.scripted import model_mode

ALPHA = 0.5
UTILITY_WEIGHT = 0.35
RECENCY_WEIGHT = 0.0


def measure_outcomes(rows: list[dict]) -> dict:
    """Report first-booking acceptance, validity, and missing feedback separately.

    All requests stay in the denominator. This describes the walkthrough;
    improvement requires comparison against a fixed control on held-out tasks.
    """
    outcomes = [row["outcome"] for row in rows if row["outcome"] is not None]
    accepted = sum(item["valid"] and item["accepted"] is True for item in outcomes)
    rejected = sum(item["valid"] and item["accepted"] is False for item in outcomes)
    invalid = sum(not item["valid"] for item in outcomes)
    return {
        "requests": len(rows),
        "valid_bookings": sum(item["valid"] for item in outcomes),
        "invalid_bookings": invalid,
        "accepted": accepted,
        "rejected": rejected,
        "unobserved": len(rows) - accepted - rejected - invalid,
        "acceptance_rate": accepted / len(rows) if rows else None,
    }


def open_memory(output: Path, archive: SQLiteArchive) -> EpisodicMemory:
    return EpisodicMemory(
        output / "episodic.sqlite", archive=archive,
        utility_weight=UTILITY_WEIGHT, recency_weight=RECENCY_WEIGHT,
        embedding_fn=embed,
    )


async def run_walkthrough(
    output: Path, *, live: bool = False, model: str | None = None
) -> dict:
    if live and not model:
        raise ValueError("A live run requires an explicit model.")
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(
            "Choose a new or empty output directory; "
            "existing evidence is preserved."
        )
    output.mkdir(parents=True, exist_ok=True)
    archive = SQLiteArchive(output / "archive.sqlite")
    tier = open_memory(output, archive)
    calendar = CalendarEnvironment(output / "calendar.json")
    agent = build_agent(
        tier, calendar, model=model if live else "calendar-scripted"
    )
    learner = MemoryLearner(
        agent.memory["episodic"], archive, UtilitySignal(attribution="equal"),
        alpha=ALPHA, upvote_threshold=0.5, downvote_threshold=0.5,
    )
    improver = OnlineMemoryImprover(
        agent, learner=learner, reward_fn=calendar.reward
    )
    agent.attach_improver(improver)
    await improver.start()
    configuration = {"algorithm": learner.algorithm, "alpha": ALPHA, "credit": "equal",
                     "utility_weight": UTILITY_WEIGHT, "recency_weight": RECENCY_WEIGHT,
                     "episodic_k": 1, "encoder": ENCODER_ID, "model": agent.model,
                     "prompt_sha256": hashlib.sha256(agent.system_prompt.content.encode()).hexdigest()}
    aliases, rows = {}, []
    mode = "live model with local calendar and scripted user reviews" if live else "scripted walkthrough; not measured model performance"
    print(mode)

    with model_mode(live):
        for number, request in enumerate(load_requests(), 1):
            calendar.begin(request)
            available_starts = calendar.read()["available_starts"]
            before = {entry.id: entry for entry in await tier.list_all()}
            answer, trajectory = await agent.run(
                request["question"],
                context=MemoryContext(user_id=request["user"], session_id=request["session"]),
            )
            if improver.status().last_error:
                raise RuntimeError(improver.status().last_error)
            written = trajectory.metadata.get(WRITTEN_KEY)
            if not written:
                raise RuntimeError("The walkthrough did not retain its episode.")
            aliases[written["entry_id"]] = f"E{number}"
            recalls = []
            for recalled in trajectory.metadata.get(RECALLED_KEY, []):
                entry_id = recalled["entry_id"]
                current = await tier.get(entry_id)
                recalls.append({"episode": aliases[entry_id], "recalled_ref": recalled["artifact_ref"],
                                "current_ref": list(current.artifact_ref),
                                "q_before": before[entry_id].score, "q_after": current.score})
            row = {"request": request["id"], "user": request["user"], "session": request["session"],
                   "question": request["question"], "answer": answer,
                   "outcome": calendar.result, "reward": calendar.reward(trajectory),
                   "available_starts": available_starts,
                   "recalled": recalls, "written_episode": f"E{number}", "written": written,
                   "tool_calls": [s.payload for s in trajectory.steps if s.kind == StepKind.TOOL_CALL]}
            rows.append(row)
            await archive.store_trajectory(agent.system_prompt, request["id"], trajectory, answer=str(answer))
            changes = ", ".join(f"{r['episode']} {r['q_before']:.3f}->{r['q_after']:.3f}" for r in recalls) or "none"
            print(f"{request['id']}: reward={row['reward']}; recalled {changes}; wrote E{number} at 0.500")
        await improver.stop()

        # These counts describe the walkthrough, not improvement over a control.
        measurement = measure_outcomes(rows)
        print(f"Accepted first bookings: {measurement['accepted']}/{measurement['requests']} "
              f"({measurement['acceptance_rate']:.0%}); valid: {measurement['valid_bookings']}; "
              f"invalid: {measurement['invalid_bookings']}; "
              f"no feedback: {measurement['unobserved']}")

        # Handoff uses a new Agent and a reopened tier. It reads the learned
        # store but has no learner and performs no write-back or calendar action.
        reopened = open_memory(output, archive)
        handoff = build_agent(reopened, CalendarEnvironment(output / "calendar.json"), model=agent.model)
        handoff.memory_hooks.read_only = True
        probes = {}
        for user in ("maya", "leo"):
            answer, trajectory = await handoff.run(
                "History only: What feedback did I give about my writing booking?",
                context=MemoryContext(user_id=user, session_id=user + "-handoff"),
            )
            probes[user] = {"answer": answer, "recalled": trajectory.metadata.get(RECALLED_KEY, [])}
            print(f"New agent, {user}: recalled {len(probes[user]['recalled'])} episode(s)")

    assert (learner.alpha, tier.utility_weight, tier.recency_weight, agent.memory_hooks.episodic_k) == (ALPHA, UTILITY_WEIGHT, RECENCY_WEIGHT, 1)
    assert agent.model == configuration["model"] and tier.embedding_fn is embed
    assert learner.algorithm == configuration["algorithm"] and learner.utility.attribution == configuration["credit"]
    assert hashlib.sha256(agent.system_prompt.content.encode()).hexdigest() == configuration["prompt_sha256"]
    report = {"mode": mode, "configuration": configuration,
              "requests_sha256": hashlib.sha256((Path(__file__).parent / "personal/requests.json").read_bytes()).hexdigest(),
              "calendar_sha256": hashlib.sha256(calendar.path.read_bytes()).hexdigest() if calendar.path.exists() else None,
              "episodes": aliases, "requests": rows, "measurement": measurement, "handoff": probes,
              "limits": "Scripted responses and scripted user reviews illustrate mechanics; this is not a performance improvement benchmark."}
    (output / "trace.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    transcript = ["# Personal calendar memory walkthrough", "", mode, ""]
    for row in rows:
        transcript += ["## " + row["request"], "", row["question"], "", str(row["answer"]), "",
                       "Reward: " + str(row["reward"]), "", "Recalled: `" + json.dumps(row["recalled"]) + "`", ""]
    (output / "transcript.md").write_text("\n".join(transcript), encoding="utf-8")
    print("Trace and transcript: " + str(output))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Use a real model with local calendar tools and scripted user reviews.")
    parser.add_argument("--model", help="Explicit LiteLLM model identifier for --live.")
    parser.add_argument("--output", type=Path, help="New or empty directory for this run's evidence.")
    args = parser.parse_args()
    if args.live and not args.model:
        parser.error("--live requires --model")
    if args.live:
        from helix.env import load_env
        load_env()
    tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6]
    output = args.output or Path(__file__).parent / "runs/personal" / tag
    asyncio.run(run_walkthrough(output, live=args.live, model=args.model))


if __name__ == "__main__":
    main()
