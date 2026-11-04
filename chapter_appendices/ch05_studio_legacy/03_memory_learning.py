"""Listing 5.3: follow a studio support case through the real memory harness.

Recall selects the case supplied as context; the agent chooses the support
action. Completed outcomes update the selected case's expected usefulness,
while the model, memory application, and learning rule remain fixed.

Default mode uses a scripted responder, local runbooks, and a fixed lexical
encoder; no API key or model download is needed. --live --model MODEL uses
an LLM with the same fictional support tools, so its actions may differ.

Every invocation owns a new directory. An existing nonempty output
directory is rejected to preserve the evidence from earlier runs.
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

from chapter_appendices.ch05_studio_legacy.studio.corpus import CORPUS_DIR, ENCODER_ID, StudioCorpus, embed
# Numbered chapter modules use import_module because import statements cannot
# contain identifiers beginning with a digit.
support = import_module("chapter_appendices.ch05_studio_legacy.02_studio_agent")
SupportSimulator, build_agent, load_tickets = (
    support.SupportSimulator, support.build_agent, support.load_tickets,
)
from chapter_appendices.ch05_studio_legacy.studio.scripted import model_mode

ALPHA = 0.5
UTILITY_WEIGHT = 0.35
RECENCY_WEIGHT = 0.0


def measure_resolution(rows: list[dict]) -> dict:
    """Count walkthrough outcomes; this is not an improvement benchmark.

    Keep every request in the denominator and report missing outcomes.
    """
    requests = len(rows)
    observed = [row["outcome"] for row in rows
                if row["outcome"] is not None and row["reward"] is not None]
    resolved = sum(outcome["resolved"] for outcome in observed)
    return {
        "requests": requests,
        "resolved": resolved,
        "unresolved": len(observed) - resolved,
        "unobserved": requests - len(observed),
        "resolution_rate": resolved / requests if requests else None,
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
    corpus, simulator = StudioCorpus(), SupportSimulator()
    agent = build_agent(
        tier, corpus, simulator, model=model if live else "studio-scripted"
    )
    learner = MemoryLearner(
        agent.memory["episodic"], archive, UtilitySignal(attribution="equal"),
        alpha=ALPHA, upvote_threshold=0.5, downvote_threshold=0.5,
    )
    improver = OnlineMemoryImprover(
        agent, learner=learner, reward_fn=simulator.reward
    )
    agent.attach_improver(improver)
    await improver.start()
    configuration = {"algorithm": learner.algorithm, "alpha": ALPHA, "credit": "equal",
                     "utility_weight": UTILITY_WEIGHT, "recency_weight": RECENCY_WEIGHT,
                     "episodic_k": 1, "encoder": ENCODER_ID, "model": agent.model,
                     "prompt_sha256": hashlib.sha256(agent.system_prompt.content.encode()).hexdigest()}
    aliases, rows = {}, []
    mode = "live model with simulated tickets" if live else "scripted walkthrough; not measured model performance"
    print(mode)

    with model_mode(live):
        for number, ticket in enumerate(load_tickets()[:4], 1):
            simulator.begin(ticket)
            before = {entry.id: entry for entry in await tier.list_all()}
            answer, trajectory = await agent.run(
                ticket["question"],
                context=MemoryContext(user_id=ticket["user"], session_id=ticket["session"]),
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
            row = {"ticket": ticket["id"], "user": ticket["user"], "session": ticket["session"],
                   "question": ticket["question"], "answer": answer,
                   "outcome": simulator.result, "reward": simulator.reward(trajectory),
                   "recalled": recalls, "written_episode": f"E{number}", "written": written,
                   "tool_calls": [s.payload for s in trajectory.steps if s.kind == StepKind.TOOL_CALL]}
            rows.append(row)
            await archive.store_trajectory(agent.system_prompt, ticket["id"], trajectory, answer=str(answer))
            changes = ", ".join(f"{r['episode']} {r['q_before']:.3f}->{r['q_after']:.3f}" for r in recalls) or "none"
            print(f"{ticket['id']}: reward={row['reward']}; recalled {changes}; wrote E{number} at 0.500")
        await improver.stop()

        # These counts describe the walkthrough, not improvement over a control.
        measurement = measure_resolution(rows)
        print(f"Observed resolutions: {measurement['resolved']}/{measurement['requests']} "
              f"({measurement['resolution_rate']:.0%}); no outcome: {measurement['unobserved']}")

        # Handoff uses a new Agent and a reopened tier. It reads the learned
        # store but has no learner and performs no write-back or support step.
        reopened = open_memory(output, archive)
        handoff = build_agent(reopened, corpus, SupportSimulator(), model=agent.model)
        handoff.memory_hooks.read_only = True
        probes = {}
        for user in ("maya", "leo"):
            answer, trajectory = await handoff.run(
                "History only: CrateSync AUTH17, VPN connected. What happened on my earlier case?",
                context=MemoryContext(user_id=user, session_id=user + "-handoff"),
            )
            probes[user] = {"answer": answer, "recalled": trajectory.metadata.get(RECALLED_KEY, [])}
            print(f"New agent, {user}: recalled {len(probes[user]['recalled'])} episode(s)")

    assert (learner.alpha, tier.utility_weight, tier.recency_weight, agent.memory_hooks.episodic_k) == (ALPHA, UTILITY_WEIGHT, RECENCY_WEIGHT, 1)
    assert agent.model == configuration["model"] and tier.embedding_fn is embed
    assert learner.algorithm == configuration["algorithm"] and learner.utility.attribution == configuration["credit"]
    assert hashlib.sha256(agent.system_prompt.content.encode()).hexdigest() == configuration["prompt_sha256"]
    report = {"mode": mode, "configuration": configuration,
              "corpus_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(CORPUS_DIR.glob("*.md"))},
              "tickets_sha256": hashlib.sha256((CORPUS_DIR.parent / "tickets.json").read_bytes()).hexdigest(),
              "episodes": aliases, "requests": rows, "measurement": measurement, "handoff": probes,
              "limits": "Scripted responses and simulated outcomes illustrate mechanics; this is not an answer-quality benchmark."}
    (output / "trace.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    transcript = ["# Studio memory walkthrough", "", mode, ""]
    for row in rows:
        transcript += ["## " + row["ticket"], "", row["question"], "", str(row["answer"]), "",
                       "Reward: " + str(row["reward"]), "", "Recalled: `" + json.dumps(row["recalled"]) + "`", ""]
    (output / "transcript.md").write_text("\n".join(transcript), encoding="utf-8")
    print("Trace and transcript: " + str(output))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Use a real model with the fictional simulator.")
    parser.add_argument("--model", help="Explicit LiteLLM model identifier for --live.")
    parser.add_argument("--output", type=Path, help="New or empty directory for this run's evidence.")
    args = parser.parse_args()
    if args.live and not args.model:
        parser.error("--live requires --model")
    if args.live:
        from helix.env import load_env
        load_env()
    tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6]
    output = args.output or Path(__file__).parent / "runs/studio" / tag
    asyncio.run(run_walkthrough(output, live=args.live, model=args.model))


if __name__ == "__main__":
    main()
