"""Listing 6.2: the chapter 5 assistant with a metacognitive scaffold.

A decision model answers typed questions at the hooks, and the scaffold's
configuration turns the answers into control: the fast path, the
deliberate path, or a question back; a confirmation before a consequential
call; a revision of an incomplete answer; and a lesson after a rejection.
Chapter 5's memory learning runs underneath unchanged.

By default the model calls come from a scripted stand-in and the decision
model's answers are replayed from a recording, so no key is needed.
--live uses gpt-5.4-mini; --decider jev-latest asks Jev directly.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from helix.agent import Agent
from helix.archive import SQLiteArchive
from helix.artifact import Subtype, genesis
from helix.hooks import HookPoint
from helix.improvement.online_memory import OnlineMemoryImprover
from helix.memory import EpisodicMemory, MemoryLearner
from helix.memory.base import MemoryContext
from helix.scaffold import MetacognitiveScaffold, config_artifact, decider_usage, record, thresholds_artifact
from helix.signals.utility import UtilitySignal

from chapters.ch05.personal.encoder import embed
from chapter_appendices.ch06_decision_scaffold_legacy.assistant.environment import WATCHED_TOOLS, AssistantEnvironment, load_requests
from chapter_appendices.ch06_decision_scaffold_legacy.assistant.recording import decision_model
from chapter_appendices.ch06_decision_scaffold_legacy.assistant.scripted import DELIBERATE_MODEL, FAST_MODEL, model_mode
from chapter_appendices.ch06_decision_scaffold_legacy.assistant.tools import make_tools

LIVE_MODEL = "gpt-5.4-mini"
# One model on both paths: the fast path does not reason, and the deliberate
# path reasons at length after the planner's checklist.
PATHS = {"fast": {"reasoning_effort": "none"}, "deliberate": {"reasoning_effort": "high"}}
# Dollars per million input and output tokens; the scripted names reuse them
# so the walkthrough's costs show the mechanics at a real model's prices.
PRICES = {LIVE_MODEL: (0.75, 4.50), FAST_MODEL: (0.75, 4.50), DELIBERATE_MODEL: (0.75, 4.50)}

PROMPT = """You are a personal assistant for one user's local demonstration
calendar and inbox. The week runs Monday 2026-10-19 to Friday 2026-10-23;
days are mon to fri, working hours are 09:00 to 17:00, and times are HH:MM.
Read before you act, never double-book the user, and do only what the
request asks. After you book or move an event, call review and report the
user's feedback exactly; a rejection does not undo the change. Earlier
interactions are dated experiences, and the current request takes priority.
When the request leaves a writing time open, use relevant prior outcomes;
if none help, use 14:00 when it is free. Finish with a short summary.
"""

PLANNER = """1. List every constraint in the request: people, days, durations,
counts, and limits that span several events.
2. Read the calendar of every person involved, not only the user's, and
check each candidate slot against every constraint before booking.
3. For a message, list every question it asks and answer each one; for a
request with several questions, answer all of them.
4. Before sending a message or moving an event someone else organized,
stop and ask the user to confirm.
5. If a name or an event could mean more than one thing, ask which one."""

REFLECTOR = """You review a personal assistant's run that the user rejected or
corrected. Write one lesson of one or two sentences that would have
produced an accepted result for requests like this one. Name the person,
time, or preference involved and what to do differently. Return only the
lesson."""

SEED_CONFIG = {
    "questions": {
        "multi_event_limit": {
            "hook": "route", "type": "probability",
            "instructions": "The request sets a limit that spans several events or people."},
        "ambiguous_reference": {
            "hook": "route", "type": "probability",
            "instructions": "The request names a person, event, or message that matches more than one item "
                            "in the state, and nothing in the request or the memory says which one."},
        "consequence": {
            "hook": "tool", "type": "choice",
            "instructions": "What does this tool call change for people other than the user?",
            "criteria": {"none": "Only the user's own calendar or drafts",
                         "other_event": "An event another person organized",
                         "sends": "A message sent in the user's name"}},
        "reply_complete": {
            "hook": "output", "type": "probability",
            "instructions": "The reply answers every question the request asks."},
    },
    "rules": [
        {"question": "ambiguous_reference", "op": ">=", "threshold": "ask", "action": "ask"},
        {"question": "multi_event_limit", "op": ">=", "threshold": "deliberate", "action": "deliberate"},
        {"question": "consequence", "op": "not_in", "value": ["none"], "action": "confirm"},
        {"question": "reply_complete", "op": "<", "threshold": "revise", "action": "revise"},
    ],
}
SEED_THRESHOLDS = {"ask": 0.5, "deliberate": 0.5, "revise": 0.5}


def seed_artifacts() -> dict:
    """The genesis versions of the four artifacts the scaffold reads."""
    return {
        "config": config_artifact("assistant.metacognition.config", SEED_CONFIG),
        "thresholds": thresholds_artifact("assistant.routing.thresholds", SEED_THRESHOLDS),
        "planner": genesis("assistant.planner", Subtype.PLANNER, PLANNER),
        "reflector": genesis("assistant.reflector", Subtype.REFLECTOR, REFLECTOR),
    }


def build_assistant(env: AssistantEnvironment, tier, decider, *, live: bool = False,
                    artifacts: dict | None = None) -> Agent:
    """The assistant with memory and a metacognitive scaffold.

    The environment supplies the scaffold's view of the world: which calls
    the user already confirmed, the run's reward, and what the decision
    model should see about the week.
    """
    arts = {**seed_artifacts(), **(artifacts or {})}
    fast, deliberate = (LIVE_MODEL, LIVE_MODEL) if live else (FAST_MODEL, DELIBERATE_MODEL)
    scaffold = MetacognitiveScaffold(
        decider, arts["config"], arts["thresholds"], planner=arts["planner"], reflector=arts["reflector"],
        fast_model=fast, fast_params=PATHS["fast"] if live else {},
        deliberate_model=deliberate, deliberate_params=PATHS["deliberate"] if live else {},
        watch_tools=WATCHED_TOOLS, confirmed=env.confirmed, reward_fn=env.reward,
        state_fn=env.decision_state,
    )
    agent = Agent(
        system_prompt=genesis("assistant.prompt", Subtype.PROMPT, PROMPT), tools=make_tools(env),
        model=fast, max_iterations=8, max_tool_calls=8, memory_tiers={"episodic": tier}, scaffold=scaffold,
    )
    # An episode and a lesson can both be relevant, so recall two.
    agent.memory_hooks.episodic_k = 2

    # Each run carries its own reward, so a signal scores a shadow run against
    # the forked week it ran in, not the live one.
    async def stamp_reward(trajectory, **_):
        trajectory.metadata["reward"] = env.reward(trajectory)

    agent.hooks.register(HookPoint.SESSION_END, stamp_reward)
    return agent


def describe(trajectory) -> str:
    """One line per run: the route and why, the monitor's verdicts, and any lesson."""
    meta = record(trajectory)
    fired = [d for d in meta["decisions"] if d["hook"] == "route" and d["actions"] and d["value"] is not None]
    why = ", ".join(f"{d['question']} {d['value']:.2f}" for d in fired)
    parts = [f"route {meta.get('route')}" + (f" ({why})" if why else "")]
    parts += [f"{t['tool']} {t['verdict']}" for t in meta["tools"]]
    if meta.get("revised"):
        parts.append("revised the reply")
    if meta.get("lesson"):
        parts.append(f"lesson: {meta['lesson']['text']}")
    return "; ".join(parts)


WALKTHROUGH = ["F01", "D01", "A01", "C01", "D08", "W01", "M01"]


async def run_walkthrough(output: Path, *, mode: str = "replay", decider_name: str = "jev-latest") -> dict:
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Choose a new or empty output directory; existing evidence is preserved.")
    output.mkdir(parents=True, exist_ok=True)
    live = mode == "live"
    decider = decision_model(mode, folder=output, name=decider_name) if live else decision_model(mode)
    archive = SQLiteArchive(output / "archive.sqlite")
    # Chapter 5's recall settings: utility counts, recency does not.
    tier = EpisodicMemory(output / "episodic.sqlite", archive=archive, utility_weight=0.35, recency_weight=0.0,
                          embedding_fn=embed)
    env = AssistantEnvironment()
    agent = build_assistant(env, tier, decider, live=live)
    learner = MemoryLearner(tier, archive, UtilitySignal(attribution="equal"), alpha=0.5)
    improver = OnlineMemoryImprover(agent, learner=learner, reward_fn=env.reward)
    agent.attach_improver(improver)
    await improver.start()
    requests = {r["id"]: r for r in load_requests()}
    rows = []
    print("live model" if live else "scripted stand-in; not measured model performance",
          "| decision model:", decider.name)
    with model_mode(live):
        for rid in WALKTHROUGH:
            request = requests[rid]
            env.reset()  # every request starts from the same week; memory persists
            env.begin(request)
            answer, trajectory = await agent.run(
                request["text"], context=MemoryContext(user_id=request["user"], session_id=request["session"]))
            reward = env.reward(trajectory)
            rows.append({"request": rid, "text": request["text"], "answer": answer, "reward": reward,
                         "metacognition": record(trajectory)})
            await archive.store_trajectory(agent.system_prompt, rid, trajectory, answer=str(answer))
            print(f"{rid}: reward {reward:.0f}; {describe(trajectory)}")
            usage = decider_usage(trajectory)
            rows[-1]["decider"] = usage.to_dict()
    await improver.stop()
    spent = [r["decider"] for r in rows]
    print(f"Decision model: {sum(s['calls'] for s in spent)} calls, "
          f"{sum(s['input_tokens'] for s in spent):,} input tokens, ${sum(s['dollars'] for s in spent):.5f}, "
          f"{sum(s['seconds'] for s in spent):.2f} s recorded")
    report = {"mode": "live" if live else "scripted", "decider": decider.name, "requests": rows}
    (output / "trace.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print("Trace: " + str(output / "trace.json"))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--record", action="store_true", help="Add missing answers to the recording with live Jev.")
    group.add_argument("--live", action="store_true", help=f"Use {LIVE_MODEL} and a live decision model.")
    parser.add_argument("--decider", default="jev-latest", help="With --live: jev-latest, openai/<model>, or llm/<model>.")
    parser.add_argument("--output", type=Path, help="New or empty directory for this run's evidence.")
    args = parser.parse_args()
    mode = "live" if args.live else "record" if args.record else "replay"
    if mode != "replay":
        from helix.env import load_env
        load_env()
    tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6]
    output = args.output or Path(__file__).parent / "runs/walkthrough" / tag
    asyncio.run(run_walkthrough(output, mode=mode, decider_name=args.decider))


if __name__ == "__main__":
    main()
