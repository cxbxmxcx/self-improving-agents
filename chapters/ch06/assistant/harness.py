"""Build a calendar assistant with procedural memory and fixed effort.

The prompt identifies the user and working week without a scheduling method.
Tool construction and context injection retrieve skills separately through
the same helper with matching inputs against fixed memory. The effort setting
reaches every model call through a PRE_MODEL hook, leaving the loop unchanged.

The bootstrapped memory is a hand-built function that computes valid starts.
Learned functions use the same name and description for behavioral comparison.
Agent sweeps then test whether those functions support successful execution
at the selected effort setting.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from chapters.ch05.personal.encoder import embed
from chapters.ch06.assistant.environment import SchedulingEnvironment
from chapters.ch06.assistant.recording import (
    recorded_seconds,
    restore_seconds,
)
from chapters.ch06.assistant.tools import make_tools
from helix.agent import Agent
from helix.artifact import Subtype, genesis
from helix.eval.effort import EffortSetting
from helix.hooks import HookPoint
from helix.memory.base import MemoryContext, Scope, ScopeKey
from helix.memory.code_skills import recalled_skill_tools
from helix.memory.hooks import DEFAULT_SKILL_K
from helix.memory.procedural import ProceduralMemory
from helix.tools import Tool

MODEL = "gpt-5.4-mini"
# Dollars per million tokens: input, output, cached input, as published.
PRICES = {MODEL: (0.75, 4.50, 0.075)}
MEMORIES = Path(__file__).parent / "memories"
GLOBAL = ScopeKey(Scope.GLOBAL, "*")
# The function every learned memory must match, and its contract. It is
# the procedural memory listing 6.1's context search found and a person
# promoted.
BOOTSTRAPPED_MEMORY = {
    "name": "find_shared_starts",
    "body_kind": "python",
    "file": "find_shared_starts.searched.py",
    "description": (
        "Find every start time, by day, when the user and all the listed"
        " attendees are free for a meeting of the given length. days are"
        " mon to fri; earliest is the first allowed start and latest_end"
        " the time the meeting must end by, both HH:MM or empty. Returns"
        " the workable starts, earliest first."
    ),
}
# The same contract written by hand, kept to check the searched memory.
HAND_WRITTEN_MEMORY = {**BOOTSTRAPPED_MEMORY, "file": "find_shared_starts.py", "created_by": "human"}
# The second task class's bootstrapped memory: one start that works on
# every day.
BOOTSTRAPPED_SAME_TIME = {
    "name": "find_common_starts",
    "body_kind": "python",
    "file": "find_common_starts.py",
    "description": (
        "Find every start time that works for the user and all the listed"
        " attendees on every one of the given days, for a meeting of the"
        " given length, so it can be held at the same time each day. days"
        " are mon to fri; earliest is the first allowed start and"
        " latest_end the time the meeting must end by, both HH:MM or"
        " empty. Returns the common starts, earliest first."
    ),
}
# The same method written as a text procedure, kept for comparison.
TEXT_PROCEDURE = {
    "name": "multi_attendee_scheduling",
    "body_kind": "instructions",
    "description": (
        "Book a meeting with several people under day and time limits."
    ),
    "file": "multi_attendee_scheduling.md",
}

PROMPT = """You are Maya Rivera's calendar assistant. Complete the user's request
with the calendar tools. Requests are about the current working week,
Monday to Friday; the calendar shows its dates. Working hours are 09:00 to
17:00. When you are done, reply with what you booked."""


class ReadOnlyCalendar:
    """What a recalled function may touch: reading calendars only."""

    def __init__(self, env: SchedulingEnvironment) -> None:
        self.read_calendar = env.read_calendar


async def install_memory(
    entry: dict[str, Any] | None,
) -> ProceduralMemory | None:
    """A procedural tier holding one memory entry, or None."""
    if entry is None:
        return None
    tier = ProceduralMemory(embedding_fn=embed)
    body = (
        entry.get("body")
        or (MEMORIES / entry["file"]).read_text(encoding="utf-8").strip()
    )
    await tier.save_skill(
        entry["name"],
        entry["description"],
        body,
        body_kind=entry["body_kind"],
        scope_key=GLOBAL,
        metadata={"created_by": entry.get("created_by", "human")},
    )
    return tier


def build_agent(
    env: SchedulingEnvironment,
    *,
    effort: str,
    memory: ProceduralMemory | None,
    recalled_tools: list[Tool] | None = None,
    model: str = MODEL,
    skill_k: int = DEFAULT_SKILL_K,
) -> Agent:
    agent = Agent(
        system_prompt=genesis(
            "prompt.ch06.assistant", Subtype.PROMPT, PROMPT
        ),
        tools=make_tools(env) + list(recalled_tools or []),
        model=model,
        max_iterations=16,
        max_tool_calls=24,
        memory_tiers=(
            {"procedural": memory} if memory is not None else None
        ),
    )
    if agent.memory_hooks is not None:
        agent.memory_hooks.read_only = True
        agent.memory_hooks.skill_k = skill_k
    agent.hooks.register(HookPoint.PRE_MODEL, EffortSetting(effort))
    return agent


async def run_task(
    task: dict[str, Any],
    effort: str,
    *,
    memory: ProceduralMemory | None = None,
    model: str = MODEL,
    skill_k: int = DEFAULT_SKILL_K,
):
    """One task at one setting -> (solved, reason, trajectory)."""
    env = SchedulingEnvironment(task)
    context = MemoryContext(user_id=env.user)
    # One k for both, so the functions the agent can call are the ones
    # its context names.
    tools = (
        await recalled_skill_tools(
            memory,
            task["text"],
            context,
            k=skill_k,
            bind={"api": ReadOnlyCalendar(env)},
        )
        if memory
        else []
    )
    agent = build_agent(
        env,
        effort=effort,
        memory=memory,
        recalled_tools=tools,
        model=model,
        skill_k=skill_k,
    )
    with recorded_seconds() as seconds:
        _, trajectory = await agent.run(task["text"], context)
    restore_seconds(trajectory, seconds)
    verdict = env.check()
    trajectory.metadata["check"] = verdict
    return verdict["success"], verdict["reason"], trajectory
