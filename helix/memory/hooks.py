"""Memory as a hook pair. SPEC §6.3 and §16.2.3, v0.3.

The agent loop does not read memory. A PRE_MODEL hook injects what the
tiers recall (facts, learned procedures, and episodes) into the system
message on the first iteration, and a
SESSION_END hook writes the finished trajectory back to the episodic tier.
Packaging both in a wrapper the Agent constructs from its `memory_tiers`
(the same shape as Guardrail) means improver clones made through
`with_artifacts` keep them, and the loop keeps its eight lines.

The wrapper also records which entries were recalled, on the trajectory and
on the bus. That record is what Chapter 5's utility signal credits from:
without it, an outcome can never be attributed back to the memory that
shaped it.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

from helix.hooks import HookPoint, HookRegistry
from helix.memory.base import (
    MemoryContext,
    MemoryEntry,
    MemoryQuery,
    MemoryTier,
    Scope,
    ScopeKey,
)
from helix.observability.bus import EventBus, get_bus
from helix.observability.events import MemoryRead, MemoryWritten
from helix.trajectory import StepKind, Trajectory

# How many episodic memories to retrieve and inject at the start of a run.
DEFAULT_EPISODIC_K = 3
# How many semantic facts to surface in the system context.
DEFAULT_SEMANTIC_K = 6
# How many stored procedures to inject; one keeps a skill's effect measurable.
DEFAULT_SKILL_K = 1

# Keys the wrapper leaves on trajectory.metadata for downstream consumers.
RECALLED_KEY = "memory_recalled"
INJECTED_KEY = "memory_injected"
WRITTEN_KEY = "memory_written"


def context_from_trajectory(trajectory: Trajectory) -> MemoryContext:
    """Rebuild the MemoryContext the agent stashed on the trajectory."""
    raw = trajectory.metadata.get("memory_context") or {}
    return MemoryContext(
        session_id=raw.get("session_id"),
        user_id=raw.get("user_id"),
        org_id=raw.get("org_id"),
    )


def _system_text(message: dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, list) and content and isinstance(content[0], dict):
        return str(content[0].get("text", ""))
    return str(content or "")


def _set_system_text(message: dict[str, Any], text: str) -> None:
    """Rewrite the system text in place, keeping any cache_control block."""
    content = message.get("content")
    if isinstance(content, list) and content and isinstance(content[0], dict):
        content[0]["text"] = text
    else:
        message["content"] = text


def render_semantic(facts: list[MemoryEntry]) -> str:
    lines = ["", "## Known facts about this user / organization"]
    for f in facts:
        k = f.content.get("key", "")
        v = f.content.get("value", "")
        if k and v:
            lines.append(f"- {k}: {v}")
    return "\n".join(lines) if len(lines) > 2 else ""


async def recall_skills(tier: MemoryTier | None, task: str, context: MemoryContext, k: int) -> list[MemoryEntry]:
    """The procedural recall for a request: the one retrieval path for skills.

    The memory hooks call it to write skills into the context, and a harness
    that turns a recalled function into a tool calls it before building the
    agent, so the function the agent can call is exactly what memory recalled.
    """
    if tier is None or k <= 0:
        return []
    try:
        return await tier.read(MemoryQuery(text=task, k=k), context)
    except Exception:
        return []


def render_procedural(skills: list[MemoryEntry]) -> str:
    """Stored procedures as context: each skill's name, then its body.

    A skill is the memory chapter 6 learns, so it is rendered whole; a
    procedure cut short is a different procedure.
    """
    lines = ["", "## Learned procedures"]
    for s in skills:
        name = s.content.get("skill_name") or s.content.get("name") or "skill"
        if s.content.get("body_kind") == "python":
            # A code skill reaches the model as a tool; the context says it
            # exists and what it does, not how its code works.
            description = str(s.content.get("description") or "").strip()
            lines.extend(["", f"### {name}", f"Use the tool `{name}`. {description}".strip()])
            continue
        body = str(s.content.get("body") or s.content.get("description") or "").strip()
        if body:
            lines.extend(["", f"### {name}", body])
    return "\n".join(lines) if len(lines) > 2 else ""


def render_episodic(memories: list[MemoryEntry]) -> str:
    lines = ["", "## Relevant past interactions"]
    for m in memories:
        q = m.content.get("user_message") or m.content.get("task") or "(prior task)"
        a = m.content.get("final_output") or m.content.get("answer") or ""
        lines.append(f"- Asked: {str(q)[:200]}")
        snippet_a = str(a)[:300]
        if snippet_a:
            lines.append(f"  Answered: {snippet_a}")
        # Chapter 6's reflector writes lessons as episodic entries.
        lesson = m.content.get("lesson")
        if lesson:
            lines.append(f"  Lesson: {str(lesson)[:300]}")
    return "\n".join(lines) if len(lines) > 2 else ""


@dataclass
class MemoryHooks:
    """Bind a set of tiers to an agent's hook registry. SPEC §16.2.3.

    This is the harness's fixed memory application mechanism. Learning updates
    the entries and values it reads, not the hook timing or rendering logic.
    Order in the system message is deliberate: the artifact's prompt first
    (it is the thing under improvement), then semantic facts, then learned
    procedures, then episodic recall, so the model sees the policy and how to
    do the task before the per-interaction context.
    """

    tiers: dict[str, MemoryTier]
    episodic_k: int = DEFAULT_EPISODIC_K
    semantic_k: int = DEFAULT_SEMANTIC_K
    skill_k: int = DEFAULT_SKILL_K
    bus: EventBus | None = None
    model: str = ""
    system_prompt_ref: tuple[str, int] | None = None
    # A fresh entry starts unproven, not distrusted: half way, so the first
    # outcome that credits it can move it either way.
    initial_utility: float = 0.5
    # Recall without write-back: evaluation runs read a memory snapshot and
    # must not grow it, or later questions would recall earlier answers.
    read_only: bool = False
    _registered: bool = field(default=False, init=False, repr=False)

    def register(self, hooks: HookRegistry) -> None:
        if self._registered:
            return
        hooks.register(HookPoint.PRE_MODEL, self.inject)
        hooks.register(HookPoint.SESSION_END, self.write_back)
        self._registered = True

    def _bus(self) -> EventBus:
        return self.bus or get_bus()

    # ---------------- PRE_MODEL: recall ----------------

    async def inject(self, messages: list[dict[str, Any]], trajectory: Trajectory, **_: Any) -> None:
        """Append recalled memory to the system message, once per run.

        Fires on every PRE_MODEL; the first-iteration check keeps it to one
        injection so tool loops do not re-read memory on every turn.
        """
        if trajectory.metadata.get(INJECTED_KEY):
            return None
        if any(s.kind == StepKind.MODEL_CALL for s in trajectory.steps):
            return None
        trajectory.metadata[INJECTED_KEY] = True
        if not messages or messages[0].get("role") != "system":
            return None

        context = context_from_trajectory(trajectory)
        task = trajectory.task
        sections: list[str] = []
        recalled: list[dict[str, Any]] = []

        sem = self.tiers.get("semantic")
        if sem is not None and (context.user_id or context.org_id):
            try:
                facts = await sem.read(MemoryQuery(text=task, k=self.semantic_k), context)
            except Exception:
                facts = []
            block = render_semantic(facts)
            if block:
                sections.append(block)
            recalled.extend(self._describe("semantic", facts, trajectory))

        # Procedures come before episodes: a skill says how to do the task,
        # an episode only shows one time it was done.
        proc = self.tiers.get("procedural")
        if proc is not None and self.skill_k > 0:
            skills = await recall_skills(proc, task, context, self.skill_k)
            block = render_procedural(skills)
            if block:
                sections.append(block)
            recalled.extend(self._describe("procedural", skills, trajectory))

        epi = self.tiers.get("episodic")
        if epi is not None:
            try:
                memories = await epi.read(MemoryQuery(text=task, k=self.episodic_k), context)
            except Exception:
                memories = []
            block = render_episodic(memories)
            if block:
                sections.append(block)
            recalled.extend(self._describe("episodic", memories, trajectory))

        trajectory.metadata[RECALLED_KEY] = recalled
        if sections:
            base = _system_text(messages[0]).rstrip()
            _set_system_text(messages[0], "\n".join([base, *sections]))
        await self._publish_reads(trajectory, recalled)
        return None

    def _describe(self, tier: str, entries: list[MemoryEntry], trajectory: Trajectory) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for e in entries:
            ref = e.artifact_ref
            if ref is not None:
                # Step 0 is the run's setup step, where the system prompt ref
                # already lives; recalled entries join it so artifacts_used
                # names every artifact that shaped the run.
                trajectory.record_artifact(0, tuple(ref))
            out.append({
                "tier": tier,
                "entry_id": e.id,
                "artifact_ref": list(ref) if ref else None,
                "scope": e.scope_key.to_str(),
                **({"retrieval_score": e.metadata["retrieval_score"]}
                   if "retrieval_score" in e.metadata else {}),
            })
        return out

    # ---------------- SESSION_END: write-back ----------------

    async def write_back(self, trajectory: Trajectory, **_: Any) -> None:
        """Persist the completed run to episodic memory for its user.

        User scope when a user_id is known, so the next session can recall
        this one; session scope for anonymous traffic; nothing when the run
        carried no context. Org and global writes are a deliberate act the
        caller performs. A refused run is not written: there is no answer
        worth remembering.
        """
        epi = self.tiers.get("episodic")
        if epi is None or self.read_only:
            return None
        if trajectory.metadata.get("refusal"):
            return None
        context = context_from_trajectory(trajectory)
        if context.user_id:
            scope_key = ScopeKey(Scope.USER, context.user_id)
        elif context.session_id:
            scope_key = ScopeKey(Scope.SESSION, context.session_id)
        else:
            return None

        output = trajectory.final_output
        entry = MemoryEntry(
            id=str(uuid.uuid4()),
            scope_key=scope_key,
            score=self.initial_utility,
            content={
                "trajectory_id": trajectory.id,
                "user_message": trajectory.task,
                "final_output": output if isinstance(output, str) else (str(output) if output is not None else ""),
                "outcome": trajectory.outcome.value if trajectory.outcome else "unknown",
                "num_steps": len(trajectory.steps),
            },
            metadata={
                "model": self.model,
                "system_prompt_ref": list(self.system_prompt_ref) if self.system_prompt_ref else None,
                "user_id": context.user_id,
                "org_id": context.org_id,
                "created_by": "agent",
            },
        )
        try:
            entry_id = await epi.write(entry)
        except Exception:
            # Memory write failures must never break the hot path.
            return None
        trajectory.metadata[WRITTEN_KEY] = {
            "tier": "episodic",
            "entry_id": entry_id,
            "artifact_ref": list(entry.artifact_ref) if entry.artifact_ref else None,
        }
        await self._bus().publish(
            MemoryWritten(
                trajectory_id=trajectory.id,
                tier="episodic",
                entry_id=entry_id,
                scope=scope_key.to_str(),
                artifact_ref=list(entry.artifact_ref) if entry.artifact_ref else None,
                operator="add",
            )
        )
        return None

    async def _publish_reads(self, trajectory: Trajectory, recalled: list[dict[str, Any]]) -> None:
        """Emit one MemoryRead per tier for the entries the run recalled."""
        by_tier: dict[str, list[dict[str, Any]]] = {}
        for r in recalled:
            by_tier.setdefault(r["tier"], []).append(r)
        for tier, rows in by_tier.items():
            await self._bus().publish(
                MemoryRead(
                    trajectory_id=trajectory.id,
                    tier=tier,
                    n_entries=len(rows),
                    entry_ids=[r["entry_id"] for r in rows],
                    artifact_refs=[r["artifact_ref"] for r in rows if r["artifact_ref"]],
                )
            )
