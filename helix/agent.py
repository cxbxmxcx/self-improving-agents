"""The canonical Agent loop.

This is the eight-line skeleton from Spec section 6.1, fleshed out with hook
firings, trajectory recording, and LiteLLM model calls. It does not change
after Chapter 2. Every later capability is a hook, an artifact under search,
or a signal being measured.

Wiring:
- system_prompt is an Artifact (kind=prompt); read once at session start.
- tools is a list of Tool objects (helix.tools).
- memory_tiers is a dict of {tier_name: MemoryTier}. Working memory is built
  per-run; episodic/semantic/procedural are supplied by the caller and shared
  across runs.
- Each Agent.run() takes an optional MemoryContext (session_id, user_id,
  org_id). The context determines which scope levels memory reads target.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any

from helix._caching import cacheable_system_message
from helix.llm_call import acompletion as helix_acompletion
from helix.artifact import Artifact
from helix.hooks import HookPoint, HookRegistry
from helix.memory.base import (
    MemoryContext,
    MemoryEntry,
    MemoryQuery,
    MemoryTier,
    Scope,
    ScopeKey,
)
from helix.memory.working import WorkingMemory
from helix.observability import span
from helix.observability.bus import EventBus, get_bus
from helix.observability.events import (
    TrajectoryCompleted,
    TrajectoryStarted,
    TrajectoryStepRecorded,
)
from helix.tools import Tool
from helix.trajectory import Outcome, StepKind, Trajectory


# How many episodic memories to retrieve and inject at the start of a run.
_EPISODIC_K = 3
# How many semantic facts to surface in the system context.
_SEMANTIC_K = 6


class Agent:
    def __init__(
        self,
        system_prompt: Artifact,
        tools: list[Tool] | None = None,
        model: str = "gpt-4o-mini",
        max_iterations: int = 10,
        max_tool_calls: int = 20,
        hooks: HookRegistry | None = None,
        bus: EventBus | None = None,
        memory_tiers: dict[str, MemoryTier] | None = None,
    ) -> None:
        if not isinstance(system_prompt.content, str):
            raise TypeError("system_prompt artifact must have string content")
        self.system_prompt = system_prompt
        self.tools: dict[str, Tool] = {t.name: t for t in (tools or [])}
        self.model = model
        self.max_iterations = max_iterations
        self.max_tool_calls = max_tool_calls
        self.hooks = hooks or HookRegistry()
        self.bus = bus or get_bus()
        # Persistent memory tiers, supplied by the caller. Working memory is
        # built per-run; episodic/semantic/procedural are shared instances.
        self.memory: dict[str, MemoryTier] = dict(memory_tiers or {})
        # Improvers registered on this agent. They are not invoked inline by
        # the hot path. They consume trajectories via the event bus.
        self.improvers: list = []  # type: list[Improver] but avoid circular import

    def with_artifacts(self, overrides: dict[str, Artifact]) -> "Agent":
        """Return a new Agent with the given artifacts swapped in. SPEC §15.2.

        The improver uses this to test candidates without the user having to
        write a factory function. For each artifact id in `overrides`, the
        matching slot on the agent is replaced; everything else (tools, model,
        hooks, memory tiers) carries over by reference.

        The single-artifact Ch 2 case swaps `system_prompt`. Future chapters
        will swap tool descriptions, tool code, memory entries, and guardrails
        by the same mechanism — the dispatch is by `artifact.kind` plus
        `artifact.id` matching the agent's current configuration.
        """
        new_system_prompt = self.system_prompt
        for art in overrides.values():
            # System prompt match: candidate has same id as current prompt.
            if art.id == self.system_prompt.id:
                if not isinstance(art.content, str):
                    raise TypeError("system_prompt artifact must have string content")
                new_system_prompt = art

        # Future kinds (tool descriptions, tool code, memory entries,
        # guardrails) wire here as the framework grows. For now Ch 2 only
        # routes the system prompt.
        return Agent(
            system_prompt=new_system_prompt,
            tools=list(self.tools.values()),
            model=self.model,
            max_iterations=self.max_iterations,
            max_tool_calls=self.max_tool_calls,
            hooks=self.hooks,
            bus=self.bus,
            memory_tiers=dict(self.memory),
        )

    def attach_improver(self, improver) -> None:
        """Register an Improver on this agent.

        Idempotent on improver_id, not on target_artifact_id: multiple
        Improvers may target the same artifact with different Search methods
        (e.g. SPO and GEPA both improving the system prompt). The Archive
        arbitrates by score regardless of which Improver produced each
        candidate.
        """
        for existing in self.improvers:
            if existing.improver_id == improver.improver_id:
                return
        self.improvers.append(improver)

    def detach_improver(self, improver_id_or_target: str) -> None:
        """Detach by improver_id (preferred) or target_artifact_id (legacy)."""
        from helix.improvement.promotion import unregister_improver_archive

        removed = [
            i for i in self.improvers
            if i.improver_id == improver_id_or_target
            or i.target_artifact_id == improver_id_or_target
        ]
        for i in removed:
            unregister_improver_archive(i.improver_id)

        self.improvers = [
            i for i in self.improvers
            if i.improver_id != improver_id_or_target
            and i.target_artifact_id != improver_id_or_target
        ]

    async def run(
        self,
        task: str,
        context: MemoryContext | None = None,
    ) -> tuple[Any, Trajectory]:
        """Run the agent against `task`, optionally scoped to a memory context.

        When context is provided (with at least session_id and ideally user_id),
        the agent enriches the system prompt with relevant episodic memories
        and semantic facts from the matching scopes. At session end, the
        completed trajectory is written to episodic memory.

        Without a context, the agent runs statelessly: working memory only,
        nothing read or written across runs. This is the Ch 2 mode.
        """
        if context is None:
            context = MemoryContext()
        trajectory = Trajectory(task=task)

        # Build the system prompt: artifact content + memory-augmented context.
        system_text = await self._build_system_prompt(self.system_prompt.content, task, context)

        # The system prompt is the same across every question in a round
        # (the reference artifact, or the candidate artifact). Mark it
        # cacheable so Anthropic prompt caching kicks in on repeated runs.
        memory = WorkingMemory(
            initial=[
                cacheable_system_message(system_text, self.model),
                {"role": "user", "content": task},
            ]
        )
        trajectory.record_artifact(0, self.system_prompt.ref)
        # Stash the context on the trajectory so SESSION_END hooks (including
        # episodic-write below) can read it.
        trajectory.metadata["memory_context"] = {
            "session_id": context.session_id,
            "user_id": context.user_id,
            "org_id": context.org_id,
        }

        await self.bus.publish(
            TrajectoryStarted(trajectory_id=trajectory.id, task=task[:200])
        )
        await self.hooks.fire(HookPoint.SESSION_START, trajectory=trajectory)

        tool_specs = [t.to_openai_schema() for t in self.tools.values()] or None

        async with span("agent.run", task=task, model=self.model):
            for iteration in range(self.max_iterations):
                await self.hooks.fire(
                    HookPoint.PRE_MODEL,
                    messages=memory.messages(),
                    trajectory=trajectory,
                )

                response = await helix_acompletion(
                    model=self.model,
                    messages=memory.messages(),
                    tools=tool_specs,
                )
                msg = response.choices[0].message
                assistant_entry: dict[str, Any] = {"role": "assistant", "content": msg.content or ""}
                if msg.tool_calls:
                    assistant_entry["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                        }
                        for tc in msg.tool_calls
                    ]
                memory.append(assistant_entry)
                trajectory.append(StepKind.MODEL_CALL, {"response": assistant_entry})

                await self.hooks.fire(
                    HookPoint.POST_MODEL, response=msg, trajectory=trajectory
                )

                if not msg.tool_calls:
                    output = msg.content or ""
                    await self.hooks.fire(
                        HookPoint.PRE_OUTPUT, response=output, trajectory=trajectory
                    )
                    trajectory.complete(output, Outcome.COMPLETED)
                    await self.hooks.fire(HookPoint.SESSION_END, trajectory=trajectory)
                    await self._write_to_episodic(task, output, trajectory, context)
                    await self.bus.publish(
                        TrajectoryCompleted(
                            trajectory_id=trajectory.id,
                            task=task[:200],
                            outcome=trajectory.outcome.value,
                            num_steps=len(trajectory.steps),
                            final_output=(output or "")[:500],
                        )
                    )
                    return output, trajectory

                # Per-run tool-call cap: stop cascading if we've hit the limit.
                # This protects against runaway costs from a single question.
                tool_calls_so_far = sum(1 for s in trajectory.steps if s.kind == StepKind.TOOL_CALL)

                for tc in msg.tool_calls:
                    if tool_calls_so_far >= self.max_tool_calls:
                        # Cap reached; cancel remaining tool calls in this turn.
                        result: Any = {"error": f"tool-call cap of {self.max_tool_calls} reached for this run"}
                        trajectory.append(
                            StepKind.TOOL_CALL,
                            {"name": tc.function.name, "arguments": {}, "cap_hit": True},
                        )
                        trajectory.append(StepKind.TOOL_RESULT, {"result": result})
                        memory.append(
                            {
                                "role": "tool",
                                "tool_call_id": tc.id,
                                "content": json.dumps(result, default=str),
                            }
                        )
                        continue
                    tool_calls_so_far += 1
                    args = json.loads(tc.function.arguments or "{}")
                    cancel_results = await self.hooks.fire(
                        HookPoint.PRE_TOOL,
                        tool_name=tc.function.name,
                        arguments=args,
                        trajectory=trajectory,
                    )
                    if "cancel" in cancel_results:
                        result: Any = {"cancelled": True}
                    else:
                        tool = self.tools.get(tc.function.name)
                        if tool is None:
                            result = {"error": f"unknown tool: {tc.function.name}"}
                        else:
                            try:
                                result = await tool.call(**args)
                            except Exception as exc:  # surface to the model, don't crash
                                result = {"error": str(exc)}

                    trajectory.append(
                        StepKind.TOOL_CALL,
                        {"name": tc.function.name, "arguments": args},
                    )
                    trajectory.append(StepKind.TOOL_RESULT, {"result": result})
                    memory.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": json.dumps(result, default=str),
                        }
                    )
                    await self.hooks.fire(
                        HookPoint.POST_TOOL,
                        tool_name=tc.function.name,
                        result=result,
                        trajectory=trajectory,
                    )

                await self.hooks.fire(HookPoint.END_OF_TURN, trajectory=trajectory)

            trajectory.complete(None, Outcome.TIMED_OUT)
            await self.hooks.fire(HookPoint.SESSION_END, trajectory=trajectory)
            await self._write_to_episodic(task, None, trajectory, context)
            await self.bus.publish(
                TrajectoryCompleted(
                    trajectory_id=trajectory.id,
                    task=task[:200],
                    outcome=trajectory.outcome.value,
                    num_steps=len(trajectory.steps),
                    final_output=None,
                )
            )
            return None, trajectory

    # ---------------- memory-augmentation helpers ----------------

    async def _build_system_prompt(
        self,
        base: str,
        task: str,
        context: MemoryContext,
    ) -> str:
        """Compose the system prompt: base + semantic facts + episodic recall.

        Order is deliberate. The artifact's base prompt comes first (it is
        the artifact under improvement). Semantic facts come next as a
        compact "what we know about this user/org" section. Episodic recall
        comes last as "recent similar interactions you handled." The model
        sees the artifact-as-policy first, then the per-interaction context.
        """
        sections: list[str] = [base.rstrip()]

        # Semantic facts: any scope visible to this context.
        sem = self.memory.get("semantic")
        if sem is not None and (context.user_id or context.org_id):
            try:
                facts = await sem.read(
                    MemoryQuery(text=task, k=_SEMANTIC_K),
                    context,
                )
            except Exception:
                facts = []
            if facts:
                lines = ["", "## Known facts about this user / organization"]
                for f in facts:
                    k = f.content.get("key", "")
                    v = f.content.get("value", "")
                    if k and v:
                        lines.append(f"- {k}: {v}")
                if len(lines) > 2:
                    sections.append("\n".join(lines))

        # Episodic recall: relevant past trajectories from visible scopes.
        epi = self.memory.get("episodic")
        if epi is not None:
            try:
                memories = await epi.read(
                    MemoryQuery(text=task, k=_EPISODIC_K),
                    context,
                )
            except Exception:
                memories = []
            if memories:
                lines = ["", "## Relevant past interactions"]
                for m in memories:
                    q = m.content.get("user_message") or m.content.get("task") or "(prior task)"
                    a = m.content.get("final_output") or m.content.get("answer") or ""
                    snippet_q = str(q)[:200]
                    snippet_a = str(a)[:300]
                    lines.append(f"- Asked: {snippet_q}")
                    if snippet_a:
                        lines.append(f"  Answered: {snippet_a}")
                if len(lines) > 2:
                    sections.append("\n".join(lines))

        return "\n".join(sections)

    async def _write_to_episodic(
        self,
        task: str,
        output: str | None,
        trajectory: Trajectory,
        context: MemoryContext,
    ) -> None:
        """Persist the completed trajectory to episodic memory.

        Writes at the most-specific available scope: session if session_id
        is set, otherwise user, otherwise nothing. Per-org / global writes
        are intentionally NOT done automatically; the caller must do those
        deliberately to avoid scope leakage.
        """
        epi = self.memory.get("episodic")
        if epi is None:
            return

        # Pick the most-specific scope. Skip if no scope available.
        scope_key: ScopeKey | None = None
        if context.session_id:
            scope_key = ScopeKey(Scope.SESSION, context.session_id)
        elif context.user_id:
            scope_key = ScopeKey(Scope.USER, context.user_id)
        else:
            return

        entry = MemoryEntry(
            id=str(uuid.uuid4()),
            scope_key=scope_key,
            content={
                "trajectory_id": trajectory.id,
                "user_message": task,
                "final_output": output or "",
                "outcome": trajectory.outcome.value,
                "num_steps": len(trajectory.steps),
            },
            metadata={
                "model": self.model,
                "system_prompt_ref": list(self.system_prompt.ref),
                "user_id": context.user_id,
                "org_id": context.org_id,
            },
        )
        try:
            await epi.write(entry)
        except Exception:
            # Memory write failures must never break the hot path.
            pass
