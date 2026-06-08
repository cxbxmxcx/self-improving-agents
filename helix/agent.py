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
from helix.hooks import HookPoint, HookRegistry, Refusal
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


def _first_refusal(hook_results: list[Any]) -> Refusal | None:
    """Return the first Refusal in a list of hook results, or None.

    Used by the agent loop at PRE_MODEL and PRE_OUTPUT to short-circuit on
    guardrail refusals. SPEC §6.2.
    """
    for r in hook_results:
        if isinstance(r, Refusal):
            return r
    return None


def _first_output_patch(hook_results: list[Any]) -> Any:
    """Return the first ('patched', OutputGuardrailPayload) entry's payload, or None."""
    for r in hook_results:
        if isinstance(r, tuple) and len(r) == 2 and r[0] == "patched":
            return r[1]
    return None


def _apply_input_patch(memory, hook_results: list[Any]) -> None:
    """If a PRE_MODEL hook returned ('patched', InputGuardrailPayload), rewrite
    the user message in working memory with the patched question. SPEC §6.2.
    """
    for r in hook_results:
        if isinstance(r, tuple) and len(r) == 2 and r[0] == "patched":
            patched = r[1]
            new_question = getattr(patched, "question", None)
            if new_question is None:
                continue
            msgs = memory._messages if hasattr(memory, "_messages") else None
            if msgs is None:
                return
            for i in range(len(msgs) - 1, -1, -1):
                if msgs[i].get("role") == "user":
                    msgs[i] = {**msgs[i], "content": new_question}
                    return
            return


class Agent:
    def __init__(
        self,
        system_prompt: Artifact,
        tools: list[Tool] | None = None,
        model: str = "gpt-4o-mini",
        temperature: float | None = None,
        max_iterations: int = 10,
        max_tool_calls: int = 20,
        hooks: HookRegistry | None = None,
        bus: EventBus | None = None,
        memory_tiers: dict[str, MemoryTier] | None = None,
        guardrails: list | None = None,  # list[Guardrail], avoid circular import
        composites: list[Artifact] | None = None,
    ) -> None:
        if not isinstance(system_prompt.content, str):
            raise TypeError("system_prompt artifact must have string content")
        self.system_prompt = system_prompt
        # Composite artifacts this agent declares (SPEC §18). Each binds a
        # coupled cluster of constituents that is improved jointly by a
        # ComposedSearch and deployed as a unit. Declaring a composite opts its
        # constituents out of solo improvement (SPEC §18.7).
        self.composites: dict[str, Artifact] = {
            c.id: c for c in (composites or [])
        }
        self.tools: dict[str, Tool] = {t.name: t for t in (tools or [])}
        self.model = model
        # Sampling temperature passed through to the LLM call; None leaves the
        # provider default. Set low to make evaluation near-deterministic so a
        # single rollout is a reliable measurement (the search uses the proposer,
        # not the agent, for exploration, so a cold agent does not hurt it).
        self.temperature = temperature
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
        # Guardrails (SPEC §16.2.1). Each wraps a CODE artifact and registers
        # itself as a PRE_MODEL or PRE_OUTPUT hook at construction. The list
        # is kept around so Agent.with_artifacts can carry it into a clone.
        self.guardrails: list = list(guardrails or [])
        for g in self.guardrails:
            self.hooks.register(g.hook_point, g.make_hook_handler())

    def with_artifacts(self, overrides: dict[str, Artifact]) -> "Agent":
        """Return a new Agent with the given artifacts swapped in. SPEC §15.2.

        The improver uses this to test candidates without the user having to
        write a factory function. For each artifact id in `overrides`, the
        matching slot on the agent is replaced; everything else (tools, model,
        memory tiers, guardrails) carries over either by reference or as a
        rebound clone.

        Dispatch by artifact:
          - `system_prompt`: swapped when an override's id matches.
          - tool descriptions: a TextDescriptionTool whose description artifact
            id matches an override gets a fresh description (Ch 3). Plain Tools
            and ToolFromArtifact instances carry over unchanged.
          - `guardrails`: each guardrail whose artifact id matches an override
            is rebuilt with the new artifact. Other guardrails carry over.

        The clone gets a fresh HookRegistry. Hooks registered on the original
        do not carry over (each Agent owns its own hook state). Guardrails are
        re-registered on the clone via the normal __init__ path.
        """
        from helix.guardrails import Guardrail  # local import: avoid circular
        from helix.tools import TextDescriptionTool

        new_system_prompt = self.system_prompt
        for art in overrides.values():
            if art.id == self.system_prompt.id:
                if not isinstance(art.content, str):
                    raise TypeError("system_prompt artifact must have string content")
                new_system_prompt = art

        # Rebuild tools: a TextDescriptionTool whose description artifact id
        # appears in overrides gets a fresh copy with the candidate
        # description. Other tools (plain Tool, ToolFromArtifact) carry over
        # by reference.
        new_tools: list = []
        for t in self.tools.values():
            desc_art = getattr(t, "description_artifact", None)
            if (
                isinstance(t, TextDescriptionTool)
                and desc_art is not None
                and desc_art.id in overrides
            ):
                new_tools.append(TextDescriptionTool(
                    name=t.name,
                    fn=t.fn,
                    description_artifact=overrides[desc_art.id],
                ))
            else:
                new_tools.append(t)

        # Rebuild guardrails: any whose artifact id appears in overrides gets
        # a new Guardrail (same phase, same fail_open) backed by the candidate
        # artifact. Others carry over with the same artifact.
        new_guardrails: list[Guardrail] = []
        for g in self.guardrails:
            if g.artifact.id in overrides:
                new_guardrails.append(Guardrail(
                    artifact=overrides[g.artifact.id],
                    phase=g.phase,
                    fail_open=g.fail_open,
                ))
            else:
                new_guardrails.append(g)

        # Composites carry over, swapped when an override matches a composite id.
        new_composites = [
            overrides.get(c.id, c) for c in self.composites.values()
        ]

        return Agent(
            system_prompt=new_system_prompt,
            tools=new_tools,
            model=self.model,
            temperature=self.temperature,
            max_iterations=self.max_iterations,
            max_tool_calls=self.max_tool_calls,
            hooks=None,  # fresh HookRegistry; guardrails re-register in __init__
            bus=self.bus,
            memory_tiers=dict(self.memory),
            guardrails=new_guardrails,
            composites=new_composites,
        )

    def find_artifact(self, artifact_id: str) -> Artifact | None:
        """Resolve an artifact id to the Artifact bound on this agent, or None.

        Searches the system prompt, tool code/description artifacts, and
        guardrail artifacts. Improvers use this to classify a target's layer for
        the online-safety check (SPEC §17.3) regardless of which slot it fills.
        """
        for art in self._bound_artifacts():
            if art.id == artifact_id:
                return art
        return None

    def _bound_artifacts(self) -> list[Artifact]:
        arts: list[Artifact] = [self.system_prompt]
        for t in self.tools.values():
            for attr in ("description_artifact", "code_artifact"):
                a = getattr(t, attr, None)
                if isinstance(a, Artifact):
                    arts.append(a)
        for g in self.guardrails:
            a = getattr(g, "artifact", None)
            if isinstance(a, Artifact):
                arts.append(a)
        arts.extend(self.composites.values())
        return arts

    def composite_constituent_ids(self) -> set[str]:
        """Ids of every constituent bound inside one of this agent's composites.

        Declaring a composite opts its constituents out of solo improvement, so
        improvers consult this set to reject a solo target that belongs to a
        composite (SPEC §18.7).
        """
        ids: set[str] = set()
        for comp in self.composites.values():
            for ref in comp.constituent_refs:
                ids.add(ref[0])
        return ids

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
        # The wiring step (SPEC §17.1): online improvers subscribe to the
        # agent's SESSION_END hook here. Offline improvers have no on_attached
        # and are unaffected (attach is a display-only convenience for them).
        on_attached = getattr(improver, "on_attached", None)
        if callable(on_attached):
            on_attached()

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
                pre_model_results = await self.hooks.fire(
                    HookPoint.PRE_MODEL,
                    messages=memory.messages(),
                    trajectory=trajectory,
                )
                # SPEC §6.2: a PRE_MODEL hook may return a Refusal to terminate
                # the run, or ("patched", payload) to swap in a rewritten
                # input. Guardrails (§16.2.1) are the canonical producer of
                # both. The agent loop inspects results in registration order;
                # the first short-circuit wins.
                refusal = _first_refusal(pre_model_results)
                if refusal is not None:
                    refusal_msg = f"[refused] {refusal.reason}"
                    trajectory.metadata["refusal"] = {
                        "reason": refusal.reason,
                        "source": refusal.source,
                        "phase": "pre_model",
                    }
                    trajectory.complete(refusal_msg, Outcome.COMPLETED)
                    await self.hooks.fire(HookPoint.SESSION_END, trajectory=trajectory)
                    await self.bus.publish(
                        TrajectoryCompleted(
                            trajectory_id=trajectory.id,
                            task=task[:200],
                            outcome=trajectory.outcome.value,
                            num_steps=len(trajectory.steps),
                            final_output=refusal_msg[:500],
                        )
                    )
                    return refusal_msg, trajectory
                _apply_input_patch(memory, pre_model_results)

                extra: dict[str, Any] = {}
                if self.temperature is not None:
                    extra["temperature"] = self.temperature
                response = await helix_acompletion(
                    model=self.model,
                    messages=memory.messages(),
                    tools=tool_specs,
                    **extra,
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
                    pre_output_results = await self.hooks.fire(
                        HookPoint.PRE_OUTPUT, response=output, trajectory=trajectory
                    )
                    # SPEC §6.2: a PRE_OUTPUT hook may return a Refusal to
                    # terminate the run with a refusal message in place of the
                    # output, or ("patched", payload) to rewrite the answer.
                    refusal = _first_refusal(pre_output_results)
                    if refusal is not None:
                        output = f"[refused] {refusal.reason}"
                        trajectory.metadata["refusal"] = {
                            "reason": refusal.reason,
                            "source": refusal.source,
                            "phase": "pre_output",
                        }
                    else:
                        patched = _first_output_patch(pre_output_results)
                        if patched is not None:
                            output = patched.answer
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

                    tool_call_step = trajectory.append(
                        StepKind.TOOL_CALL,
                        {"name": tc.function.name, "arguments": args},
                    )
                    # Record artifact lineage for artifact-backed tools
                    # (TextDescriptionTool, ToolFromArtifact). Plain Tools have
                    # no artifact_refs and contribute nothing here.
                    tool_obj = self.tools.get(tc.function.name)
                    refs = getattr(tool_obj, "artifact_refs", None)
                    if callable(refs):
                        for ref in refs():
                            trajectory.record_artifact(tool_call_step.index, ref)
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
