"""The canonical Agent loop.

This is the eight-line skeleton from Spec §6.1, fleshed out with hook firings,
trajectory recording, and LiteLLM model calls. It does not change after Ch 2.
Every later capability is a hook, an artifact under search, or a signal being
measured.

v0 wiring:
- system_prompt is an Artifact (kind=prompt); read once at session start.
- tools is a list of Tool objects (helix.tools).
- working memory is the messages list.
- no improvement loop yet (that's v1, added as a separate driver script).
"""

from __future__ import annotations

import json
from typing import Any

import litellm

from helix._caching import cacheable_system_message
from helix.artifact import Artifact
from helix.hooks import HookPoint, HookRegistry
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
        # Improvers registered on this agent. They are not invoked inline by
        # the hot path. They consume trajectories via the event bus.
        self.improvers: list = []  # type: list[Improver] but avoid circular import

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
        self.improvers = [
            i for i in self.improvers
            if i.improver_id != improver_id_or_target
            and i.target_artifact_id != improver_id_or_target
        ]

    async def run(self, task: str) -> tuple[Any, Trajectory]:
        trajectory = Trajectory(task=task)
        # The system prompt is the same across every question in a round
        # (the reference artifact, or the candidate artifact). Mark it
        # cacheable so Anthropic prompt caching kicks in on repeated runs.
        memory = WorkingMemory(
            initial=[
                cacheable_system_message(self.system_prompt.content, self.model),
                {"role": "user", "content": task},
            ]
        )
        trajectory.record_artifact(0, self.system_prompt.ref)

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

                response = await litellm.acompletion(
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
