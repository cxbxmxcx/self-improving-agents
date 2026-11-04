"""The reflector: turn a rejected or corrected run into a lesson in memory.

At SESSION_END, after memory has written the episode, the reflector reads
the run's reward. A run that fell short, and that the decision model judges
worth a lesson, gets one System 2 call with the reflector prompt, and the
lesson is written to the episodic tier in the run's scope. From there it is
an ordinary entry: it keeps its initial value until a later request recalls
it, and chapter 5's learner moves its value like any other.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

from helix.hooks import HookPoint
from helix.memory.base import MemoryEntry, Scope, ScopeKey
from helix.memory.hooks import context_from_trajectory
from helix.observability.events import MemoryWritten
from helix.scaffold.common import Wrapper, actions_of, note_artifact, record, request_text, resolve
from helix.trajectory import Trajectory


class Reflector(Wrapper):
    name = "reflector"

    async def session_end(self, trajectory: Trajectory | None = None, **_: Any) -> None:
        scaffold = self.scaffold
        if trajectory is None or scaffold.reflector is None or scaffold.reward_fn is None:
            return None
        tier = self.agent.memory.get("episodic")
        hooks = self.agent.memory_hooks
        # Shadow and evaluation runs read memory without growing it.
        if tier is None or (hooks is not None and hooks.read_only) or trajectory.metadata.get("refusal"):
            return None
        reward = await resolve(scaffold.reward_fn(trajectory))
        if reward is None or reward >= 1.0:
            return None
        meta = record(trajectory)
        state = {"request": request_text(trajectory), "reply": str(trajectory.final_output or ""),
                 "actions": actions_of(trajectory), "reward": reward}
        state = await self.state("reflect", state, trajectory)
        if scaffold.decision_config.questions_for("reflect"):
            _, actions, _ = await self.decide("reflect", state, trajectory)
            if "reflect" not in actions:
                await self.publish(HookPoint.SESSION_END.value, trajectory, "skip", reward=reward)
                return None
        scope = self._scope(trajectory)
        if scope is None:
            return None
        model, params = scaffold.system2(self.agent, scaffold.reflection_model)
        user = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False, default=str)
        note_artifact(trajectory, scaffold.reflector.ref)
        try:
            lesson = await self.system2("reflect", model, params, str(scaffold.reflector.content), user, trajectory)
        except Exception as exc:
            if scaffold.on_error != "safe":
                raise
            meta["errors"].append({"hook": "reflect", "where": "reflection", "error": f"{type(exc).__name__}: {exc}"})
            return None
        if not lesson:
            return None
        await self._write(trajectory, scope, lesson, reward, model)
        return None

    @staticmethod
    def _scope(trajectory: Trajectory) -> ScopeKey | None:
        """The same scope rule as the episode write-back: the user, else the session."""
        context = context_from_trajectory(trajectory)
        if context.user_id:
            return ScopeKey(Scope.USER, context.user_id)
        if context.session_id:
            return ScopeKey(Scope.SESSION, context.session_id)
        return None

    async def _write(self, trajectory: Trajectory, scope: ScopeKey, lesson: str, reward: float, model: str) -> None:
        scaffold, hooks = self.scaffold, self.agent.memory_hooks
        context = context_from_trajectory(trajectory)
        entry = MemoryEntry(
            id=str(uuid.uuid4()),
            scope_key=scope,
            score=hooks.initial_utility if hooks is not None else 0.5,
            content={
                "trajectory_id": trajectory.id,
                "user_message": trajectory.task,
                "final_output": "",
                "summary": lesson,
                "lesson": lesson,
                "outcome": trajectory.outcome.value if trajectory.outcome else "unknown",
                "reward": reward,
            },
            metadata={
                "kind": "lesson",
                "created_by": "reflector",
                "reflector_ref": list(scaffold.reflector.ref),
                "model": model,
                "user_id": context.user_id,
                "org_id": context.org_id,
            },
        )
        meta = record(trajectory)
        try:
            entry_id = await self.agent.memory["episodic"].write(entry)
        except Exception as exc:
            # A failed write must not break the run that already finished.
            meta["errors"].append({"hook": "reflect", "where": "memory", "error": f"{type(exc).__name__}: {exc}"})
            return
        ref = list(entry.artifact_ref) if entry.artifact_ref else None
        meta["lesson"] = {"entry_id": entry_id, "text": lesson, "scope": scope.to_str(), "artifact_ref": ref}
        await self.agent.bus.publish(MemoryWritten(trajectory_id=trajectory.id, tier="episodic", entry_id=entry_id,
                                                   scope=scope.to_str(), artifact_ref=ref, operator="add"))
        await self.publish(HookPoint.SESSION_END.value, trajectory, "lesson", reward=reward, entry_id=entry_id)
