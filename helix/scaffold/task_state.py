"""Task state: notice when the work is done and say so.

After each turn of tool calls the decision model scores progress. When the
finish rule fires, the next model call carries a note to reply now, with
tool calls switched off. The agent's turn and tool-call limits remain the
hard stop; this only ends a run that has already done its job.
"""

from __future__ import annotations

from typing import Any

from helix.hooks import HookPoint, RequestPatch
from helix.scaffold.common import Wrapper, actions_of, append_to_system, record, request_text
from helix.trajectory import Trajectory

FINISH_NOTE = (
    "## Finish now\n"
    "The task looks complete. Reply to the user with what was done; do not call more tools."
)


class TaskState(Wrapper):
    name = "task_state"

    async def end_of_turn(self, trajectory: Trajectory | None = None, **_: Any) -> None:
        if trajectory is None or not self.scaffold.decision_config.questions_for("progress"):
            return None
        meta = record(trajectory)
        if meta.get("finish"):
            return None
        state = {"request": request_text(trajectory), "actions": actions_of(trajectory)}
        answers, actions, _ = await self.decide("progress", await self.state("progress", state, trajectory),
                                                trajectory)
        positions = [a.position for a in answers.values() if a.kind == "score" and a.position is not None]
        if positions:
            meta["progress"] = max(positions)
        meta["finish"] = "finish" in actions
        await self.publish(HookPoint.END_OF_TURN.value, trajectory, "finish" if meta["finish"] else "continue",
                           progress=meta.get("progress"))
        return None

    async def pre_model(self, messages: list[dict[str, Any]] | None = None,
                        trajectory: Trajectory | None = None, **_: Any) -> RequestPatch | None:
        if trajectory is None:
            return None
        meta = record(trajectory)
        if not meta.get("finish"):
            return None
        if not meta.get("finish_noted"):
            append_to_system(messages or [], FINISH_NOTE)
            meta["finish_noted"] = True
        if not self.agent.tools:
            return None
        # Registered after the router, so this keeps the route's model and adds only the switch.
        return RequestPatch(params={"tool_choice": "none"}, source="task_state:finish")
