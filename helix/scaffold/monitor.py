"""The monitor: confirm consequential tool calls, and revise an incomplete reply.

Before a watched tool runs, the monitor asks what the call changes for
people other than the user; when a confirmation rule fires and the user has
not confirmed, the call is cancelled with a reason the model can relay.
Before the reply goes out, it asks whether the reply is complete and, when
the revise rule fires, makes one System 2 call to repair it.
"""

from __future__ import annotations

import json
from typing import Any

from helix.guardrails import OutputGuardrailPayload
from helix.hooks import HookPoint
from helix.scaffold.common import Wrapper, actions_of, note_state, pinned_verdict, record, request_text, resolve
from helix.trajectory import Trajectory

REVISION_PROMPT = (
    "You revise an assistant's draft reply before it reaches the user. The draft may miss part "
    "of the request. Rewrite it so it answers every question and covers every item the request "
    "asks for, using only facts from the request and the tool results. Return only the revised reply."
)


class Monitor(Wrapper):
    name = "monitor"

    async def pre_tool(self, tool_name: str = "", arguments: dict[str, Any] | None = None,
                       trajectory: Trajectory | None = None, **_: Any) -> Any:
        scaffold = self.scaffold
        if trajectory is None or tool_name not in scaffold.watch_tools:
            return None
        arguments = arguments or {}
        meta = record(trajectory)
        # A retried call gets the same verdict without asking again.
        for seen in meta["tools"]:
            if seen["tool"] == tool_name and seen["arguments"] == arguments:
                return self._result(seen)
        if scaffold.confirmed is not None and await resolve(scaffold.confirmed(trajectory, tool_name, arguments)):
            check = {"tool": tool_name, "arguments": arguments, "verdict": "confirmed"}
        else:
            state = {"request": request_text(trajectory), "tool": tool_name, "arguments": arguments}
            state = await self.state("tool", state, trajectory)
            pin = pinned_verdict("tool")
            if pin is not None:
                note_state(trajectory, "tool", state, tool=tool_name)
                check = {"tool": tool_name, "arguments": arguments, "verdict": pin, "pinned": True,
                         "reason": self._reason(tool_name, {}, [])}
            else:
                answers, actions, fired = await self.decide("tool", state, trajectory, tool=tool_name)
                check = {"tool": tool_name, "arguments": arguments,
                         "verdict": "confirm" if "confirm" in actions else "allow",
                         "reason": self._reason(tool_name, answers, fired)}
        meta["tools"].append(check)
        await self.publish(HookPoint.PRE_TOOL.value, trajectory, check["verdict"], tool=tool_name)
        return self._result(check)

    @staticmethod
    def _result(check: dict[str, Any]) -> Any:
        return ("cancel", check["reason"]) if check["verdict"] == "confirm" else None

    def _reason(self, tool_name: str, answers: dict[str, Any], fired: list[dict[str, Any]]) -> str:
        """Tell the model what to confirm, in the words of the chosen option."""
        questions = self.scaffold.decision_config.questions
        details: list[str] = []
        for f in fired:
            if f["action"] != "confirm":
                continue
            spec, answer = questions[f["question"]], answers.get(f["question"])
            if answer is None:
                details.append("the call could not be checked")
            elif spec.type == "choice":
                details.append((spec.criteria or {}).get(str(answer.value)) or str(answer.value))
            else:
                details.append(spec.instructions)
        suffix = f": {'; '.join(dict.fromkeys(details))}" if details else ""
        return f"needs the user's confirmation before {tool_name}{suffix}"

    async def pre_output(self, response: Any = None, trajectory: Trajectory | None = None, **_: Any) -> Any:
        scaffold = self.scaffold
        if trajectory is None:
            return None
        meta = record(trajectory)
        # A question back to the user is not a reply to complete: neither the
        # ask route's question nor the request to confirm a call it held.
        held = any(t["verdict"] == "confirm" for t in meta["tools"])
        if meta.get("route") == "ask" or held or meta.get("revised") or not scaffold.decision_config.questions_for("output"):
            return None
        reply = response if isinstance(response, str) else str(response or "")
        state = {"request": request_text(trajectory), "reply": reply, "actions": actions_of(trajectory)}
        state = await self.state("output", state, trajectory)
        _, actions, _ = await self.decide("output", state, trajectory)
        if "revise" not in actions:
            meta["revised"] = False
            await self.publish(HookPoint.PRE_OUTPUT.value, trajectory, "keep")
            return None
        model, params = scaffold.system2(self.agent, scaffold.revision_model)
        user = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False, default=str)
        try:
            revised = await self.system2("revise", model, params, REVISION_PROMPT, user, trajectory)
        except Exception as exc:
            if scaffold.on_error != "safe":
                raise
            meta["errors"].append({"hook": "output", "where": "revision", "error": f"{type(exc).__name__}: {exc}"})
            revised = ""
        meta["revised"] = bool(revised)
        await self.publish(HookPoint.PRE_OUTPUT.value, trajectory, "revise" if revised else "keep")
        if not revised:
            return None
        return ("patched", OutputGuardrailPayload(answer=revised))
