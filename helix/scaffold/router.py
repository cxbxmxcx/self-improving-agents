"""The router: choose the fast path, the deliberate path, or a question back.

At the first PRE_MODEL call the router asks the decision model the routing
questions and keeps the route for the whole run. Asking beats deliberating,
and deliberating beats the fast path, because a question to the user is the
cheapest way to avoid acting on a wrong reading. On every later call it
returns the same request patch, so the run stays on its path.
"""

from __future__ import annotations

from typing import Any

from helix.hooks import HookPoint, RequestPatch
from helix.scaffold.common import (
    Wrapper,
    append_to_system,
    note_artifact,
    note_state,
    pinned_verdict,
    record,
    request_text,
    system_text,
)
from helix.trajectory import Trajectory

PLANNER_HEADING = "## Plan before you act"
ASK_NOTE = (
    "## Ask before you act\n"
    "The request may be ambiguous: {reasons}\n"
    "{context}"
    "Do not call tools. Ask the user one short question that names the options."
)
# Highest priority first.
ROUTE_ORDER = ("ask", "deliberate")


def choose_route(actions: set[str]) -> str:
    """The route the fired actions call for: asking beats deliberating, and
    deliberating beats the fast path."""
    return next((r for r in ROUTE_ORDER if r in actions), "fast")


def ask_context(state: Any) -> str:
    """The lists the decision model saw besides the request, such as the
    week's events, so a question asked without tools can name the options."""
    if not isinstance(state, dict):
        return ""
    lines: list[str] = []
    for key, value in state.items():
        items = [v for v in value if isinstance(v, str)] if isinstance(value, list) else []
        if key not in ("request", "memory") and items:
            lines += [f"{key.capitalize()}:", *(f"- {item}" for item in items)]
    return "\n".join(lines) + "\n" if lines else ""


class Router(Wrapper):
    name = "router"

    async def pre_model(self, messages: list[dict[str, Any]] | None = None,
                        trajectory: Trajectory | None = None, **_: Any) -> RequestPatch | None:
        if trajectory is None:
            return None
        meta = record(trajectory)
        if "route" not in meta:
            await self._choose(messages or [], trajectory, meta)
        return self.patch(meta["route"])

    async def _choose(self, messages: list[dict[str, Any]], trajectory: Trajectory, meta: dict[str, Any]) -> None:
        scaffold = self.scaffold
        state: dict[str, Any] = {"request": request_text(trajectory)}
        recalled = self._recalled(messages)
        if recalled:
            state["memory"] = recalled
        state = await self.state("route", state, trajectory)
        pin = pinned_verdict("route")
        if pin is not None:
            note_state(trajectory, "route", state)
            route, fired = pin, []
            meta["pinned"] = {**meta.get("pinned", {}), "route": pin}
        else:
            _, actions, fired = await self.decide("route", state, trajectory)
            route = choose_route(actions)
        meta["route"] = route
        if route == "deliberate" and scaffold.planner is not None:
            append_to_system(messages, f"{PLANNER_HEADING}\n{scaffold.planner.content}")
            note_artifact(trajectory, scaffold.planner.ref)
        elif route == "ask":
            questions = scaffold.decision_config.questions
            asked = [f["question"] for f in fired if f["action"] == "ask"] or [
                r.question for r in scaffold.decision_config.rules_for("route") if r.action == "ask"]
            reasons = "; ".join(dict.fromkeys(questions[q].instructions for q in asked)) or "it can be read more than one way."
            append_to_system(messages, ASK_NOTE.format(reasons=reasons, context=ask_context(state)))
        await self.publish(HookPoint.PRE_MODEL.value, trajectory, route,
                           fired=[f["question"] for f in fired], pinned=pin is not None)

    def _recalled(self, messages: list[dict[str, Any]]) -> str:
        """What memory added to the system message this run, so a recalled
        lesson can change the route."""
        base = str(self.agent.system_prompt.content).rstrip()
        text = system_text(messages)
        return text[len(base):].strip() if text.startswith(base) else ""

    def patch(self, route: str) -> RequestPatch:
        scaffold = self.scaffold
        if route == "deliberate":
            return RequestPatch(model=scaffold.deliberate_model, params=dict(scaffold.deliberate_params),
                                source="router:deliberate")
        params = dict(scaffold.fast_params)
        if route == "ask" and self.agent.tools:
            params["tool_choice"] = "none"
        return RequestPatch(model=scaffold.fast_model, params=params, source=f"router:{route}")
