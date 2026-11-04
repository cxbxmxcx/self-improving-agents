"""What the scaffold's wrappers share: the run record, the decision call,
the System 2 call, and the HookFired event.

Every wrapper keeps its per-run state in `trajectory.metadata["metacognition"]`,
never on itself, so one agent can serve concurrent runs and the record is
exactly what the evaluations and the Observatory read afterwards.

The record keeps each state the decision model was asked about. With those,
a recorded run can be decided again under a different configuration, which
is how chapter 6 scores a routing variant without running the agent.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any

from helix._async import resolve
from helix.deciders.base import Answer, DecisionUsage, ask_measured
from helix.llm_call import acompletion
from helix.observability.events import HookFired
from helix.scaffold.config import apply_rules
from helix.trajectory import StepKind, Trajectory

if TYPE_CHECKING:
    from helix.agent import Agent
    from helix.scaffold.scaffold import MetacognitiveScaffold

META_KEY = "metacognition"
PINNABLE = {"route": frozenset({"fast", "deliberate", "ask"}), "tool": frozenset({"confirm", "allow"})}
_PINNED: ContextVar[dict[str, str] | None] = ContextVar("helix_scaffold_pinned", default=None)


@contextmanager
def pinned(**verdicts: str) -> Iterator[None]:
    """Fix the route or the confirmation verdict for the runs inside the block.

    Evaluation runs each request once down every path; a pinned wrapper takes
    the given path without asking the decision model, but still records the
    state it would have asked about. Production runs never enter this block.
    """
    for hook, verdict in verdicts.items():
        if verdict not in PINNABLE.get(hook, ()):
            raise ValueError(f"cannot pin {hook}={verdict!r}; pinnable: {dict(PINNABLE)}")
    token = _PINNED.set({**(_PINNED.get() or {}), **verdicts})
    try:
        yield
    finally:
        _PINNED.reset(token)


def pinned_verdict(hook: str) -> str | None:
    return (_PINNED.get() or {}).get(hook)


def record(trajectory: Trajectory) -> dict[str, Any]:
    """The run's metacognition record, created on first use."""
    meta = trajectory.metadata.get(META_KEY)
    if meta is None:
        meta = {"decisions": [], "decider_calls": [], "llm_calls": [], "tools": [], "states": [], "errors": []}
        trajectory.metadata[META_KEY] = meta
    return meta


def note_state(trajectory: Trajectory, hook: str, state: Any, **extra: Any) -> None:
    """Keep the state a hook decided on, so the decision can be replayed."""
    record(trajectory)["states"].append({"hook": hook, "turn": model_calls(trajectory), **extra, "state": state})


def decider_usage(trajectory: Trajectory) -> DecisionUsage:
    """What the decision model cost this run, summed from its calls."""
    total = DecisionUsage()
    for call in record(trajectory)["decider_calls"]:
        total.add(DecisionUsage(calls=call.get("calls", 1), input_tokens=call.get("input_tokens", 0),
                                dollars=call.get("dollars", 0.0), seconds=call.get("seconds", 0.0),
                                version=call.get("version")))
    return total


def note_artifact(trajectory: Trajectory, ref: tuple[str, int]) -> None:
    """Record an artifact read at the coming step, once."""
    index = len(trajectory.steps)
    if ref not in trajectory.artifacts_used.get(index, []):
        trajectory.record_artifact(index, ref)


def model_calls(trajectory: Trajectory) -> int:
    return sum(1 for s in trajectory.steps if s.kind == StepKind.MODEL_CALL)


def actions_of(trajectory: Trajectory, limit: int = 300) -> list[dict[str, Any]]:
    """The tool calls so far with their results, clipped for a decision state."""
    actions: list[dict[str, Any]] = []
    pending: dict[str, Any] | None = None
    for step in trajectory.steps:
        if step.kind == StepKind.TOOL_CALL:
            pending = {"tool": step.payload.get("name"), "arguments": step.payload.get("arguments", {})}
        elif step.kind == StepKind.TOOL_RESULT and pending is not None:
            result = json.dumps(step.payload.get("result"), default=str, ensure_ascii=False)
            pending["result"] = result if len(result) <= limit else result[:limit] + "..."
            actions.append(pending)
            pending = None
    return actions


def request_text(trajectory: Trajectory) -> str:
    """The request every decision state carries.

    Input guardrails patch the model's copy only after all PRE_MODEL hooks
    have fired, so this is the user's original text. An application that must
    filter what leaves for the decision model does so in `state_fn`.
    """
    return trajectory.task


def system_text(messages: list[dict[str, Any]]) -> str:
    content = messages[0].get("content") if messages else ""
    if isinstance(content, list) and content and isinstance(content[0], dict):
        return str(content[0].get("text", ""))
    return str(content or "")


def append_to_system(messages: list[dict[str, Any]], block: str) -> None:
    """Append a section to the system message in place, keeping any cache marker."""
    if not messages or messages[0].get("role") != "system":
        return
    text = system_text(messages).rstrip() + "\n\n" + block.strip()
    content = messages[0].get("content")
    if isinstance(content, list) and content and isinstance(content[0], dict):
        content[0]["text"] = text
    else:
        messages[0]["content"] = text


class Wrapper:
    """Shared plumbing for the router, monitor, task state, and reflector."""

    name = "wrapper"

    def __init__(self, scaffold: MetacognitiveScaffold, agent: Agent) -> None:
        self.scaffold = scaffold
        self.agent = agent

    async def state(self, hook: str, state: dict[str, Any], trajectory: Trajectory) -> Any:
        """The decision state, enriched by the application when it supplies `state_fn`."""
        if self.scaffold.state_fn is None:
            return state
        return await resolve(self.scaffold.state_fn(hook, state, trajectory))

    async def decide(self, hook: str, state: Any, trajectory: Trajectory,
                     **extra: Any) -> tuple[dict[str, Answer], set[str], list[dict[str, Any]]]:
        """Ask this hook's questions in one call and apply its rules.

        Returns the answers, the actions that fired, and the fired rules. On
        a failed call the scaffold either raises or, with `on_error="safe"`,
        returns no answers, which fires only the safety rules.
        """
        scaffold = self.scaffold
        questions = scaffold.decision_config.questions_for(hook)
        rules = scaffold.decision_config.rules_for(hook)
        meta = record(trajectory)
        note_state(trajectory, hook, state, **extra)
        note_artifact(trajectory, scaffold.config.ref)
        if any(r.threshold_name for r in rules):
            note_artifact(trajectory, scaffold.thresholds.ref)
        answers: dict[str, Answer] = {}
        if questions:
            try:
                answers, call = await ask_measured(scaffold.decider, state, questions)
            except Exception as exc:
                if scaffold.on_error != "safe":
                    raise
                meta["errors"].append({"hook": hook, "where": "decider", "error": f"{type(exc).__name__}: {exc}"})
            else:
                meta["decider_calls"].append({"hook": hook, **call.to_dict()})
        actions, fired_rules = apply_rules(scaffold.decision_config, hook, answers, scaffold.threshold_values)
        turn = model_calls(trajectory)
        for name in questions:
            answer = answers.get(name)
            meta["decisions"].append({
                "hook": hook, "turn": turn, "question": name, **extra,
                **(answer.to_dict() if answer is not None else {"kind": None, "value": None}),
                "actions": [f["action"] for f in fired_rules if f["question"] == name],
            })
        return answers, actions, fired_rules

    async def system2(self, purpose: str, model: str, params: dict[str, Any], system: str,
                      user: str, trajectory: Trajectory) -> str:
        """One slow, deliberate model call outside the loop, recorded for the cost signal."""
        started = time.perf_counter()
        response = await acompletion(model=model, messages=[{"role": "system", "content": system},
                                                            {"role": "user", "content": user}], **params)
        usage = getattr(response, "usage", None)
        tokens = {k: getattr(usage, k) for k in ("prompt_tokens", "completion_tokens")
                  if isinstance(getattr(usage, k, None), int)}
        record(trajectory)["llm_calls"].append({"purpose": purpose, "model": model, "usage": tokens or None,
                                                "seconds": round(time.perf_counter() - started, 4)})
        return (response.choices[0].message.content or "").strip()

    async def publish(self, point: str, trajectory: Trajectory, verdict: str, **detail: Any) -> None:
        await self.agent.bus.publish(HookFired(point=point, trajectory_id=trajectory.id, wrapper=self.name,
                                               verdict=verdict, detail=detail))
