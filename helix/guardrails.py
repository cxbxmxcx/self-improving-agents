"""Guardrail wrapper: turn a CODE artifact into a typed Agent constructor input.

A Guardrail wraps a CODE artifact whose content is a Python source string
exposing one async function called `check`. The framework defines two
payload contracts (input and output) and one verdict contract. SPEC §16.2.

The Guardrail binds to a lifecycle phase:
  - phase="input"  -> registered on the agent's PRE_MODEL hook
  - phase="output" -> registered on the agent's PRE_OUTPUT hook

Verdict semantics:
  - allow=True, patched_payload=None  -> request proceeds unchanged
  - allow=True, patched_payload=<new> -> request proceeds with the patched payload
  - allow=False                       -> request refused; agent loop terminates

Refusal is a hard stop. The framework does not loop back, retry, or rewrite.
Critic-style "try again with feedback" patterns are agent-level compositions
(answer agent + critic agent), not guardrail behavior.
"""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ConfigDict

from helix.artifact import Artifact, ArtifactKind, ParentRef
from helix.hooks import HookPoint, Refusal


def _last_user_text(messages: Any) -> str:
    """The content of the most recent user message in a messages array.

    Input guardrails evaluate this (the text the model is about to see) rather
    than the original task, so they react to any prior PRE_MODEL patch (fix #8).
    """
    for m in reversed(messages or []):
        if isinstance(m, dict) and m.get("role") == "user":
            content = m.get("content")
            if isinstance(content, str):
                return content
    return ""


def _sources_from_trajectory(trajectory: Any) -> list[dict[str, Any]]:
    """The structured tool/retrieval results the agent used this run.

    Output guardrails inspect or redact these as `sources` (fix #8). Tool
    results that are lists of dicts (retrieval hits) are collected; other tool
    outputs are ignored.
    """
    sources: list[dict[str, Any]] = []
    for step in getattr(trajectory, "steps", []):
        kind = getattr(step, "kind", None)
        if getattr(kind, "value", kind) != "tool_result":
            continue
        result = (getattr(step, "payload", {}) or {}).get("result")
        if isinstance(result, list):
            sources.extend(r for r in result if isinstance(r, dict))
    return sources


# ---------------------------------------------------------------------------
# Pydantic payload contracts. SPEC §16.2.1.
# ---------------------------------------------------------------------------


class InputGuardrailPayload(BaseModel):
    """The payload an input guardrail sees before the agent's first model call.

    `question` is the user's input. `context` is an open dict for whatever
    request-scoped metadata the caller wants to expose (user id, session id,
    tenant scope, traceparent, etc.).
    """
    model_config = ConfigDict(extra="forbid")

    question: str
    context: dict[str, Any] = {}


class OutputGuardrailPayload(BaseModel):
    """The payload an output guardrail sees before the final response goes out."""
    model_config = ConfigDict(extra="forbid")

    answer: str
    sources: list[dict[str, Any]] = []
    trajectory_metadata: dict[str, Any] = {}


class GuardrailVerdict(BaseModel):
    """The contract a guardrail's `check` function must return.

    `allow=False` is a hard refusal. `patched_payload` lets the guardrail
    rewrite the payload (e.g. PII-scrub an input, redact citations from an
    output). When `patched_payload` is None the original payload is used.
    """
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    allow: bool
    reason: str = ""
    patched_payload: InputGuardrailPayload | OutputGuardrailPayload | None = None


class GuardrailFailure(Exception):
    """Raised when a guardrail's compiled code raises and fail_open is False."""

    def __init__(self, artifact_ref: ParentRef, inner: Exception) -> None:
        self.artifact_ref = artifact_ref
        self.inner = inner
        super().__init__(f"guardrail {artifact_ref} crashed: {type(inner).__name__}: {inner}")


# ---------------------------------------------------------------------------
# Code-artifact compilation
# ---------------------------------------------------------------------------


CheckFn = Callable[[InputGuardrailPayload | OutputGuardrailPayload], Awaitable[GuardrailVerdict]]


def compile_code_artifact(
    artifact: Artifact,
    expected_callable: str = "check",
) -> CheckFn:
    """Exec the source string in `artifact.content` and return the named callable.

    The artifact must be `ArtifactKind.CODE` and its content must be a string.
    The function is executed in a fresh namespace seeded with the guardrail
    payload / verdict types so guardrail code can `from helix.guardrails import
    InputGuardrailPayload, ...` if it wants typed handling, or use them by
    name without import.
    """
    if artifact.kind != ArtifactKind.CODE:
        raise ValueError(
            f"compile_code_artifact: expected ArtifactKind.CODE, got {artifact.kind}"
        )
    if not isinstance(artifact.content, str):
        raise ValueError(
            f"compile_code_artifact: artifact content must be a string source, "
            f"got {type(artifact.content).__name__}"
        )

    namespace: dict[str, Any] = {
        "InputGuardrailPayload": InputGuardrailPayload,
        "OutputGuardrailPayload": OutputGuardrailPayload,
        "GuardrailVerdict": GuardrailVerdict,
    }
    exec(compile(artifact.content, f"<artifact:{artifact.id}:v{artifact.version}>", "exec"),
         namespace)
    fn = namespace.get(expected_callable)
    if fn is None:
        raise ValueError(
            f"compile_code_artifact: artifact {artifact.id}:v{artifact.version} "
            f"does not define `{expected_callable}`"
        )
    if not inspect.iscoroutinefunction(fn):
        raise ValueError(
            f"compile_code_artifact: `{expected_callable}` in artifact "
            f"{artifact.id}:v{artifact.version} must be async"
        )
    return fn  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Guardrail wrapper
# ---------------------------------------------------------------------------


@dataclass
class Guardrail:
    """A CODE artifact + a lifecycle phase. SPEC §16.2.1.

    The agent compiles the artifact at construction, validates the function
    signature, and registers a hook handler on the appropriate point. At
    runtime each invocation records the artifact ref in
    `Trajectory.artifacts_used` for lineage.
    """

    artifact: Artifact
    phase: str = "input"  # "input" | "output"
    fail_open: bool = False
    _check: CheckFn | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        if self.phase not in ("input", "output"):
            raise ValueError(f"Guardrail phase must be 'input' or 'output', got {self.phase!r}")
        # Compile eagerly so misbehaving artifacts fail fast at construction.
        self._check = compile_code_artifact(self.artifact)

    @property
    def hook_point(self) -> HookPoint:
        return HookPoint.PRE_MODEL if self.phase == "input" else HookPoint.PRE_OUTPUT

    async def evaluate(
        self,
        payload: InputGuardrailPayload | OutputGuardrailPayload,
    ) -> GuardrailVerdict:
        """Invoke the compiled check. Wraps exceptions per `fail_open`."""
        if self._check is None:
            # Defensive: __post_init__ should always have compiled the artifact.
            self._check = compile_code_artifact(self.artifact)
        try:
            verdict = await self._check(payload)
        except Exception as exc:  # noqa: BLE001
            if self.fail_open:
                return GuardrailVerdict(
                    allow=True,
                    reason=f"guardrail_error_open: {type(exc).__name__}: {exc}",
                )
            raise GuardrailFailure(self.artifact.ref, exc) from exc
        return verdict

    def make_hook_handler(self) -> Callable[..., Awaitable[Any]]:
        """Return an async hook handler the Agent registers on the appropriate point.

        The handler reads the canonical PRE_MODEL / PRE_OUTPUT kwargs (which
        the agent loop always supplies) and constructs the typed payload
        before calling the guardrail's `check`. This keeps the agent's hook
        signatures stable for other hook handlers; only guardrails care about
        the typed payload contract.

        The handler returns one of:
          - None: request proceeds unchanged.
          - Refusal(...): the agent loop short-circuits and terminates.
          - ("patched", payload): the agent uses the patched payload for the
            rest of the turn.
        """
        guardrail = self

        if self.phase == "input":
            async def _input_handler(
                messages=None,
                trajectory=None,
                **_: Any,
            ) -> Any:
                if trajectory is None:
                    return None
                if hasattr(trajectory, "record_artifact"):
                    step_index = len(getattr(trajectory, "steps", []))
                    trajectory.record_artifact(step_index, guardrail.artifact.ref)
                payload = InputGuardrailPayload(
                    question=_last_user_text(messages) or (getattr(trajectory, "task", "") or ""),
                    context=dict(getattr(trajectory, "metadata", {}) or {}),
                )
                verdict = await guardrail.evaluate(payload)
                if not verdict.allow:
                    return Refusal(
                        reason=verdict.reason,
                        source=f"guardrail:{guardrail.artifact.id}:v{guardrail.artifact.version}",
                    )
                if verdict.patched_payload is not None:
                    return ("patched", verdict.patched_payload)
                return None

            return _input_handler

        async def _output_handler(
            response=None,
            trajectory=None,
            **_: Any,
        ) -> Any:
            if trajectory is None:
                return None
            if hasattr(trajectory, "record_artifact"):
                step_index = len(getattr(trajectory, "steps", []))
                trajectory.record_artifact(step_index, guardrail.artifact.ref)
            payload = OutputGuardrailPayload(
                answer=response if isinstance(response, str) else str(response or ""),
                sources=_sources_from_trajectory(trajectory),
                trajectory_metadata=dict(getattr(trajectory, "metadata", {}) or {}),
            )
            verdict = await guardrail.evaluate(payload)
            if not verdict.allow:
                return Refusal(
                    reason=verdict.reason,
                    source=f"guardrail:{guardrail.artifact.id}:v{guardrail.artifact.version}",
                )
            if verdict.patched_payload is not None:
                return ("patched", verdict.patched_payload)
            return None

        return _output_handler
