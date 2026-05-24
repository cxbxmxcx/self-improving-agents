"""The Hook system.

Hooks fire at the canonical points in the agent loop. They can read trajectory,
mutate messages (only at PRE_MODEL), short-circuit (refuse at PRE_MODEL or
PRE_OUTPUT, cancel a tool call at PRE_TOOL), mutate request/response payloads,
or emit side effects. SPEC §6.2.

The agent loop is fixed by Chapter 2. New capabilities are added as hooks. By
the end of the book HelixAgent has ~20 registered hooks. The loop is unchanged.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any


@dataclass(frozen=True)
class Refusal:
    """Sentinel returned by a hook to terminate the agent run immediately.

    Refusal is a hard stop. The agent loop produces a Refusal result with
    `reason` and `source` filled and does not call the model again. SPEC §6.2.

    Returned by guardrail hooks (input phase via PRE_MODEL, output phase via
    PRE_OUTPUT). Callers can construct it manually inside any hook that wants
    the same short-circuit semantics.
    """
    reason: str
    source: str = ""


class HookPoint(str, Enum):
    SESSION_START = "session_start"
    PRE_MODEL = "pre_model"
    POST_MODEL = "post_model"
    PRE_TOOL = "pre_tool"
    POST_TOOL = "post_tool"
    END_OF_TURN = "end_of_turn"
    PRE_OUTPUT = "pre_output"
    SESSION_END = "session_end"
    POST_ARTIFACT_MUTATION = "post_artifact_mutation"


HookFn = Callable[..., Awaitable[Any]]


class HookRegistry:
    """Stores hooks per point and fires them in registration order.

    Registration order is the v0 ordering policy. A priority-based scheme can
    be added later without changing the agent loop or hook signatures.
    """

    def __init__(self) -> None:
        self._hooks: dict[HookPoint, list[HookFn]] = {p: [] for p in HookPoint}

    def register(self, point: HookPoint, fn: HookFn) -> None:
        self._hooks[point].append(fn)

    async def fire(self, point: HookPoint, **kwargs: Any) -> list[Any]:
        """Fire all hooks at this point. Returns each hook's result in order.

        Short-circuit semantics (SPEC §6.2):
          - A hook returning the sentinel string "cancel" at PRE_TOOL signals
            the loop to skip that tool call.
          - A hook returning a Refusal at PRE_MODEL or PRE_OUTPUT signals the
            loop to terminate the run with that refusal.
          - A hook returning ("patched", <payload>) at PRE_MODEL or PRE_OUTPUT
            signals the loop to use the patched payload for the rest of the turn.

        Other return values are passed through for the caller to interpret.
        """
        results: list[Any] = []
        for fn in self._hooks[point]:
            results.append(await fn(**kwargs))
        return results
