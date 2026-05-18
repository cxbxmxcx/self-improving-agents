"""The Hook system.

Hooks fire at the canonical points in the agent loop. They can read trajectory,
mutate messages (only at PRE_MODEL), short-circuit (cancel a tool call at
PRE_TOOL), or emit side effects. Spec §6.

The agent loop is fixed by Chapter 2. New capabilities are added as hooks. By
the end of the book HelixAgent has ~20 registered hooks. The loop is unchanged.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from enum import Enum
from typing import Any


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

        A hook returning the sentinel string "cancel" at PRE_TOOL signals the
        loop to skip that tool call. Other return values are passed through
        unchanged for the caller to interpret.
        """
        results: list[Any] = []
        for fn in self._hooks[point]:
            results.append(await fn(**kwargs))
        return results
