"""Replay model replies or record missing replies from live calls.

Each call key includes the model, messages, tools, and selected parameters.
Replay requires an existing reply for that key, while record mode makes a
live call only when the reply is absent. A changed request can therefore
require a new recording even when it describes a similar task.

Each saved reply includes its reported usage and model-call duration.
Replay restores those durations onto the trajectory so the output describes
the recorded calls. These values exclude local replay time and do not measure
complete service latency.
"""

from __future__ import annotations

import contextvars
import hashlib
import json
import time
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from helix.trajectory import StepKind, Trajectory

MODES = ("replay", "record")
# The agent's model calls, and the calls learning mechanisms make to write memory.
TARGETS = ("helix.agent.helix_acompletion", "helix.memory.learning_mechanisms.helix_acompletion",
           "helix.search.gepa.helix_acompletion")
_served: contextvars.ContextVar[list[float] | None] = contextvars.ContextVar("served_seconds", default=None)


def call_key(kwargs: dict[str, Any]) -> str:
    params = {k: kwargs[k] for k in ("reasoning_effort", "temperature") if k in kwargs}
    canonical = json.dumps({"model": kwargs.get("model"), "messages": kwargs.get("messages"),
                            "tools": kwargs.get("tools"), "params": params}, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _plain(response: Any) -> dict[str, Any]:
    """The parts of a live response the agent reads, as JSON."""
    from helix.agent import _usage_of

    msg = response.choices[0].message
    calls = [{"id": tc.id, "name": tc.function.name, "arguments": tc.function.arguments}
             for tc in (msg.tool_calls or [])]
    return {"content": msg.content or "", "tool_calls": calls, "usage": _usage_of(response) or {}}


def _response(row: dict[str, Any]) -> SimpleNamespace:
    calls = [SimpleNamespace(id=c["id"], type="function",
                             function=SimpleNamespace(name=c["name"], arguments=c["arguments"]))
             for c in row["tool_calls"]] or None
    usage = row.get("usage") or {}
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=row["content"], tool_calls=calls))],
        usage=SimpleNamespace(
            prompt_tokens=usage.get("prompt_tokens", 0), completion_tokens=usage.get("completion_tokens", 0),
            completion_tokens_details=SimpleNamespace(reasoning_tokens=usage.get("reasoning_tokens", 0)),
            prompt_tokens_details=SimpleNamespace(cached_tokens=usage.get("cached_tokens", 0))))


class Recording:
    """A JSONL file of model replies keyed by call."""

    def __init__(self, path: Path, *, mode: str = "replay", distinct_repeats: bool = False) -> None:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self.path = Path(path)
        self.mode = mode
        # A search that asks the same question several times wants several
        # answers; elsewhere a repeated call must replay the same reply.
        self.distinct_repeats = distinct_repeats
        self._seen: dict[str, int] = {}
        self.rows: dict[str, dict[str, Any]] = {}
        self.live_calls = 0
        self.live_rows: list[dict[str, Any]] = []
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    self.rows.setdefault(row["key"], row)

    async def __call__(self, **kwargs: Any) -> Any:
        key = call_key(kwargs)
        if self.distinct_repeats:
            n = self._seen.get(key, 0)
            self._seen[key] = n + 1
            key = key if n == 0 else f"{key}#{n}"
        row = self.rows.get(key)
        if row is None:
            if self.mode == "replay":
                raise KeyError(f"no recorded reply for this call in {self.path.name}; rerun with --record")
            from helix.llm_call import acompletion

            started = time.perf_counter()
            response = await acompletion(**kwargs)
            row = {"key": key, "model": kwargs.get("model"), "effort": kwargs.get("reasoning_effort"),
                   "seconds": round(time.perf_counter() - started, 4), **_plain(response)}
            self.rows[key] = row
            self.live_calls += 1
            self.live_rows.append(row)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row) + "\n")
        served = _served.get()
        if served is not None:
            served.append(float(row.get("seconds") or 0.0))
        return _response(row)

    def spent(self, prices: dict[str, tuple[float, ...]]) -> float:
        """Dollars the live calls of this session cost, from their recorded usage."""
        from helix.signals.cost import call_dollars

        return sum(call_dollars(row.get("model") or "", row.get("usage") or {}, prices) or 0.0
                   for row in self.live_rows)

    @contextmanager
    def active(self) -> Iterator[Recording]:
        with ExitStack() as stack:
            for target in TARGETS:
                stack.enter_context(patch(target, new=self.__call__))
            yield self


@contextmanager
def recorded_seconds() -> Iterator[list[float]]:
    """Collect the recorded seconds of the calls one run makes, in order.

    Each run sets its own list, so concurrent runs never mix their calls.
    """
    token = _served.set([])
    try:
        yield _served.get()
    finally:
        _served.reset(token)


def restore_seconds(trajectory: Trajectory, seconds: list[float]) -> None:
    """Give each model call in a run the seconds its recorded reply took live."""
    calls = [s for s in trajectory.steps if s.kind == StepKind.MODEL_CALL]
    for step, value in zip(calls, seconds):
        step.payload["seconds"] = value
