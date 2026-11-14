"""Record and replay model calls, so every run, test, and search can be reproduced.

Every model call helix makes goes through `helix.llm_call.acompletion`. This
module installs one router in its place, and at every module that imported it
under its own name, which sends each call to the transport active in the
caller's context, or to the real call when none is active. A `Recording` is
such a transport:

- replay serves saved replies and fails on a call it has no reply for;
- record serves saved replies and calls the model only for the ones it lacks;
- live calls the model every time and saves each reply.

Keys and rows match the book's chapter 6 recordings, so those files replay
unchanged. The active transport is a context variable: concurrent runs can
use different recordings, and tasks started inside a run inherit its own.

A recorded reply keeps its usage and the seconds the live call took. Replay
cannot reproduce latency by waiting, so `served_seconds` collects those
values in call order and `restore_seconds` writes them back onto a run's
trajectory afterwards.
"""

from __future__ import annotations

import contextvars
import hashlib
import json
import sys
import time
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

MODES = ("replay", "record", "live")
# Call parameters that change a reply and so belong in its key. The book keys
# on the first two; the rest enter a key only when a call sets them, and no
# chapter 6 call does, so the book's recordings keep their keys.
KEY_PARAMS = ("reasoning_effort", "temperature", "top_p", "max_tokens", "max_completion_tokens", "tool_choice",
              "parallel_tool_calls", "response_format", "seed", "verbosity", "stop", "thinking")

Transport = Callable[..., Awaitable[Any]]

_transport: contextvars.ContextVar[Transport | None] = contextvars.ContextVar("metacog_transport", default=None)
_served: contextvars.ContextVar[list[float] | None] = contextvars.ContextVar("metacog_served", default=None)
# The real helix.llm_call.acompletion, kept once the router is installed.
_original: Transport | None = None


class MissingReply(KeyError):
    """Replay met a call the recording has no reply for."""


def call_key(kwargs: dict[str, Any]) -> str:
    """The key of one call: model, messages, tools, and the parameters that change a reply."""
    params = {k: kwargs[k] for k in KEY_PARAMS if k in kwargs}
    canonical = json.dumps({"model": kwargs.get("model"), "messages": kwargs.get("messages"),
                            "tools": kwargs.get("tools"), "params": params}, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


async def _dispatch(**kwargs: Any) -> Any:
    """The real call: Claude models through Anthropic's SDK (helix.metacognition.claude), every other model through LiteLLM."""
    from helix.metacognition import claude
    if claude.is_claude(kwargs.get("model")):
        return await claude.acompletion(**kwargs)
    return await _original(**kwargs)  # type: ignore[misc]


async def _routed(**kwargs: Any) -> Any:
    transport = _transport.get()
    if transport is None:
        return await _dispatch(**kwargs)
    return await transport(**kwargs)


def install() -> None:
    """Route helix's model calls through the active transport. Safe to call repeatedly.

    Without an active transport the router calls the real function, so
    installing changes nothing until a recording is activated.
    """
    global _original
    if _original is not None:
        return
    from helix import llm_call

    original = llm_call.acompletion
    _original = original
    # Modules imported later bind the router through helix.llm_call; modules
    # already imported hold the real function under their own names. This
    # module is skipped: its own `_original` must keep the real function.
    llm_call.acompletion = _routed
    this = sys.modules[__name__]
    for module in list(sys.modules.values()):
        if module is this:
            continue
        try:
            namespace = vars(module)
        except TypeError:
            continue
        for attr, value in list(namespace.items()):
            if value is original:
                setattr(module, attr, _routed)


async def _call_model(**kwargs: Any) -> Any:
    install()
    return await _dispatch(**kwargs)


def _plain(response: Any) -> dict[str, Any]:
    """The parts of a live response the agent loop reads, as JSON, and why the reply ended."""
    from helix.agent import _usage_of

    choice = response.choices[0]
    msg = choice.message
    calls = [{"id": tc.id, "name": tc.function.name, "arguments": tc.function.arguments}
             for tc in (msg.tool_calls or [])]
    row = {"content": msg.content or "", "tool_calls": calls, "usage": _usage_of(response) or {}}
    finish = getattr(choice, "finish_reason", None)
    if isinstance(finish, str):  # a truncated or refused reply shows in the recording
        row["finish_reason"] = finish
    return row


def _response(row: dict[str, Any]) -> SimpleNamespace:
    """A response object rebuilt from a saved row, shaped like LiteLLM's."""
    calls = [SimpleNamespace(id=c["id"], type="function",
                             function=SimpleNamespace(name=c["name"], arguments=c["arguments"]))
             for c in row["tool_calls"]] or None
    usage = row.get("usage") or {}
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=row["content"], tool_calls=calls))],
        usage=SimpleNamespace(
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            completion_tokens_details=SimpleNamespace(reasoning_tokens=usage.get("reasoning_tokens", 0)),
            prompt_tokens_details=SimpleNamespace(cached_tokens=usage.get("cached_tokens", 0)),
        ),
    )


class Recording:
    """A JSONL file of model replies keyed by call.

    `distinct_repeats` gives a repeated identical call its own reply, which a
    search that asks the same question several times needs; elsewhere a
    repeated call replays the same reply.
    """

    def __init__(self, path: str | Path, *, mode: str = "replay", distinct_repeats: bool = False) -> None:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, not {mode!r}")
        self.path = Path(path)
        self.mode = mode
        self.distinct_repeats = distinct_repeats
        self.rows: dict[str, dict[str, Any]] = {}
        self.live_calls = 0
        self.live_rows: list[dict[str, Any]] = []
        self.used_rows: dict[str, dict[str, Any]] = {}  # every distinct reply served, live or replayed
        self._seen: dict[str, int] = {}
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    self.rows.setdefault(row["key"], row)

    async def __call__(self, **kwargs: Any) -> SimpleNamespace:
        key = call_key(kwargs)
        if self.distinct_repeats:
            n = self._seen.get(key, 0)
            self._seen[key] = n + 1
            key = key if n == 0 else f"{key}#{n}"
        row = None if self.mode == "live" else self.rows.get(key)
        if row is None:
            if self.mode == "replay":
                raise MissingReply(f"no recorded reply for this call in {self.path.name}; "
                                   "record it with mode='record' (a paid call)")
            started = time.perf_counter()
            response = await _call_model(**kwargs)
            row = {"key": key, "model": kwargs.get("model"), "effort": kwargs.get("reasoning_effort"),
                   "seconds": round(time.perf_counter() - started, 4), **_plain(response)}
            self.rows.setdefault(key, row)
            self.live_calls += 1
            self.live_rows.append(row)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8", newline="\n") as fh:  # LF on every platform
                fh.write(json.dumps(row) + "\n")
        self.used_rows.setdefault(key, row)
        served = _served.get()
        if served is not None:
            served.append(float(row.get("seconds") or 0.0))
        return _response(row)

    @contextmanager
    def active(self) -> Iterator[Recording]:
        """Serve this context's model calls, and those of every task it starts, from this recording."""
        install()
        token = _transport.set(self)
        try:
            yield self
        finally:
            _transport.reset(token)

    def used(self, model: str, prices: dict[str, tuple[float, ...]] | None = None) -> tuple[int, float]:
        """Distinct calls to `model` this session served, live or replayed, and what they cost."""
        from helix.signals.cost import call_dollars

        rows = [row for row in self.used_rows.values() if row.get("model") == model]
        return len(rows), sum(call_dollars(model, row.get("usage") or {}, prices) or 0.0 for row in rows)

    def spent(self, prices: dict[str, tuple[float, ...]] | None = None) -> float:
        """Dollars this session's live calls cost, from their recorded usage."""
        from helix.signals.cost import call_dollars

        return sum(call_dollars(row.get("model") or "", row.get("usage") or {}, prices) or 0.0
                   for row in self.live_rows)


class BudgetSpent(BaseException):
    """New spending reached the cap. A BaseException, so no learner's `except Exception` swallows it."""


class Spend:
    """One cap, in dollars of new model calls, across several recordings."""

    def __init__(self, limit: float, prices: dict[str, tuple[float, ...]] | None = None) -> None:
        if prices is None:
            from helix.metacognition.pricing import PRICES as prices
        self.limit = limit
        self.prices = prices
        self.recordings: list[Recording] = []

    @property
    def total(self) -> float:
        return sum(recording.spent(self.prices) for recording in self.recordings)

    def largest(self, model: str | None, effort: str | None) -> float:
        """The most one new call at this model and effort has cost: what the next one might."""
        from helix.signals.cost import call_dollars

        rows = [row for recording in self.recordings for row in recording.live_rows
                if (row.get("model"), row.get("effort")) == (model, effort)]
        return max((call_dollars(model or "", row.get("usage") or {}, self.prices) or 0.0 for row in rows), default=0.0)


class CappedRecording(Recording):
    """A recording that stops before a new call, across every recording under the same cap, could pass the budget."""

    def __init__(self, path: str | Path, *, mode: str = "replay", spend: Spend, distinct_repeats: bool = False) -> None:
        super().__init__(path, mode=mode, distinct_repeats=distinct_repeats)
        self.spend = spend
        spend.recordings.append(self)

    async def __call__(self, **kwargs: Any) -> SimpleNamespace:
        spent = self.spend.total
        if spent + self.spend.largest(kwargs.get("model"), kwargs.get("reasoning_effort")) > self.spend.limit:
            raise BudgetSpent(f"${spent:.2f} of new calls, and the next could pass the ${self.spend.limit:.2f} budget")
        return await super().__call__(**kwargs)


@contextmanager
def served_seconds() -> Iterator[list[float]]:
    """Collect, in order, the recorded seconds of the calls made in this context."""
    token = _served.set([])
    try:
        yield _served.get()  # type: ignore[misc]
    finally:
        _served.reset(token)


def restore_seconds(trajectory: Any, seconds: list[float]) -> None:
    """Give each model call in a run the seconds its recorded reply took live."""
    from helix.trajectory import StepKind

    calls = [s for s in trajectory.steps if s.kind == StepKind.MODEL_CALL]
    # Unequal lengths are normal: with no recording active, nothing was served.
    for step, value in zip(calls, seconds, strict=False):
        step.payload["seconds"] = value
