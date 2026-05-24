"""OfflineImprover: the long-running per-artifact improvement loop.

An OfflineImprover is a registered, schedulable owner of one artifact's
improvement plan. It has an Agent (the definition under improvement), a
Signal, a Search, an Archive, an EvalSource, and a Policy. It runs rounds
asynchronously per its Schedule. SPEC §15.

Lifecycle:
    improver = OfflineImprover(agent=agent, ...)
    await improver.start()               # spawns background asyncio.Task
    # agent definition stays externally owned; improver clones it to test
    status = improver.status             # readable any time
    await improver.trigger_round()       # force a round now (manual or in-between)
    await improver.wait_until_idle()     # block until no in-flight rounds
    await improver.stop()                # graceful shutdown

The improver does not own the Agent. It receives the Agent by reference and
calls `agent.with_artifacts({target_id: candidate})` to test candidates. The
Agent's user reads the current-best from the archive on each request. This
is what makes the in-process / out-of-process split survivable.

For event-driven online improvement, see helix.improvement.online_improver.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any

from helix.agent import Agent
from helix.archive.archive import Archive
from helix.artifact import Artifact
from helix.eval.source import EvalSource
from helix.improvement.policy import ImproverMode, ImproverPolicy, Schedule
from helix.improvement.promotion import (
    ensure_default_handler_registered,
    register_improver_archive,
    unregister_improver_archive,
)
from helix.improvement.round import RoundResult, run_improvement_round
from helix.observability.bus import EventBus, get_bus
from helix.search.base import Search
from helix.signal import Signal

_log = logging.getLogger("helix.improver")


@dataclass
class ImproverStatus:
    improver_id: str
    target_artifact_id: str
    running: bool
    rounds_completed: int
    rounds_in_flight: int
    last_round_at: float | None
    last_round_result: RoundResult | None
    last_error: str | None
    schedule: str


class OfflineImprover:
    """A per-artifact offline improvement loop. SPEC §15.

    Constructor surface: the Agent under improvement plus the target artifact,
    a signal/search/archive/eval, and a policy. The improver clones the agent
    via `agent.with_artifacts({target_id: candidate})` to test each candidate.

    For event-driven online improvement (no labeled eval set, auto-promote on
    rolling spot-check), see helix.improvement.online_improver.OnlineImprover.
    """

    def __init__(
        self,
        *,
        agent,                          # Agent: the definition under improvement
        target_artifact_id: str,
        signal: Signal,
        search: Search,
        archive: Archive,
        eval_source: EvalSource,
        policy: ImproverPolicy | None = None,
        seed_fallback: Artifact | None = None,
        improver_id: str | None = None,
        bus: EventBus | None = None,
    ) -> None:
        self.agent = agent
        self.target_artifact_id = target_artifact_id
        self.signal = signal
        self.search = search
        self.archive = archive
        self.eval_source = eval_source
        self.policy = policy or ImproverPolicy()
        self.seed_fallback = seed_fallback
        self.improver_id = improver_id or f"imp-{uuid.uuid4().hex[:8]}"
        self.bus = bus or get_bus()

        # OfflineImprover places no layer restrictions on its target; offline
        # is the safe mode for L3/L4 artifacts because promotion is gated on
        # human review. OnlineImprover carries the L1/L2-only restriction.

        # Wire this Improver into the default promotion handler so the
        # bus-level handler can look up the archive when it fires.
        # DESIGN_NOTES.md section 10.
        register_improver_archive(self.improver_id, self.archive)
        ensure_default_handler_registered(self.bus)

        self._task: asyncio.Task | None = None
        self._stop_event = asyncio.Event()
        self._paused = False  # toggled by BackgroundImproverRunner; suppresses scheduled rounds
        self._in_flight = 0
        self._rounds_completed = 0
        self._last_round_at: float | None = None
        self._last_round_result: RoundResult | None = None
        self._last_error: str | None = None

    # ---------------- lifecycle ----------------

    async def start(self) -> None:
        """Spawn the background loop. Idempotent: no-op if already running."""
        if self._task is not None and not self._task.done():
            return
        self._stop_event.clear()
        self._task = asyncio.create_task(self._run_forever(), name=f"improver:{self.improver_id}")

    async def stop(self) -> None:
        """Signal the loop to stop after the current round; wait for it to exit."""
        if self._task is None:
            return
        self._stop_event.set()
        try:
            await asyncio.wait_for(self._task, timeout=30.0)
        except asyncio.TimeoutError:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        self._task = None

    async def wait_until_idle(self, timeout: float | None = None) -> None:
        """Block until no rounds are in flight."""
        async def poll():
            while self._in_flight > 0:
                await asyncio.sleep(0.05)
        if timeout is None:
            await poll()
        else:
            await asyncio.wait_for(poll(), timeout=timeout)

    async def trigger_round(self) -> RoundResult:
        """Run a single round right now, regardless of schedule.

        Useful for tests, notebooks, and chapter scripts that want one
        deterministic round. Blocks until the round completes.
        """
        return await self._run_one_round()

    # ---------------- status ----------------

    @property
    def status(self) -> ImproverStatus:
        return ImproverStatus(
            improver_id=self.improver_id,
            target_artifact_id=self.target_artifact_id,
            running=self._task is not None and not self._task.done(),
            rounds_completed=self._rounds_completed,
            rounds_in_flight=self._in_flight,
            last_round_at=self._last_round_at,
            last_round_result=self._last_round_result,
            last_error=self._last_error,
            schedule=self.policy.schedule.value,
        )

    # ---------------- internal ----------------

    async def _run_forever(self) -> None:
        try:
            while not self._stop_event.is_set():
                if self.policy.schedule == Schedule.MANUAL:
                    # Manual mode just keeps the task alive; trigger_round
                    # is the only way rounds fire.
                    await asyncio.sleep(0.5)
                    continue

                if self._paused:
                    # Paused by an external controller (BackgroundImproverRunner).
                    # In-flight rounds finish; new scheduled rounds are suppressed
                    # until resumed. trigger_round() still works.
                    await asyncio.sleep(0.5)
                    continue

                if self._in_flight >= self.policy.max_in_flight:
                    await asyncio.sleep(0.5)
                    continue

                try:
                    await self._run_one_round()
                except Exception as exc:  # noqa: BLE001
                    self._last_error = f"{type(exc).__name__}: {exc}"
                    _log.exception("improver %s round failed: %s", self.improver_id, exc)
                    # In continuous mode, back off briefly to avoid tight error loops.
                    await asyncio.sleep(5.0)

                if self.policy.schedule == Schedule.INTERVAL:
                    # Wait the interval (or until stop is signaled).
                    try:
                        await asyncio.wait_for(
                            self._stop_event.wait(),
                            timeout=self.policy.interval_sec,
                        )
                        # If we got here, stop was set during the wait.
                        return
                    except asyncio.TimeoutError:
                        continue
                # CONTINUOUS: loop again immediately.
        except asyncio.CancelledError:
            return

    async def _seed(self) -> Artifact:
        """Resolve which artifact to seed the round from."""
        best = await self.archive.best(k=1)
        if best:
            return best[0].artifact
        existing = await self.archive.by_id(self.target_artifact_id)
        if existing:
            return existing.artifact
        if self.seed_fallback is None:
            raise RuntimeError(
                f"archive empty and no seed_fallback provided for {self.target_artifact_id}"
            )
        # Store the fallback so future best() calls find it.
        from helix.search.base import Variant
        # We don't record a measurement for the fallback; see SPEC honesty
        # decision in spo_offline_loop.py history. The first round measures it.
        if hasattr(self.archive, "_store_artifact"):
            self.archive._store_artifact(self.seed_fallback)  # type: ignore[attr-defined]
            self.archive._conn.commit()  # type: ignore[attr-defined]
        return self.seed_fallback

    def _capped_build_agent(self, candidate: Artifact):
        """Clone the agent with `candidate` swapped in for the target artifact,
        applying the policy's per-question caps (max iterations, max tool calls).

        SPEC §15.2: the improver tests candidates by calling
        `agent.with_artifacts({target_id: candidate})`, then trims the clone
        to the policy's per-question budgets.
        """
        clone = self.agent.with_artifacts({self.target_artifact_id: candidate})
        clone.max_iterations = self.policy.max_iterations_per_question
        clone.max_tool_calls = self.policy.max_tool_calls_per_question
        return clone

    async def _run_one_round(self) -> RoundResult:
        if self.policy.quiesce_on_empty_eval:
            try:
                questions = await self.eval_source.get_questions(
                    n=self.policy.questions_per_round
                )
            except Exception:
                questions = []
            if not questions:
                # No traffic / no questions; idle gracefully.
                await asyncio.sleep(1.0)
                raise _NoQuestions()

        self._in_flight += 1
        t0 = time.time()
        try:
            seed = await self._seed()
            result = await run_improvement_round(
                round_index=self._rounds_completed + 1,
                seed_artifact=seed,
                agent_factory=lambda: self._capped_build_agent(seed),
                clone_agent_with_artifact=self._capped_build_agent,
                search=self.search,
                signal=self.signal,
                archive=self.archive,
                eval_source=self.eval_source,
                budget=self.policy.budget_per_round,
                improver_id=self.improver_id,
                promote_threshold_win_rate=self.policy.promote_threshold_win_rate,
                questions_per_round=self.policy.questions_per_round,
                max_concurrent_questions=self.policy.max_concurrent_questions,
                bus=self.bus,
                mode="offline",  # OfflineImprover is offline by class identity
                auto_promote=self.policy.auto_promote,
            )
            self._rounds_completed += 1
            self._last_round_at = t0
            self._last_round_result = result
            self._last_error = None

            # Optional feedback hook: search methods that care about round
            # outcomes (StrategyChain rotates on failure) can hook here.
            # Other Searches don't implement it and are not called.
            on_result = getattr(self.search, "on_round_result", None)
            if on_result is not None:
                try:
                    maybe_coro = on_result(result)
                    import inspect
                    if inspect.isawaitable(maybe_coro):
                        await maybe_coro
                except Exception as exc:  # noqa: BLE001
                    _log.warning("on_round_result hook on %s failed: %s", type(self.search).__name__, exc)

            return result
        finally:
            self._in_flight -= 1


class _NoQuestions(Exception):
    """Internal signal: eval source had nothing this tick."""
