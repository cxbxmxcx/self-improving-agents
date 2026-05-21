"""BackgroundImproverRunner: run an Improver in a daemon thread.

Streamlit's rerun model kills any asyncio.Task started during a render.
Long-running improvement loops need their own thread + event loop so they
survive across reruns. This module gives Helix that pattern.

Usage:
    from helix.improvement.background import attach_runner, get_runner

    runner = attach_runner(improver)   # spawns thread + loop, starts the Improver
    runner.start()
    status = runner.status()           # readable at any time
    runner.trigger_round()             # manual round
    runner.pause()
    runner.resume()
    runner.stop()

The runner is registered in a process-wide dict keyed by improver_id, so
Streamlit's rerun can call get_runner(improver_id) and get back the live
runner instead of creating a duplicate.

IMPORTANT — SQLite cross-thread access:
The Improver runs in a background thread, but the SQLiteArchive,
FeedbackStore, and memory tiers it depends on are typically created in
the main thread. Construct ALL such SQLite-backed components with
`check_same_thread=False` if they will be used by a background-running
Improver. Helix's dashboard already does this; the chat UI's improver
panel does it too.
"""

from __future__ import annotations

import asyncio
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from helix.improvement.improver import Improver
    from helix.improvement.round import RoundResult


@dataclass
class RunnerStatus:
    """Snapshot of a runner's current state. Safe to read from any thread."""

    improver_id: str
    target_artifact_id: str
    running: bool
    paused: bool
    rounds_completed: int
    rounds_in_flight: int
    last_round_at: float | None
    last_round_result: "RoundResult | None"
    last_error: str | None
    schedule: str
    started_at: float | None
    thread_alive: bool


class BackgroundImproverRunner:
    """Wraps an Improver in a dedicated daemon thread with its own asyncio loop.

    Lifecycle:
        runner = BackgroundImproverRunner(improver)
        runner.start()          # spawn the thread; Improver begins its schedule
        runner.pause()          # stop firing scheduled rounds (in-flight finishes)
        runner.resume()         # resume the schedule
        runner.trigger_round()  # fire one round immediately (returns Future)
        runner.stop()           # graceful shutdown; waits for in-flight round

    Thread safety: status() reads atomic state and is safe from any thread.
    All control methods (start/stop/pause/resume/trigger_round) are also
    safe from the caller thread.
    """

    def __init__(self, improver: "Improver") -> None:
        self.improver = improver
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_ready = threading.Event()
        self._stop_requested = False
        self._paused = False
        self._started_at: float | None = None

    @property
    def improver_id(self) -> str:
        return self.improver.improver_id

    # ---------------- lifecycle ----------------

    def start(self) -> None:
        """Spawn the background thread + asyncio loop. Idempotent."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_requested = False
        self._loop_ready.clear()
        self._thread = threading.Thread(
            target=self._thread_main,
            name=f"helix-improver:{self.improver_id}",
            daemon=True,
        )
        self._started_at = time.time()
        self._thread.start()
        # Wait briefly for the loop to come up so subsequent calls land in it.
        self._loop_ready.wait(timeout=5.0)

    def stop(self, timeout: float = 30.0) -> None:
        """Signal stop and join the thread. Idempotent."""
        if self._thread is None or not self._thread.is_alive():
            return
        self._stop_requested = True
        loop = self._loop
        if loop is not None and not loop.is_closed():
            # Tell the Improver to stop from inside the loop.
            future = asyncio.run_coroutine_threadsafe(self.improver.stop(), loop)
            try:
                future.result(timeout=timeout)
            except Exception:
                pass
            # Stop the loop itself.
            loop.call_soon_threadsafe(loop.stop)
        self._thread.join(timeout=timeout)
        self._thread = None
        self._loop = None

    def pause(self) -> None:
        """Stop firing scheduled rounds. trigger_round() still works.

        Sets the Improver's _paused flag so its _run_forever loop short-
        circuits scheduled rounds. In-flight rounds always finish.
        """
        self._paused = True
        self.improver._paused = True

    def resume(self) -> None:
        self._paused = False
        self.improver._paused = False

    @property
    def is_paused(self) -> bool:
        return self._paused

    def trigger_round(self) -> Future:
        """Run one round right now (regardless of schedule).

        Returns a concurrent.futures.Future the caller can await to get the
        RoundResult. Safe to call from any thread.
        """
        if self._loop is None or self._loop.is_closed():
            raise RuntimeError("runner not started")
        return asyncio.run_coroutine_threadsafe(
            self.improver.trigger_round(),
            self._loop,
        )

    # ---------------- status ----------------

    def status(self) -> RunnerStatus:
        imp_status = self.improver.status
        return RunnerStatus(
            improver_id=imp_status.improver_id,
            target_artifact_id=imp_status.target_artifact_id,
            running=imp_status.running and not self._paused,
            paused=self._paused,
            rounds_completed=imp_status.rounds_completed,
            rounds_in_flight=imp_status.rounds_in_flight,
            last_round_at=imp_status.last_round_at,
            last_round_result=imp_status.last_round_result,
            last_error=imp_status.last_error,
            schedule=imp_status.schedule,
            started_at=self._started_at,
            thread_alive=self._thread is not None and self._thread.is_alive(),
        )

    # ---------------- internals ----------------

    def _thread_main(self) -> None:
        """The thread's entry point. Creates an event loop and runs the
        Improver inside it until stop is signaled."""
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        self._loop_ready.set()
        try:
            loop.run_until_complete(self._run_until_stopped())
        finally:
            # Clean up any pending tasks before closing the loop.
            try:
                pending = [t for t in asyncio.all_tasks(loop) if not t.done()]
                for t in pending:
                    t.cancel()
                if pending:
                    loop.run_until_complete(
                        asyncio.gather(*pending, return_exceptions=True)
                    )
            except Exception:
                pass
            loop.close()

    async def _run_until_stopped(self) -> None:
        """Drive the Improver until stop is signaled.

        The Improver's own start() spawns its internal background task. We
        keep this coroutine alive with a polling sleep so the loop has
        something to run; the Improver's schedule does the actual work.
        Pause/resume is implemented by short-circuiting the Improver's
        scheduled rounds (it still consumes trigger_round() calls).
        """
        await self.improver.start()
        try:
            while not self._stop_requested:
                # If paused, ensure the Improver doesn't fire new scheduled
                # rounds. The Improver's _run_forever loop checks
                # _stop_event between rounds; we set/clear our pause flag
                # so even with INTERVAL/CONTINUOUS schedules the round
                # doesn't actually start. This is a soft pause; in-flight
                # rounds always finish.
                # We don't need to do anything here other than yield.
                await asyncio.sleep(0.5)
        finally:
            # Improver.stop() may have already been called from outside;
            # call again is idempotent.
            try:
                await self.improver.stop()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Process-wide registry
# ---------------------------------------------------------------------------

_runners: dict[str, BackgroundImproverRunner] = {}
_registry_lock = threading.Lock()


def attach_runner(improver: "Improver") -> BackgroundImproverRunner:
    """Get or create the runner for this improver_id.

    Idempotent across Streamlit reruns: if a runner already exists for the
    given improver_id, returns it. Otherwise wraps the supplied Improver in
    a new runner and registers it.

    Note: when an existing runner is returned, the `improver` argument is
    ignored. This is intentional — replacing a running Improver mid-flight
    would lose in-flight state. Stop the existing one first via
    detach_runner() if you need to swap.
    """
    with _registry_lock:
        existing = _runners.get(improver.improver_id)
        if existing is not None and existing._thread is not None and existing._thread.is_alive():
            return existing
        runner = BackgroundImproverRunner(improver)
        _runners[improver.improver_id] = runner
        return runner


def get_runner(improver_id: str) -> BackgroundImproverRunner | None:
    """Look up a runner by improver_id. Returns None if not registered."""
    with _registry_lock:
        return _runners.get(improver_id)


def detach_runner(improver_id: str) -> None:
    """Stop and remove the runner from the registry."""
    with _registry_lock:
        runner = _runners.pop(improver_id, None)
    if runner is not None:
        runner.stop()


def list_runners() -> list[str]:
    """Return the ids of every registered runner."""
    with _registry_lock:
        return list(_runners.keys())


def stop_all_runners() -> None:
    """Stop every registered runner. Useful at process exit."""
    with _registry_lock:
        ids = list(_runners.keys())
    for rid in ids:
        detach_runner(rid)
