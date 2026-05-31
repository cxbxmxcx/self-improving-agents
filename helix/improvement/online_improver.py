"""OnlineImprover: the event-driven per-artifact improver. SPEC §17.

Where OfflineImprover owns a driver loop and tests candidates against a
labeled EvalSource, OnlineImprover has no driver loop. It subscribes to the
agent's SESSION_END hook and reacts to each completed trajectory.

Lifecycle on a completed trajectory:

  1. Spot-check the trajectory with the signal (rubric judge or similar).
  2. Append the score to a rolling window.
  3. If the rolling average drops below threshold and the window is full,
     propose a candidate via search.propose().
  4. Shadow-evaluate the candidate on the next `policy.shadow_sample` real
     requests (both reference and candidate run; the candidate's output is
     logged only, not served).
  5. If the candidate's mean shadow score beats the reference's, publish
     `CandidateWins(mode="online", auto_promote=True)`. The default
     promotion handler chain writes `archive.promote()`. The next request
     reads the new live champion.

The agent is never blocked by the improver. Spot-check and propose work
happens in the background after SESSION_END fires; the user's response
goes out as soon as the agent's loop completes.

Layer safety: the constructor refuses target artifacts at layer ≥ 3.
Online auto-promotes; L3 (planner/monitor) and L4 (code) need a deploy
gate. The check happens at construction time, the earliest possible
failure.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections import deque
from dataclasses import dataclass
from typing import Any

from helix.archive.archive import Archive
from helix.artifact import Artifact
from helix.hooks import HookPoint
from helix.improvement.guards import reject_composite_constituent
from helix.improvement.policy import ImproverPolicy
from helix.improvement.promotion import (
    ensure_default_handler_registered,
    register_improver_archive,
)
from helix.observability.bus import EventBus, get_bus
from helix.observability.events import CandidateWins
from helix.search.base import SearchBudget, Variant
from helix.signal import GapMeasurement, Signal
from helix.trajectory import Trajectory

_log = logging.getLogger("helix.online_improver")


@dataclass
class OnlineImproverStatus:
    improver_id: str
    target_artifact_id: str
    running: bool
    trajectories_seen: int
    spot_checks_done: int
    candidates_proposed: int
    candidates_promoted: int
    rolling_average: float | None
    last_error: str | None


class OnlineImprover:
    """Event-driven per-artifact improver. SPEC §17.

    Constructor takes the Agent under improvement, the target artifact id, a
    signal that can score one trajectory (rubric judge or similar), a Search
    method for proposing candidates, an Archive, and a policy. No eval source:
    the agent's live trajectories are the source of measurement.
    """

    def __init__(
        self,
        *,
        agent,                          # Agent under improvement
        target_artifact_id: str,
        signal: Signal,
        search,                         # Search instance
        archive: Archive,
        policy: ImproverPolicy | None = None,
        improver_id: str | None = None,
        bus: EventBus | None = None,
    ) -> None:
        self.agent = agent
        self.target_artifact_id = target_artifact_id
        self.signal = signal
        self.search = search
        self.archive = archive
        self.policy = policy or ImproverPolicy()
        self.improver_id = improver_id or f"online-{uuid.uuid4().hex[:8]}"
        self.bus = bus or get_bus()

        # Layer safety: online auto-promotes; L3/L4 need a deploy gate.
        # Resolve the target artifact wherever it is bound on the agent (not
        # just the system prompt) so the check fires for any L3/L4 target, then
        # raise at construction. SPEC §17.3. A target the agent does not bind
        # is a misconfiguration, not a deployed L3/L4 artifact, so it is left
        # for the first proposal to surface rather than failing construction.
        target_art = None
        if hasattr(self.agent, "find_artifact"):
            target_art = self.agent.find_artifact(target_artifact_id)
        if target_art is None and getattr(self.agent, "system_prompt", None) is not None:
            if self.agent.system_prompt.id == target_artifact_id:
                target_art = self.agent.system_prompt
        if target_art is not None and target_art.layer >= 3:
            raise ValueError(
                f"OnlineImprover {self.improver_id}: cannot target an "
                f"L{target_art.layer} artifact (kind={target_art.kind.value}"
                f"{', subtype=' + target_art.subtype.value if target_art.subtype else ''}). "
                f"Online auto-promotes; L3/L4 need a deploy gate. "
                f"Use OfflineImprover for L3/L4 work."
            )

        # Wire into the default promotion handler chain so CandidateWins
        # produces an archive.promote() under online + auto_promote.
        # Replace-enforcement: a constituent of a declared composite is improved
        # jointly, not by a solo improver (SPEC §18.7).
        reject_composite_constituent(self.agent, target_artifact_id, self.improver_id)

        register_improver_archive(self.improver_id, self.archive)
        ensure_default_handler_registered(self.bus)

        # Runtime state
        self._rolling: deque[float] = deque(maxlen=self.policy.rolling_window)
        self._trajectories_seen = 0
        self._spot_checks_done = 0
        self._candidates_proposed = 0
        self._candidates_promoted = 0
        self._last_error: str | None = None
        self._stopped = True
        self._subscribed = False  # SESSION_END hook registered (idempotent)
        self._candidate_in_flight: Variant | None = None
        self._shadow_buffer: list[tuple[list[float], list[float]]] = []
        # Reference to the live champion the improver is currently comparing
        # against. None until the first spot-check; refreshed from the archive
        # on each candidate proposal.
        self._reference: Artifact | None = None
        # Backlog of in-flight shadow evaluations keyed by candidate variant.
        self._shadow_ref_scores: list[float] = []
        self._shadow_cand_scores: list[float] = []
        # Per-request (reference, candidate) score pairs. Kept alongside the
        # flat lists so the promotion gate can compute a per-request win rate
        # rather than only comparing averages (SPEC §17.2 step 5).
        self._shadow_pairs: list[tuple[float, float]] = []
        self._shadow_requests_remaining: int = 0

    # ---------------- lifecycle ----------------

    def _ensure_subscribed(self) -> None:
        """Register the SESSION_END handler exactly once. Subscription is kept
        separate from running state so attach (the wiring step) and start/stop
        compose without double-registering on a stop/start cycle."""
        if self._subscribed:
            return
        self.agent.hooks.register(HookPoint.SESSION_END, self._on_session_end)
        self._subscribed = True

    def on_attached(self) -> None:
        """The wiring step (SPEC §17.1): agent.attach_improver calls this to
        subscribe the improver to the agent's SESSION_END hook."""
        self._ensure_subscribed()

    async def start(self) -> None:
        """Activate the improver, subscribing if attach did not already. The
        SESSION_END handler no-ops while stopped, so this just flips it live."""
        self._ensure_subscribed()
        self._stopped = False
        _log.info("OnlineImprover %s active on %s SESSION_END",
                  self.improver_id, type(self.agent).__name__)

    async def stop(self) -> None:
        """Mark the improver stopped. The SESSION_END handler stays registered
        but no-ops via the _stopped flag (HookRegistry has no remove API)."""
        self._stopped = True

    @property
    def status(self) -> OnlineImproverStatus:
        avg = (
            sum(self._rolling) / len(self._rolling)
            if self._rolling
            else None
        )
        return OnlineImproverStatus(
            improver_id=self.improver_id,
            target_artifact_id=self.target_artifact_id,
            running=not self._stopped,
            trajectories_seen=self._trajectories_seen,
            spot_checks_done=self._spot_checks_done,
            candidates_proposed=self._candidates_proposed,
            candidates_promoted=self._candidates_promoted,
            rolling_average=avg,
            last_error=self._last_error,
        )

    # ---------------- hook handler ----------------

    async def _on_session_end(self, trajectory: Trajectory, **_: Any) -> None:
        """The agent just completed a request. Spot-check, accumulate, decide."""
        if self._stopped:
            return
        self._trajectories_seen += 1

        # Probabilistic sampling.
        if self._sample_rate_skip():
            return

        try:
            await self._handle_trajectory(trajectory)
        except Exception as exc:  # noqa: BLE001
            self._last_error = f"{type(exc).__name__}: {exc}"
            _log.exception("OnlineImprover %s handler failed: %s", self.improver_id, exc)

    def _sample_rate_skip(self) -> bool:
        rate = max(0.0, min(1.0, self.policy.sample_rate))
        if rate >= 1.0:
            return False
        if rate <= 0.0:
            return True
        # Deterministic per-trajectory: every 1/rate trajectory is scored.
        step = max(1, int(round(1.0 / rate)))
        return (self._trajectories_seen - 1) % step != 0

    async def _handle_trajectory(self, trajectory: Trajectory) -> None:
        # If a candidate is in flight, route this trajectory through the
        # shadow-eval bookkeeping instead of the rolling window.
        if self._candidate_in_flight is not None:
            await self._handle_shadow_request(trajectory)
            return

        # Resolve the live champion so the signal scores the trajectory in
        # context. The agent already used it; we just need the artifact.
        live = await self._resolve_live_champion()
        if live is None:
            return  # nothing to compare against yet

        m = await self.signal.measure(candidate=live, trajectory=trajectory)
        self._spot_checks_done += 1
        if m.score is None:
            return
        self._rolling.append(m.score)

        if len(self._rolling) < self.policy.rolling_window:
            return
        avg = sum(self._rolling) / len(self._rolling)
        if avg >= self.policy.rolling_threshold:
            return

        # Rolling average dropped below threshold. Propose a candidate.
        await self._propose_candidate(seed=live)

    async def _resolve_live_champion(self) -> Artifact | None:
        live = await self.archive.live_champion(self.target_artifact_id)
        if live is not None:
            return live
        # Fall back to the agent's current binding for the artifact (Ch 2:
        # system_prompt). Production deployments should always have a live
        # champion recorded, so this is a safety net for first-run cases.
        if hasattr(self.agent, "system_prompt") and self.agent.system_prompt.id == self.target_artifact_id:
            return self.agent.system_prompt
        return None

    async def _propose_candidate(self, *, seed: Artifact) -> None:
        budget = self.policy.budget_per_round
        try:
            async for variant in self.search.propose(
                seed=seed, signal=self.signal, archive=self.archive, budget=budget
            ):
                self._candidate_in_flight = variant
                self._candidates_proposed += 1
                self._reference = seed
                self._shadow_ref_scores = []
                self._shadow_cand_scores = []
                self._shadow_pairs = []
                self._shadow_requests_remaining = self.policy.shadow_sample
                _log.info(
                    "OnlineImprover %s proposed candidate v%d (target %s)",
                    self.improver_id, variant.artifact.version, self.target_artifact_id,
                )
                return
        except Exception as exc:  # noqa: BLE001
            self._last_error = f"propose failed: {type(exc).__name__}: {exc}"
            _log.exception("OnlineImprover %s propose failed: %s", self.improver_id, exc)

    async def _handle_shadow_request(self, trajectory: Trajectory) -> None:
        """During shadow eval the next N user requests get measured twice:
        the live agent already answered (that's `trajectory`); now we also
        run the candidate against the same task and score both."""
        assert self._candidate_in_flight is not None
        assert self._reference is not None
        candidate_prompt = self._candidate_in_flight.artifact

        # Score the live agent's trajectory under the live prompt.
        ref_m = await self.signal.measure(
            candidate=self._reference, trajectory=trajectory
        )
        ref_score = ref_m.score
        if ref_score is not None:
            self._shadow_ref_scores.append(ref_score)

        # Run the candidate against the same task and score it.
        clone = self.agent.with_artifacts({self.target_artifact_id: candidate_prompt})
        cand_score: float | None = None
        try:
            answer, cand_traj = await clone.run(trajectory.task)
        except Exception as exc:  # noqa: BLE001
            self._last_error = f"shadow run failed: {type(exc).__name__}: {exc}"
            cand_traj = None
        if cand_traj is not None:
            cand_m = await self.signal.measure(
                candidate=candidate_prompt, trajectory=cand_traj
            )
            if cand_m.score is not None:
                cand_score = cand_m.score
                self._shadow_cand_scores.append(cand_score)

        # Pair the two scores for this same request so the promotion gate can
        # measure a per-request win rate, not just a difference of averages.
        if ref_score is not None and cand_score is not None:
            self._shadow_pairs.append((ref_score, cand_score))

        self._shadow_requests_remaining -= 1
        if self._shadow_requests_remaining > 0:
            return

        # Shadow eval complete. Compare averages and (if cand wins) fire
        # CandidateWins to invoke the promotion handler chain.
        await self._finalize_shadow()

    async def _finalize_shadow(self) -> None:
        candidate_variant = self._candidate_in_flight
        reference = self._reference
        ref_scores = self._shadow_ref_scores
        cand_scores = self._shadow_cand_scores
        pairs = self._shadow_pairs
        # Reset shadow state regardless of outcome so the next round can start.
        self._candidate_in_flight = None
        self._reference = None
        self._shadow_ref_scores = []
        self._shadow_cand_scores = []
        self._shadow_pairs = []
        self._shadow_requests_remaining = 0
        self._rolling.clear()

        if not ref_scores or not cand_scores or candidate_variant is None or reference is None:
            return

        ref_avg = sum(ref_scores) / len(ref_scores)
        cand_avg = sum(cand_scores) / len(cand_scores)

        # Record both shadow measurements so the dashboard can show the run.
        ref_measurement = GapMeasurement(
            score=ref_avg,
            confidence=min(1.0, len(ref_scores) / max(1, self.policy.shadow_sample)),
            signal_id=self.signal.signal_id,
            signal_version=self.signal.signal_version,
            metadata={
                "role": "reference_shadow",
                "n_questions": len(ref_scores),
                "improver_id": self.improver_id,
            },
        )
        cand_measurement = GapMeasurement(
            score=cand_avg,
            confidence=min(1.0, len(cand_scores) / max(1, self.policy.shadow_sample)),
            signal_id=self.signal.signal_id,
            signal_version=self.signal.signal_version,
            metadata={
                "role": "candidate_shadow",
                "n_questions": len(cand_scores),
                "improver_id": self.improver_id,
            },
        )
        await self.archive.record(candidate_variant, cand_measurement)
        ref_variant = Variant(
            artifact=reference,
            parent=reference,
            search_method=reference.created_by,
            metadata={"role": "reference_in_online_round"},
        )
        await self.archive.record(ref_variant, ref_measurement)

        # The candidate has to beat the reference at all before we signal a
        # win; sub-par candidates are recorded above but go no further.
        if cand_avg <= ref_avg:
            return

        # Promotion gate: the candidate must win at least
        # promote_threshold_win_rate of the shadow requests head-to-head
        # (SPEC §17.2 step 5). Comparing per-request wins, not just averages,
        # stops a candidate that wins big on one request but loses the
        # majority from auto-deploying.
        if pairs:
            wins = sum(1 for r, c in pairs if c > r)
            win_rate = wins / len(pairs)
        else:
            win_rate = 0.0
        cleared_threshold = win_rate >= self.policy.promote_threshold_win_rate

        # Candidate beat the reference. Fire CandidateWins so the improvement
        # is observable; the default promotion handler auto-promotes only when
        # auto_promote is set AND the win-rate bar was cleared.
        await self.bus.publish(CandidateWins(
            improver_id=self.improver_id,
            target_artifact_id=self.target_artifact_id,
            candidate_version=candidate_variant.artifact.version,
            reference_version=reference.version,
            candidate_score=cand_avg,
            reference_score=ref_avg,
            mode="online",
            auto_promote=self.policy.auto_promote and cleared_threshold,
        ))
        if self.policy.auto_promote and cleared_threshold:
            self._candidates_promoted += 1
