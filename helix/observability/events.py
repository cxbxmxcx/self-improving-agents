"""Typed event taxonomy. SPEC section 10.1.

Every artifact mutation, every signal measurement, every search step, every
hook firing, every trajectory step emits a structured event over the EventBus.
The framework does not ship a dashboard. It emits clean events and lets
readers plug in their existing observability stack.

This module defines the event dataclasses only. The bus that distributes them
lives in helix.observability.bus; subscribers like the console renderer live
in helix.observability.console.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Event:
    """Base class for all events.

    Every event carries a unique id, a timestamp, and an event_type slug.
    Subscribers filter on `event_type`; the EventBus dispatches by it.
    """

    event_type: str = "event"
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)

    def fields(self) -> dict[str, Any]:
        """Serialize to a flat dict suitable for OTel span attributes and JSON."""
        out: dict[str, Any] = {}
        for k, v in self.__dict__.items():
            if k in ("event_type", "event_id", "timestamp"):
                continue
            out[k] = v
        return out


# ---------------------------------------------------------------------------
# Artifact-lifecycle events
# ---------------------------------------------------------------------------

@dataclass
class ArtifactCreated(Event):
    event_type: str = "artifact_created"
    artifact_id: str = ""
    version: int = 0
    kind: str = ""
    parent_id: str | None = None
    parent_version: int | None = None
    created_by: str = ""


@dataclass
class ArtifactDuplicate(Event):
    """A mutation produced content identical to an existing version of the same
    artifact id, so it collapsed onto that version instead of creating a new row
    (SPEC §1.3). Observability can trend this as wasted search effort."""
    event_type: str = "artifact_duplicate"
    artifact_id: str = ""
    version: int = 0           # the version the search minted (not stored)
    canonical_version: int = 0  # the existing version it collapsed onto
    search_method: str = ""


@dataclass
class ArtifactMeasured(Event):
    event_type: str = "artifact_measured"
    artifact_id: str = ""
    version: int = 0
    signal_kind: str = ""
    score: float | None = None
    preference: str = "none"
    confidence: float = 0.0
    cost_tokens: int = 0
    cost_dollars: float = 0.0


# ---------------------------------------------------------------------------
# Search-lifecycle events
# ---------------------------------------------------------------------------

@dataclass
class SearchStarted(Event):
    event_type: str = "search_started"
    search_kind: str = ""
    seed_artifact_id: str = ""
    seed_version: int = 0
    budget: dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchCompleted(Event):
    event_type: str = "search_completed"
    search_kind: str = ""
    winner_artifact_id: str = ""
    winner_version: int = 0
    candidates_evaluated: int = 0
    cost_tokens: int = 0
    cost_dollars: float = 0.0


@dataclass
class SearchStrategySwitched(Event):
    """A StrategyChain Search rotated from one strategy to the next.

    Fires when the active strategy's failure budget is exhausted and the
    chain moves to the next strategy in the ordered list. `reason` explains
    the trigger ('failures_exhausted', 'all_strategies_retired', etc.).
    """
    event_type: str = "search_strategy_switched"
    from_kind: str = ""
    to_kind: str = ""
    reason: str = ""
    failures_at_switch: int = 0


@dataclass
class MutationProposed(Event):
    """A Search produced a candidate. Carries the proposer's truncated rationale.

    Pedagogical purpose: the reader can see *why* the search mutated this way,
    not just that a new variant exists. The `rationale` field is whatever the
    Search wants to surface (typically `last_feedback` from the prior round,
    truncated). The `proposer_model` is the LLM that actually wrote the new
    candidate text.
    """
    event_type: str = "mutation_proposed"
    search_kind: str = ""
    parent_artifact_id: str = ""
    parent_version: int = 0
    candidate_version: int = 0
    proposer_model: str = ""
    rationale: str = ""  # truncated by the publisher to ~150 chars
    candidate_excerpt: str = ""  # first ~150 chars of the candidate's content


# ---------------------------------------------------------------------------
# Hook events
# ---------------------------------------------------------------------------

@dataclass
class HookFired(Event):
    event_type: str = "hook_fired"
    point: str = ""
    trajectory_id: str = ""


# ---------------------------------------------------------------------------
# Trajectory events
# ---------------------------------------------------------------------------

@dataclass
class TrajectoryStarted(Event):
    event_type: str = "trajectory_started"
    trajectory_id: str = ""
    task: str = ""


@dataclass
class TrajectoryStepRecorded(Event):
    event_type: str = "trajectory_step_recorded"
    trajectory_id: str = ""
    step_index: int = 0
    step_kind: str = ""


@dataclass
class TrajectoryCompleted(Event):
    event_type: str = "trajectory_completed"
    trajectory_id: str = ""
    task: str = ""
    outcome: str = ""
    num_steps: int = 0
    final_output: str | None = None


# ---------------------------------------------------------------------------
# Improver events
# ---------------------------------------------------------------------------

@dataclass
class SignalIdentified(Event):
    """Surfaces which Signal (with model and wrapper chain) is about to judge.

    Emitted before the per-question judge loop in an improvement round. Lets
    the reader see exactly which measurement primitive is acting, since a
    plain '[judge]' tag does not tell them whether it is PairwiseJudge,
    ContrastiveJudge, a Composite, or wrapped in SwapAndAgree.
    """
    event_type: str = "signal_identified"
    improver_id: str = ""
    signal_class: str = ""
    signal_kind: str = ""
    model: str = ""
    wrapper_chain: str = ""  # e.g. "SwapAndAgree(PairwiseJudge)"


@dataclass
class ImproverRoundStarted(Event):
    event_type: str = "improver_round_started"
    improver_id: str = ""
    target_artifact_id: str = ""
    target_version: int = 0
    round_index: int = 0
    n_questions: int = 0


@dataclass
class ImproverRoundCompleted(Event):
    event_type: str = "improver_round_completed"
    improver_id: str = ""
    target_artifact_id: str = ""
    candidate_version: int = 0
    candidate_score: float | None = None
    reference_score: float | None = None
    n_candidate_wins: int = 0
    n_reference_wins: int = 0
    n_ties: int = 0
    promoted: bool = False
    elapsed_sec: float = 0.0


@dataclass
class CandidateWins(Event):
    """A candidate beat the reference in an improvement round.

    Fires after the candidate's measurement has been recorded in the
    archive. Whether the candidate becomes the live champion is decided
    by handlers on this event (see helix.improvement.promotion). Offline
    mode handlers no-op; online mode handlers call archive.promote.
    """
    event_type: str = "candidate_wins"
    improver_id: str = ""
    target_artifact_id: str = ""
    candidate_version: int = 0
    reference_version: int = 0
    candidate_score: float | None = None
    reference_score: float | None = None
    mode: str = "offline"  # "offline" | "online"
    auto_promote: bool = False  # whether the default handler should promote


@dataclass
class Promoted(Event):
    """The live champion for an artifact id just changed.

    Distinct from CandidateWins: a CandidateWins says the loop found an
    improvement; a Promoted says that improvement is now what the agent
    serves. Online winners produce both events back to back. Offline
    winners produce CandidateWins only until a human acts.
    """
    event_type: str = "promoted"
    artifact_id: str = ""
    version: int = 0
    approver: str = ""  # user id, or "improver:<id>" when automatic
    reason: str = ""


@dataclass
class PairPassQuestionStarted(Event):
    event_type: str = "pair_pass_question_started"
    improver_id: str = ""
    role: str = ""  # "reference" or "candidate"
    question_id: str = ""
    band: int = 0


@dataclass
class PairPassQuestionCompleted(Event):
    event_type: str = "pair_pass_question_completed"
    improver_id: str = ""
    role: str = ""
    question_id: str = ""
    band: int = 0
    elapsed_sec: float = 0.0
    num_tool_calls: int = 0


@dataclass
class JudgeQuestionCompleted(Event):
    event_type: str = "judge_question_completed"
    improver_id: str = ""
    question_id: str = ""
    band: int = 0
    preference: str = ""
    confidence: float = 0.0


# ---------------------------------------------------------------------------
# HITL events
# ---------------------------------------------------------------------------

@dataclass
class ProposalCreated(Event):
    event_type: str = "proposal_created"
    proposal_id: str = ""
    candidate_artifact_id: str = ""
    candidate_version: int = 0


@dataclass
class ProposalDecided(Event):
    event_type: str = "proposal_decided"
    proposal_id: str = ""
    decision: str = ""  # "approve" | "reject" | "defer"
    reviewer: str = ""


# ---------------------------------------------------------------------------
# Drift events
# ---------------------------------------------------------------------------

@dataclass
class RateLimitWaited(Event):
    """The rate budget made a caller sleep before issuing an LLM call.

    Emitted by helix.rate_budget.TokenBucket when adding the next call would
    exceed the per-minute cap. The caller resumes after wait_sec.
    """
    event_type: str = "rate_limit_waited"
    provider: str = ""
    model: str = ""
    wait_sec: float = 0.0
    reason: str = ""  # "tokens_cap" | "rpm_cap"


@dataclass
class BudgetExceeded(Event):
    """The active SearchBudget has been exhausted mid-round.

    Round drivers catch this and abort gracefully, returning a partial
    RoundResult. Useful when a round runs longer than expected and the
    user-set budget kicks in.
    """
    event_type: str = "budget_exceeded"
    improver_id: str = ""
    budget_axis: str = ""  # "tokens" | "dollars" | "wall_clock_sec" | "candidates"
    spent: float = 0.0
    cap: float = 0.0


@dataclass
class DriftDetected(Event):
    event_type: str = "drift_detected"
    artifact_id: str = ""
    version: int = 0
    drift_kind: str = ""
    severity: float = 0.0
