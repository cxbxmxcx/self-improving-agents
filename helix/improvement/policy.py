"""ImproverPolicy: when to run, how often, with what budget, and how winners
are promoted.

Three schedule modes:

- MANUAL: rounds run only when trigger_round() is called.
- INTERVAL: a round fires every N seconds, with optional jitter.
- CONTINUOUS: rounds fire as fast as the budget allows.

Two improvement modes:

- OFFLINE (default): winning candidates are recorded in the archive but
  do not become live until a human calls archive.promote(). Safe for any
  artifact layer (L1-L4). This is the deploy-gate cadence.
- ONLINE: winning candidates are auto-promoted as the live champion on
  round completion. The next request the agent serves uses the new
  version. Restricted to L1 (prompt) and L2 (memory) artifacts; the
  Improver constructor raises if asked to run online on L3 or L4.

The per-round budget caps tokens/dollars/wallclock for one round. Across
rounds the Improver can run forever; only individual rounds are bounded.
SPEC section 4.4. DESIGN_NOTES.md section 10.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from helix.search.base import SearchBudget


class Schedule(str, Enum):
    MANUAL = "manual"
    INTERVAL = "interval"
    CONTINUOUS = "continuous"


class ImproverMode(str, Enum):
    """Whether winning candidates auto-promote or wait for a human gate.

    OFFLINE: the deploy-gate cadence from Chapter 1. Winners are recorded
    but humans must promote them.

    ONLINE: the auto-apply cadence from Chapter 1. Winners reach the live
    agent on the next request. Only safe for prompt (L1) and memory (L2)
    artifacts; the Improver refuses to construct in ONLINE mode if the
    target is metacognitive (L3) or code (L4).
    """
    OFFLINE = "offline"
    ONLINE = "online"


@dataclass
class ImproverPolicy:
    schedule: Schedule = Schedule.MANUAL
    mode: ImproverMode = ImproverMode.OFFLINE
    interval_sec: float = 300.0   # how often rounds fire in INTERVAL mode
    jitter_sec: float = 0.0       # random jitter added to interval
    max_in_flight: int = 1        # only N rounds run concurrently per Improver
    quiesce_on_empty_eval: bool = True  # pause when the EvalSource has nothing
    questions_per_round: int | None = None  # None = all available
    budget_per_round: SearchBudget = field(default_factory=SearchBudget)
    promote_threshold_win_rate: float = 0.5  # candidate must beat this to be promoted
    # Whether the default candidate_wins handler should auto-promote on
    # ONLINE mode. Set False to disable the default and register your own
    # promotion policy hook (e.g., conditional on confidence thresholds or
    # Pareto dominance). Has no effect in OFFLINE mode.
    auto_promote: bool = True
    # Parallelism cap for agent runs and judge calls inside one round.
    # Default of 2 keeps peak token throughput well inside Anthropic's
    # entry-tier 450K tokens/min budget. The token-aware rate budget
    # (helix.rate_budget) provides the second line of defense. Raise this
    # to 5+ once you've tiered up your account.
    max_concurrent_questions: int = 2
    # Per-question cap on agent loop iterations and tool calls. Protects
    # against runaway costs from a single question.
    max_iterations_per_question: int = 10
    max_tool_calls_per_question: int = 20

    # ---- OnlineImprover-specific knobs (SPEC §17) ----
    # Fraction of completed trajectories that get spot-checked. 1.0 means score
    # every response; 0.05 means roughly one in twenty. Real deployments set
    # this low to keep judge cost bounded.
    sample_rate: float = 1.0
    # Rolling window for the gap signal. Drop below rolling_threshold over the
    # last rolling_window spot-checks triggers a candidate proposal.
    rolling_window: int = 4
    rolling_threshold: float = 0.70
    # When a candidate is proposed, how many subsequent requests it sees as a
    # shadow evaluation before the OnlineImprover compares it to the reference.
    shadow_sample: int = 3
