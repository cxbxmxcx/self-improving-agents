"""ImproverPolicy: when to run, how often, with what budget.

Three schedule modes:

- MANUAL: rounds run only when trigger_round() is called.
- INTERVAL: a round fires every N seconds, with optional jitter.
- CONTINUOUS: rounds fire as fast as the budget allows.

The per-round budget caps tokens/dollars/wallclock for one round. Across
rounds the Improver can run forever; only individual rounds are bounded.
SPEC section 4.4.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from helix.search.base import SearchBudget


class Schedule(str, Enum):
    MANUAL = "manual"
    INTERVAL = "interval"
    CONTINUOUS = "continuous"


@dataclass
class ImproverPolicy:
    schedule: Schedule = Schedule.MANUAL
    interval_sec: float = 300.0   # how often rounds fire in INTERVAL mode
    jitter_sec: float = 0.0       # random jitter added to interval
    max_in_flight: int = 1        # only N rounds run concurrently per Improver
    quiesce_on_empty_eval: bool = True  # pause when the EvalSource has nothing
    questions_per_round: int | None = None  # None = all available
    budget_per_round: SearchBudget = field(default_factory=SearchBudget)
    promote_threshold_win_rate: float = 0.5  # candidate must beat this to be promoted
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
