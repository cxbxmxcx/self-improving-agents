"""The improvers: per-artifact improvement drivers. SPEC §15, §17.

The framework defines two peer classes:

  - OfflineImprover: a long-running driver loop that tests candidates against
    a labeled EvalSource. The agent stays externally owned; the improver
    clones it via `agent.with_artifacts(...)` for each candidate.
  - OnlineImprover: an event-driven subscriber to the agent's SESSION_END
    hook. No driver loop; reacts to each completed trajectory by spot-checking,
    accumulating a rolling window, proposing candidates when scores drop, and
    auto-promoting via the CandidateWins event chain.

They share no base class. "Improver" is a category noun for either.

Both communicate with the agent through the Archive (durable) and the event
bus (live). In production either can detach to a separate service consuming
events off a queue and writing to a shared archive; the Agent picks up the
new champion on its next request.
"""

from helix.improvement.improver import OfflineImprover, ImproverStatus
from helix.improvement.online_improver import OnlineImprover, OnlineImproverStatus
from helix.improvement.policy import ImproverMode, ImproverPolicy, Schedule
from helix.improvement.round import RoundResult, run_improvement_round

__all__ = [
    "OfflineImprover",
    "OnlineImprover",
    "ImproverStatus",
    "OnlineImproverStatus",
    "ImproverMode",
    "ImproverPolicy",
    "Schedule",
    "RoundResult",
    "run_improvement_round",
]
