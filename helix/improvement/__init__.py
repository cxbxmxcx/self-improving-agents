"""The Improver: a long-running, async, per-artifact improvement loop.

An Improver targets one artifact on an Agent (or a standalone artifact) and
runs the search-by-signal loop continuously. The Improver is the cold path:
it observes trajectories (from the event bus), produces candidate mutations,
measures them against an eval source, and records winners to the archive.

The Agent is the hot path: it serves requests using whatever artifact the
archive currently identifies as best. Improver and Agent communicate through
the Archive (durable) and the event bus (event stream).

In production the Improver detaches: it runs as a separate service consuming
the same events off a queue, and writes updates to a shared archive. The
Agent picks up the new champion on its next request without any code change.
"""

from helix.improvement.improver import Improver, ImproverStatus
from helix.improvement.policy import ImproverMode, ImproverPolicy, Schedule
from helix.improvement.round import RoundResult, run_improvement_round

__all__ = [
    "Improver",
    "ImproverStatus",
    "ImproverMode",
    "ImproverPolicy",
    "Schedule",
    "RoundResult",
    "run_improvement_round",
]
