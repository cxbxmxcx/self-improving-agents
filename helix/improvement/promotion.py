"""Default promotion policy for the candidate_wins event.

When an improvement round produces a winner, the round publishes a
CandidateWins event carrying the candidate, the reference, the round
scores, and the Improver's mode. This module registers one global
handler on the event bus that decides what to do.

Default behavior:
  - mode == "online" and event.auto_promote: archive.promote(...) is
    called with approver=f"improver:<id>" and reason=<round summary>.
    The archive's promote() in turn emits the Promoted event that
    observers (dashboard, chat UI, audit log) watch.
  - mode == "offline" or event.auto_promote is False: no-op. A human
    must promote via the chat UI / dashboard before the candidate
    becomes the live champion.

To replace this behavior with a custom promotion policy (confidence
thresholds, Pareto dominance, two-judge agreement), set
ImproverPolicy.auto_promote=False on the Improver and register your own
handler on the bus for the "candidate_wins" event. The default handler
will short-circuit because auto_promote is False, and your handler can
do whatever it wants.

DESIGN_NOTES.md section 10 has the full reasoning.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from helix.observability.bus import EventBus, get_bus
from helix.observability.events import CandidateWins

if TYPE_CHECKING:
    from helix.archive.archive import Archive

_log = logging.getLogger("helix.promotion")


# Registry: improver_id -> archive. The default handler needs an Archive
# to call promote() on, and the bus event only carries ids, so the
# Improver constructor populates this map at startup. When the Improver
# is detached we remove the entry so we don't leak archive references.
_archive_by_improver: dict[str, "Archive"] = {}


def register_improver_archive(improver_id: str, archive: "Archive") -> None:
    """Wire an Improver's archive into the default promotion handler.

    Called by Improver.__init__. Idempotent: re-registering an id
    overwrites the previous archive (useful when an Improver is rebuilt).
    """
    _archive_by_improver[improver_id] = archive


def unregister_improver_archive(improver_id: str) -> None:
    """Remove an Improver from the promotion registry (on detach)."""
    _archive_by_improver.pop(improver_id, None)


def _round_summary(event: CandidateWins) -> str:
    """One-line human-readable reason for the audit trail."""
    cs = f"{event.candidate_score:.3f}" if event.candidate_score is not None else "n/a"
    rs = f"{event.reference_score:.3f}" if event.reference_score is not None else "n/a"
    return (
        f"auto-promoted by {event.improver_id}: "
        f"candidate v{event.candidate_version} ({cs}) > "
        f"reference v{event.reference_version} ({rs})"
    )


async def default_promotion_handler(event: CandidateWins) -> None:
    """The single global handler for CandidateWins events.

    Offline mode: no-op (await a human).
    Online mode with auto_promote=False: no-op (user wrote their own handler).
    Online mode with auto_promote=True: call archive.promote() and let
    the archive emit the Promoted event downstream.
    """
    if event.mode != "online":
        return
    if not event.auto_promote:
        return

    archive = _archive_by_improver.get(event.improver_id)
    if archive is None:
        _log.warning(
            "candidate_wins: improver %s has no registered archive; "
            "skipping auto-promote",
            event.improver_id,
        )
        return

    try:
        await archive.promote(
            artifact_id=event.target_artifact_id,
            version=event.candidate_version,
            approver=f"improver:{event.improver_id}",
            reason=_round_summary(event),
        )
    except ValueError as exc:
        _log.error("auto-promote failed for %s: %s", event.improver_id, exc)


# Module-level subscription. Imported lazily by Improver.__init__ so the
# bus singleton is the one the rest of the package uses. We track whether
# we've already registered so re-imports (or test code that re-imports
# the package) don't stack duplicate handlers.
_subscribed_buses: set[int] = set()


def ensure_default_handler_registered(bus: EventBus | None = None) -> None:
    """Subscribe the default handler on the given bus (or the singleton).

    Safe to call repeatedly; only the first call per bus subscribes.
    """
    target = bus or get_bus()
    if id(target) in _subscribed_buses:
        return
    target.subscribe("candidate_wins", default_promotion_handler)
    _subscribed_buses.add(id(target))
