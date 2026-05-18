"""Observability subsystem.

Two surfaces:

1. OpenTelemetry spans via `span()` for distributed-tracing visibility.
2. A typed event bus via `EventBus.publish(event)` / `EventBus.subscribe()` for
   in-process subscribers and cross-process integration. SPEC section 10.

The event taxonomy from SPEC section 10.1 lives in `events.py`. Every event
publish also emits an OTel span attribute so a single trace backend can carry
both views. SPEC section 10.3: the bus is the integration surface.
"""

from helix.observability.bus import EventBus, get_bus
from helix.observability.console import ConsoleRenderer, attach_console_renderer
from helix.observability.events import (
    ArtifactCreated,
    ArtifactMeasured,
    BudgetExceeded,
    DriftDetected,
    Event,
    HookFired,
    ImproverRoundCompleted,
    ImproverRoundStarted,
    JudgeQuestionCompleted,
    MutationProposed,
    PairPassQuestionCompleted,
    PairPassQuestionStarted,
    ProposalCreated,
    ProposalDecided,
    SearchCompleted,
    SearchStarted,
    SearchStrategySwitched,
    SignalIdentified,
    TrajectoryCompleted,
    TrajectoryStarted,
    TrajectoryStepRecorded,
)
from helix.observability.spans import span

__all__ = [
    "EventBus",
    "get_bus",
    "ConsoleRenderer",
    "attach_console_renderer",
    "span",
    "Event",
    "ArtifactCreated",
    "ArtifactMeasured",
    "SearchStarted",
    "SearchCompleted",
    "SearchStrategySwitched",
    "HookFired",
    "TrajectoryStarted",
    "TrajectoryStepRecorded",
    "TrajectoryCompleted",
    "ProposalCreated",
    "ProposalDecided",
    "BudgetExceeded",
    "DriftDetected",
    "ImproverRoundStarted",
    "ImproverRoundCompleted",
    "MutationProposed",
    "SignalIdentified",
    "PairPassQuestionStarted",
    "PairPassQuestionCompleted",
    "JudgeQuestionCompleted",
]
