"""Default console subscriber.

Renders events as they happen, one line per event. The reader running a
chapter script sees the cold path's progress streaming to their terminal
even though no print() lives in their code. SPEC section 10.2 — this is one
of the bus's consumers, alongside drift detection and cost dashboards.

In production the same events go through Phoenix / Langfuse / etc. via OTel
exporters; the console renderer is for local development and book pedagogy.
"""

from __future__ import annotations

import sys
from typing import Any, TextIO

from helix.observability.bus import EventBus, get_bus
from helix.observability.events import (
    ArtifactCreated,
    ArtifactMeasured,
    Event,
    HookFired,
    ImproverRoundCompleted,
    ImproverRoundStarted,
    JudgeQuestionCompleted,
    MutationProposed,
    PairPassQuestionCompleted,
    PairPassQuestionStarted,
    RateLimitWaited,
    SearchCompleted,
    SearchStarted,
    SearchStrategySwitched,
    SignalIdentified,
    TrajectoryCompleted,
    TrajectoryStarted,
)


class ConsoleRenderer:
    """Pretty-print events to stderr as they flow through the bus.

    Verbose mode (the default) prints all primitive events with hierarchy.
    Non-verbose mode filters to a smaller set of progress events.
    """

    DEFAULT_INTERESTING = {
        "improver_round_started",
        "improver_round_completed",
        "pair_pass_question_completed",
        "judge_question_completed",
        "search_started",
        "search_completed",
        "search_strategy_switched",
        "signal_identified",
        "artifact_measured",
        "rate_limit_waited",
    }

    VERBOSE_INCLUDE = DEFAULT_INTERESTING | {
        "pair_pass_question_started",
        "mutation_proposed",
        "artifact_created",
    }

    def __init__(
        self,
        stream: TextIO = sys.stderr,
        verbose: bool = True,
        include: set[str] | None = None,
    ) -> None:
        self.stream = stream
        self.verbose = verbose
        if include is not None:
            self.include = include
        elif verbose:
            self.include = self.VERBOSE_INCLUDE
        else:
            self.include = self.DEFAULT_INTERESTING

    def attach(self, bus: EventBus | None = None) -> None:
        bus = bus or get_bus()
        bus.subscribe("*", self._handle)

    def _handle(self, event: Event) -> None:
        if not self.verbose and event.event_type not in self.include:
            return
        line = self._render(event)
        if line:
            print(line, file=self.stream, flush=True)

    def _render(self, event: Event) -> str:
        et = event.event_type
        if isinstance(event, ImproverRoundStarted):
            return (
                f"\n[improver round {event.round_index}] START  "
                f"target=({event.target_artifact_id}, v{event.target_version})  "
                f"questions={event.n_questions}"
            )
        if isinstance(event, ImproverRoundCompleted):
            cscore = f"{event.candidate_score:.3f}" if event.candidate_score is not None else "n/a"
            rscore = f"{event.reference_score:.3f}" if event.reference_score is not None else "n/a"
            tag = "PROMOTED" if event.promoted else "kept reference"
            return (
                f"[improver] round complete  "
                f"cand v{event.candidate_version} score={cscore}  ref score={rscore}  "
                f"W/L/T={event.n_candidate_wins}/{event.n_reference_wins}/{event.n_ties}  "
                f"{tag}  [{event.elapsed_sec:.1f}s]\n"
            )
        if isinstance(event, SearchStarted):
            return (
                f"  [search {event.search_kind:<14}] propose START  "
                f"seed=v{event.seed_version}"
            )
        if isinstance(event, SearchCompleted):
            return (
                f"  [search {event.search_kind:<14}] propose DONE   "
                f"winner=v{event.winner_version}  candidates_evaluated={event.candidates_evaluated}  "
                f"tokens={event.cost_tokens}"
            )
        if isinstance(event, RateLimitWaited):
            return (
                f"  [rate budget] ⏳ waited {event.wait_sec:.1f}s  "
                f"({event.provider}/{event.model})  reason={event.reason}"
            )
        if isinstance(event, SearchStrategySwitched):
            return (
                f"\n>>> [strategy chain] switching {event.from_kind} -> {event.to_kind}  "
                f"reason={event.reason}  failures={event.failures_at_switch}\n"
            )
        if isinstance(event, MutationProposed):
            rationale = event.rationale[:120].replace("\n", " ")
            excerpt = event.candidate_excerpt[:80].replace("\n", " ")
            return (
                f"  [search {event.search_kind:<14}] mutation by {event.proposer_model}\n"
                f"    rationale: {rationale}\n"
                f"    candidate v{event.candidate_version} excerpt: {excerpt}..."
            )
        if isinstance(event, SignalIdentified):
            wrapped = (
                f"{event.wrapper_chain}({event.signal_class}, model={event.model})"
                if event.wrapper_chain
                else f"{event.signal_class}(model={event.model})"
            )
            return f"  [signal] judging with {wrapped}  kind={event.signal_kind}"
        if isinstance(event, ArtifactCreated):
            parent = f"v{event.parent_version}" if event.parent_version else "genesis"
            return f"  [artifact created  ] {event.artifact_id} v{event.version}  parent={parent}  by={event.created_by}"
        if isinstance(event, ArtifactMeasured):
            score = f"{event.score:.3f}" if event.score is not None else "n/a"
            role = event.signal_kind
            return (
                f"  [artifact measured ] {event.artifact_id} v{event.version}  "
                f"role={role}  score={score}  pref={event.preference}"
            )
        if isinstance(event, PairPassQuestionStarted):
            return f"    [{event.role:>9}] {event.question_id} (band {event.band})  starting..."
        if isinstance(event, PairPassQuestionCompleted):
            return (
                f"    [{event.role:>9}] {event.question_id}  done  "
                f"{event.elapsed_sec:>4.1f}s  {event.num_tool_calls} tool calls"
            )
        if isinstance(event, JudgeQuestionCompleted):
            arrow = "<-" if event.preference == "left" else ("->" if event.preference == "right" else "==")
            return (
                f"    [judge   ] {event.question_id} (band {event.band})  "
                f"{arrow}  preference={event.preference}  confidence={event.confidence:.2f}"
            )
        if isinstance(event, TrajectoryStarted):
            return f"      [trajectory ] started  {event.trajectory_id[:8]}  task={event.task[:60]}"
        if isinstance(event, TrajectoryCompleted):
            return f"      [trajectory ] {event.outcome}  steps={event.num_steps}  {event.trajectory_id[:8]}"
        if isinstance(event, HookFired):
            return f"      [hook       ] {event.point}  {event.trajectory_id[:8]}"
        # Fallback for verbose mode or new event types
        return f"  [{et}] {event.fields()}"


def attach_console_renderer(verbose: bool = True) -> ConsoleRenderer:
    """Convenience: create a ConsoleRenderer, attach it to the default bus."""
    r = ConsoleRenderer(verbose=verbose)
    r.attach()
    return r
