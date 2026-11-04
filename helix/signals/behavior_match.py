"""BehaviorMatchSignal: whether learned memory behaves exactly like the memory it must match.

Chapter 6 measures a learning mechanism by what its memory does, not how it
reads. Two functions match on a case when they return the same result for
the same input; text, style, and comments carry no weight. The score is the
share of cases that match, and anything short of every case triggers.

The caller supplies `run`, which executes a function's source on one case
and returns its result, so the signal knows nothing about calendars. A
function that fails on a case counts as a mismatch there, and the first few
mismatches are reported as feedback a person can read.
"""

from __future__ import annotations

import hashlib
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from helix._async import resolve
from helix.artifact import Artifact
from helix.signal import Cost, GapMeasurement, SignalKind, derive_signal_id

Run = Callable[[str, Any], Any | Awaitable[Any]]
SHOWN = 3  # mismatches written into the feedback


class BehaviorMatchSignal:
    """Share of cases where the candidate's result equals the reference's. SPEC §3."""

    def __init__(self, reference_source: str, run: Run, *, case_id: Callable[[Any], str] = str,
                 version: int = 1) -> None:
        self.reference_source = reference_source
        self.run = run
        self.case_id = case_id
        self._version = version

    @property
    def kind(self) -> SignalKind:
        return SignalKind.GROUND_TRUTH

    @property
    def cost_estimate(self) -> Cost:
        return Cost(wall_clock_sec=0.01)

    @property
    def signal_id(self) -> str:
        digest = hashlib.sha256(self.reference_source.encode("utf-8")).hexdigest()[:12]
        return derive_signal_id("BehaviorMatchSignal", {"reference": digest})

    @property
    def signal_version(self) -> int:
        return self._version

    async def _result(self, source: str, case: Any) -> tuple[bool, Any]:
        try:
            return True, await resolve(self.run(source, case))
        except Exception as exc:  # a crash on a case is a mismatch on that case
            return False, f"{type(exc).__name__}: {exc}"

    async def measure(self, candidate: Artifact | str, trajectory: Any = None, reference: Any = None,
                      ground_truth: Sequence[Any] | None = None) -> GapMeasurement:
        cases = list(ground_truth or [])
        if not cases:
            return GapMeasurement(confidence=0.0, signal_id=self.signal_id, signal_version=self.signal_version,
                                  feedback="behavior match: no cases to compare", metadata={"empty": True})
        source = candidate.content if isinstance(candidate, Artifact) else candidate
        mismatches = []
        for case in cases:
            _, expected = await self._result(self.reference_source, case)
            ok, got = await self._result(source, case)
            if not ok or got != expected:
                mismatches.append({"case": self.case_id(case), "expected": expected, "got": got})
        matched = len(cases) - len(mismatches)
        feedback = None if not mismatches else "; ".join(
            f"{m['case']}: expected {m['expected']}, got {m['got']}" for m in mismatches[:SHOWN])
        return GapMeasurement(
            score=matched / len(cases),
            signal_id=self.signal_id,
            signal_version=self.signal_version,
            triggered=bool(mismatches),
            raw_value=float(matched),
            feedback=feedback,
            metadata={"cases": len(cases), "matched": matched,
                      "mismatched": [m["case"] for m in mismatches]},
        )
