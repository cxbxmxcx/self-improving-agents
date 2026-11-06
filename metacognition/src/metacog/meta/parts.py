"""The policy parts of the online loop: verdicts, router, controller, detector, and gate.

These follow chapter 6 (`chapters/ch06/assistant/online.py`): the controller
chooses a reasoning setting by lookup, the detector quarantines a procedure
when `limit` of its last `window` judged uses were rejected, and the gate
admits a new version only when it clears a bar on held-back cases. The
defaults are the study's, which it notes were set by hand, not tuned. The
router is new: the study had one procedure and one kind of request, so it
never had to decide which procedure a request belongs to.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from metacog.ladders import bare

# Seven of eight: one miss is tolerated, two are not (helix.eval.effort.ADEQUATE).
ADEQUATE = 7 / 8
SOLID = 1.0
EFFORTS = ("none", "low", "medium", "high")

# Try a procedure's source on one recorded case: (reward, reason, observations).
Practice = Callable[[str, str], Awaitable[tuple[float, str, str]]]


@dataclass
class Verdict:
    """Whether a run's outcome was accepted, why, and who said so."""

    accepted: bool
    reason: str = ""
    # "verifier" (code), "human" (feedback), "judge" (a model), or "harness" (the run failed outright).
    source: str = "verifier"

    @classmethod
    def coerce(cls, value: Any, source: str = "verifier") -> Verdict | None:
        """A Verdict from a Verdict, a bool, a (bool, reason) pair, or a checker dict such as
        {"success": True, "reason": "..."}; None stays None (no verdict yet)."""
        if value is None or isinstance(value, Verdict):
            return value
        if isinstance(value, bool):
            return cls(value, "", source)
        if isinstance(value, tuple) and len(value) == 2:
            return cls(bool(value[0]), str(value[1]), source)
        if isinstance(value, dict):
            accepted = value.get("accepted", value.get("success"))
            if accepted is not None:
                return cls(bool(accepted), str(value.get("reason", "")), source)
        raise TypeError(f"a verifier must return a Verdict, bool, (bool, reason), or dict, not {value!r}")


@dataclass
class Router:
    """Which procedures a request belongs to.

    Each procedure is scored by lexical similarity between the request and the
    procedure's description, or its closest example request when it has
    examples. A procedure matches when its score clears its threshold, and the
    `k` best matches are kept. Without thresholds every request matches the `k`
    best procedures, which suited the study: one procedure, one kind of
    request. `metacog.offline.calibrate_routing` sets thresholds from labeled
    requests. A request that matches nothing is in no procedure's class: it
    runs at the agent's own setting and teaches no procedure anything.
    """

    k: int = 2
    thresholds: dict[str, float] = field(default_factory=dict)
    default: float = 0.0
    examples: dict[str, list[str]] = field(default_factory=dict)
    _vectors: dict[str, list[float]] = field(default_factory=dict, init=False, repr=False, compare=False)

    def _vector(self, text: str) -> list[float]:
        from metacog.retrieval.lexical import embed

        if text not in self._vectors:
            self._vectors[text] = embed([text])[0]
        return self._vectors[text]

    def scores(self, request: str, descriptions: Mapping[str, str], *, leave_out: str | None = None
               ) -> dict[str, float]:
        """Each procedure's score for `request`; `leave_out` drops one example (for calibration)."""
        query = self._vector(request)
        out = {}
        for name, description in descriptions.items():
            texts = [description, *(e for e in self.examples.get(name, []) if e != leave_out)]
            out[name] = max(sum(a * b for a, b in zip(query, self._vector(t), strict=True)) for t in texts)
        return out

    def match(self, request: str, descriptions: Mapping[str, str]) -> tuple[list[str], dict[str, float]]:
        """The matching procedures, best first, and every procedure's score."""
        scores = self.scores(request, descriptions)
        ranked = sorted(scores.items(), key=lambda pair: -pair[1])  # stable: ties keep declaration order
        matched = [name for name, score in ranked[: self.k] if score >= self.thresholds.get(name, self.default)]
        return matched, scores

    def to_dict(self) -> dict[str, Any]:
        return {"k": self.k, "thresholds": dict(self.thresholds), "default": self.default,
                "examples": {name: list(texts) for name, texts in self.examples.items()}}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Router:
        return cls(k=data.get("k", 2), thresholds=dict(data.get("thresholds", {})), default=data.get("default", 0.0),
                   examples={name: list(texts) for name, texts in data.get("examples", {}).items()})


@dataclass
class Controller:
    """Reasoning by lookup: the setting each memory state was measured to need.

    `table` maps "memory" (a trusted procedure is offered) and "no_memory" to a
    reasoning setting, for every model and procedure; `per_procedure` overrides
    it by procedure. Both are set by hand. `per_model` holds measured floors by
    model, as `Floors.controller()` writes them: {model: {"table": {...},
    "per_procedure": {...}}}. Measured floors come first for their model. Once
    any model has measured floors, a model without them (and without a
    hand-set table) runs at the agent's own setting until it is calibrated.
    With no tables at all, a trusted procedure runs at the lowest rung and
    everything else at the agent's own setting.
    """

    table: dict[str, str] | None = None
    per_procedure: dict[str, dict[str, str]] = field(default_factory=dict)
    per_model: dict[str, dict[str, Any]] = field(default_factory=dict)
    ladder: tuple[str, ...] = EFFORTS

    def effort(self, procedure: str | None, trusted: bool, model: str | None = None) -> str | None:
        arm = "memory" if trusted else "no_memory"
        measured = (self.per_model.get(bare(model)) or {}) if model else {}
        for table in ((measured.get("per_procedure") or {}).get(procedure or ""), measured.get("table"),
                      self.per_procedure.get(procedure or ""), self.table):
            if table is not None:
                return table.get(arm)
        if self.per_model:
            return None  # floors were measured for other models only
        return self.ladder[0] if trusted else None

    @classmethod
    def from_floors(cls, *floors: str | Path | Mapping[str, Any]) -> Controller:
        """Build from one or more floor tables (paths or dicts). A table is {"memory": ..., "no_memory": ...},
        or that nested by procedure; one with a "model" (as `Floors.save` writes) holds that model's floors."""
        controller = cls()
        for item in floors:
            data = json.loads(Path(item).read_text(encoding="utf-8")) if isinstance(item, (str, Path)) else item
            if {"memory", "no_memory"} & set(data):
                table = {k: data[k] for k in ("memory", "no_memory") if k in data}
                if data.get("model"):  # arms calibrated separately add up
                    controller.per_model.setdefault(bare(data["model"]), {}).setdefault("table", {}).update(table)
                else:
                    controller.table = {**(controller.table or {}), **table}
            else:
                controller.per_procedure.update({name: dict(table) for name, table in data.items()})
        return controller


@dataclass
class Detector:
    """Notice when a trusted procedure stops working: `limit` rejections in the last `window` judged uses.

    `limits` lowers the bar for a source you trust more: {"human": 1} quarantines on one rejection from
    a person, while other sources, such as a judge, still need `limit`. When only a sample of runs is
    reviewed, and late, waiting for a second reviewed failure keeps a stale procedure in service (D2).
    """

    window: int = 4
    limit: int = 2
    limits: dict[str, int] = field(default_factory=dict)

    def observe(self, recent: list[bool], accepted: bool) -> tuple[list[bool], bool]:
        recent = [*recent, accepted][-self.window:]
        return recent, sum(not ok for ok in recent) >= self.limit

    def record(self, window: list[list[Any]], run_id: str | None, accepted: bool,
               counted: Sequence[str] = (), source: str | None = None) -> tuple[list[list[Any]], bool]:
        """`observe`, one verdict per run: the window holds [run_id, accepted, source] entries, and a later
        verdict on a run still in it (a person's review after a judge's) replaces the earlier one. A run
        counted once and since gone from the window (`counted`) isn't counted again. Returns the window and
        whether to quarantine."""
        entries = [[entry[0], entry[1], entry[2] if len(entry) > 2 else None] for entry in window]
        for entry in entries:
            if run_id is not None and entry[0] == run_id:
                entry[1], entry[2] = accepted, source
                break
        else:
            if run_id is not None and run_id in counted:
                return entries, False
            entries = [*entries, [run_id, accepted, source]][-self.window:]
        rejections = [entry[2] for entry in entries if not entry[1]]
        return entries, len(rejections) >= self.limit or any(
            rejections.count(name) >= limit for name, limit in self.limits.items())


@dataclass
class LearningBudget:
    """What learning may spend on one procedure in any 24 hours: attempts, dollars, or both.

    It is checked before an attempt starts, so one attempt can carry the spend past the
    dollar limit; the next one then waits. A skipped attempt is logged with the reason.
    """

    attempts_per_day: int | None = None
    dollars_per_day: float | None = None

    def refusal(self, recent: list[dict[str, Any]]) -> str | None:
        """Why a new attempt may not start, given the attempts of the last 24 hours, or None."""
        spent = [a for a in recent if not str(a.get("outcome", "")).startswith("skipped")]
        if self.attempts_per_day is not None and len(spent) >= self.attempts_per_day:
            return f"{len(spent)} attempts in the last 24 hours, the budget's limit"
        dollars = sum(a.get("dollars") or 0.0 for a in spent)
        if self.dollars_per_day is not None and dollars >= self.dollars_per_day:
            return f"${dollars:.4f} spent in the last 24 hours of a ${self.dollars_per_day:.2f} budget"
        return None


@dataclass
class GateResult:
    accepted: int
    total: int
    passed: bool
    failures: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"accepted": self.accepted, "total": self.total, "passed": self.passed, "failures": self.failures}


@dataclass
class Gate:
    """Admit a version only when it clears the bar on cases it never learned from.

    `cases` are case ids to practice on. Or set `holdout`: the last `holdout`
    accepted runs since the procedure went untrusted are held back from the
    learner and become the cases, so relearning waits until there are enough.

    `practice` is how the gate tries a candidate on its cases, when that differs
    from how the learner practices: the learner can replay recorded runs while
    the gate checks labeled cases from a case file with a function. It takes
    the same forms as Metacognition's `practice` and defaults to it.
    """

    cases: list[str] | None = None
    holdout: int = 0
    bar: float = ADEQUATE
    min_cases: int = 1
    practice: Any = None

    async def check(self, source: str, practice: Practice, cases: list[str]) -> GateResult:
        if len(cases) < self.min_cases:
            return GateResult(0, len(cases), False, [f"only {len(cases)} gate cases; {self.min_cases} needed"])
        accepted, failures = 0, []
        for case_id in cases:
            reward, reason, _ = await practice(source, case_id)
            accepted += int(reward >= 1.0)
            if reward < 1.0:
                failures.append(f"{case_id}: {reason}")
        total = len(cases)
        return GateResult(accepted, total, total > 0 and accepted / total >= self.bar - 1e-9, failures)
