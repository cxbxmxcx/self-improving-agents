"""What good is for a decision tier, and how far a candidate is from it.

A search can only improve what it can measure, and a score alone doesn't say what to change. So a candidate is tried on
labeled requests, each attempt's outcome sits some distance from good, and the candidate's gap is the mean of those
distances. The report here is that gap written out for the model that rewrites the candidate: how large it is, what it
is made of, which memories it sits in, and every request that was tried.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

_SAID = {"program": "a decision program did the work", "read_only": "handed off read-only",
         "llm": "the language model served it", "llm_wrong": "the language model got it wrong",
         "handoff": "handed off undone", "lost": "handed off read-only, so nothing could do it", "wrong": "wrong action",
         "refused": "the guard refused the program, so it never ran"}
_NOT_DONE = ("handoff", "lost", "llm_wrong", "refused")
_AT_GOOD = ("program", "read_only")
_LONGEST_REASON = 300


@dataclass(frozen=True)
class Good:
    """What good is, as how far each outcome of a request sits from it. A candidate at good has a gap of 0."""

    llm_right: float = 0.4  # right, but a language model did work a decision program could do
    not_done: float = 1.0  # handed off, lost, or the language model got it wrong
    wrong_action: float = 3.0  # an action that should not have been taken, or was taken wrongly

    def gap(self, outcome: str) -> float:
        if outcome in _AT_GOOD:
            return 0.0
        if outcome == "llm":
            return self.llm_right
        return self.wrong_action if outcome == "wrong" else self.not_done

    def describe(self) -> str:
        return ("Good is every request right, with no wrong action and no language model doing work a decision program "
                "could do. Each request is measured by its distance from good: 0 if a decision program did the work, or "
                f"a request that should not be acted on went read-only; {self.llm_right:g} if it was right but the "
                f"language model served it; {self.not_done:g} if it wasn't done; {self.wrong_action:g} for a wrong action.")


@dataclass
class Attempt:
    """One request tried on a candidate, and what happened."""

    case_id: str
    request: str
    should_act: bool
    outcome: str  # one of _SAID's keys
    reward: float
    reason: str
    memory: str | None = None  # the stored memory that served it, if one did
    answers: dict[str, Any] = field(default_factory=dict)  # what Jev said
    dollars: float = 0.0
    seconds: float = 0.0


def outcome_of(right: bool, *, program: str | None = None, should_act: bool = True) -> str:
    """The outcome of a request served online, as a gap counts it.

    `program` is what a decision program did with the request (`MetaReport.program`): "served" when it acted with no
    language model, "read_only" when it sent the request on with no way to act, and anything else when the language
    model did the work with its tools.
    """
    if program == "served":
        return "program" if right else "wrong"
    if program == "read_only":
        return "lost" if should_act else "read_only"
    return "llm" if right else "llm_wrong"


def gap_of(attempts: Sequence[Attempt], good: Good) -> float:
    """The candidate's distance from good: 0 is good."""
    return sum(good.gap(attempt.outcome) for attempt in attempts) / len(attempts) if attempts else 1.0


def _mean(attempts: Sequence[Attempt], name: str) -> float:
    return sum(getattr(attempt, name) for attempt in attempts) / len(attempts)


def report(attempts: Sequence[Attempt], good: Good) -> str:
    """The gap written out for whoever rewrites the candidate, ending with every request that was tried."""
    if not attempts:
        return "No requests were tried."
    if all(attempt.outcome == "refused" for attempt in attempts):
        return f"{good.describe()}\n\nGap from good: {gap_of(attempts, good):.2f}. No request ran: {attempts[0].reason}"
    count = len(attempts)
    lines = [good.describe(), "", f"Gap from good: {gap_of(attempts, good):.2f} over {count} requests (good is 0)."]
    for name, outcomes in (("wrong actions", ("wrong",)), ("not done", _NOT_DONE),
                           ("right, but served by the language model", ("llm",)), ("at good", _AT_GOOD)):
        group = [attempt for attempt in attempts if attempt.outcome in outcomes]
        lines.append(f"- {name}: {len(group)} requests, {sum(good.gap(a.outcome) for a in group) / count:.2f} of the gap")
    served = [attempt for attempt in attempts if attempt.outcome == "program"]
    modeled = [attempt for attempt in attempts if attempt.outcome in ("llm", "llm_wrong")]
    if modeled:
        cost = f"The language model costs ${_mean(modeled, 'dollars'):.5f} and {_mean(modeled, 'seconds'):.2f} s a request"
        lines.append(cost + (f"; a decision program costs ${_mean(served, 'dollars'):.5f} and "
                             f"{_mean(served, 'seconds'):.2f} s." if served else "."))
    by_memory: dict[str, list[Attempt]] = defaultdict(list)
    for attempt in attempts:
        if good.gap(attempt.outcome) > 0:
            by_memory[attempt.memory or "(no memory)"].append(attempt)
    if by_memory:
        lines += ["", "Where the gap is, largest first:"]
        for name, group in sorted(by_memory.items(), key=lambda item: -sum(good.gap(a.outcome) for a in item[1])):
            kinds = ", ".join(f"{number} {_SAID[outcome]}" for outcome, number in Counter(a.outcome for a in group).items())
            lines.append(f"- {name}: {sum(good.gap(a.outcome) for a in group) / count:.2f} of the gap ({kinds})")
    lines += ["", "Every request tried, furthest from good first:"]
    for attempt in sorted(attempts, key=lambda a: -good.gap(a.outcome)):
        memory = f", memory {attempt.memory}" if attempt.memory else ""
        cost = f"; ${attempt.dollars:.5f}, {attempt.seconds:.2f} s" if attempt.dollars or attempt.seconds else ""
        said = f"; Jev answered {json.dumps(attempt.answers)}" if attempt.answers else ""
        lines.append(f"[{good.gap(attempt.outcome):g}] should {'act' if attempt.should_act else 'not act'}: "
                     f"{attempt.request!r} -> {_SAID[attempt.outcome]}{memory}: "
                     f"{attempt.reason[:_LONGEST_REASON]}{cost}{said}")
    return "\n".join(lines)


__all__ = ["Attempt", "Good", "gap_of", "outcome_of", "report"]
