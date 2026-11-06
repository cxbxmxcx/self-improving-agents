"""Learning problems from the domains' recordings: what A5's tournaments (L2) and design search (L3) measure.

A learning problem is what a learner faces. It has a procedure's contract, the judged runs to learn from, how
a candidate is tried while learning, and the held-out cases its function is scored on (does the function's
answer get accepted?). Every problem here comes from runs the agent made without a procedure:

  expenses        D1: the expense agent's runs on the 16 development trips; scored on the 16 validation trips
  refunds         D2: the returns desk's runs on the first 16 returns under the holiday policy; scored on the
                  last 16 returns of the holiday stream
  multi_attendee  chapter 6's calendar classes: the assistant's runs on the 16 development requests; scored on
  same_time       the 16 validation requests

Every problem learns under one regime (REGIMES): chapter 7's "low_writer" (runs at low effort, a writer at low
effort, 12 writer calls), or "weak_writer", the same with a writer that does no reasoning. A regime that leaves
the seeds room to improve is what a design search needs.
The runs replay from the recordings, and any the recordings lack are paid calls.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from helix.memory.learning_mechanisms import Episode
from helix.memory.mechanism_runtime import MechanismBudget

from metacog import RunConfig, Runner
from metacog.exceptions import AgentsException
from metacog.meta import LearningProblem, Verdict
from metacog.model_settings import ModelSettings, Reasoning

REPO = Path(__file__).resolve().parents[3]
CH06 = REPO / "chapters" / "ch06" / "assistant" / "recorded"
# Where chapter 6 recorded each calendar class's runs without memory (as chapter 7's regime sweep reads them).
CALENDAR_RUNS = {"multi_attendee": CH06 / "reasoning_floor.jsonl", "same_time": CH06 / "generalization.jsonl"}


@dataclass(frozen=True)
class Regime:
    """How much experience a learner gets and what it may spend (chapter 7's regimes)."""

    efforts: tuple[str, ...] = ("low",)
    tasks: int = 16
    writer_effort: str = "low"
    writer_calls: int = 12

    @property
    def budget(self) -> MechanismBudget:
        return MechanismBudget(writer_calls=self.writer_calls, seconds=1800)

    @property
    def writer_params(self) -> dict[str, Any]:
        return {"reasoning_effort": self.writer_effort}


LOW_WRITER = Regime()
# Chapter 7's regimes leave the seeds room to improve, or not. Under low_writer the best seeds learn D1 and D2
# perfectly, so a design search has nothing to beat there. weak_writer keeps the same runs and gives the writer
# no reasoning at all, the cheapest setting chapter 7 lets a design choose: can a learner make up for it?
REGIMES = {"low_writer": LOW_WRITER, "weak_writer": Regime(writer_effort="none"),
           "few_runs": Regime(tasks=8, writer_calls=8)}


async def runs_without_a_procedure(agent: Any, tasks: Sequence[dict[str, Any]], world: Callable[[Any], Any],
                                   check: Callable[..., Any], efforts: Sequence[str]) -> list[Episode]:
    """The agent's runs on `tasks` with no procedure, judged by the domain's checker. They run as `calibrate`
    runs its no-memory arm, so the runs the floors phases recorded replay."""
    base = RunConfig(max_tool_calls=24)
    episodes = []
    for task in tasks:
        for effort in efforts:
            settings = (base.model_settings or ModelSettings()).resolve(ModelSettings(reasoning=Reasoning(effort)))
            config = replace(base, model_settings=settings, metacognition_mode="off", case_id=task["id"],
                             case_data=task)
            try:
                result = await Runner.run(agent, task["text"], context=world(task), max_turns=12, run_config=config)
            except AgentsException as exc:
                episodes.append(Episode(task["id"], task["text"], [], False, f"the run failed: {exc}"))
                continue
            verdict = Verdict.coerce(check(result.context_wrapper, result))
            episodes.append(Episode.from_trajectory(task["id"], result.trajectory, bool(verdict and verdict.accepted),
                                                    verdict.reason if verdict else "no verdict"))
    return episodes


async def expenses_problem(regime: Regime = LOW_WRITER) -> LearningProblem:
    """D1: learn the reimbursable total from the development trips; scored on the validation trips."""
    from examples.expenses.agent import assistant, check, practice, reimbursable_total
    from examples.expenses.world import ExpenseWorld, task_set

    episodes = await runs_without_a_procedure(assistant(), task_set("dev")[: regime.tasks], ExpenseWorld, check,
                                              regime.efforts)
    held_out = [t["id"] for t in task_set("validation")]
    return LearningProblem("expenses", reimbursable_total.contract, episodes, practice, held_out, practice)


async def refunds_problem(regime: Regime = LOW_WRITER) -> LearningProblem:
    """D2: learn what a return gets once the holiday policy applies; scored on later holiday returns."""
    from examples.refunds.agent import check, practice, refund_decision_in_tool_terms, returns_desk
    from examples.refunds.world import ReturnsWorld, task_set

    stream = task_set("stream")
    episodes = await runs_without_a_procedure(returns_desk(), stream[: regime.tasks], ReturnsWorld, check,
                                              regime.efforts)
    held_out = [t["id"] for t in stream[-16:]]
    return LearningProblem("refunds", refund_decision_in_tool_terms.contract, episodes, practice, held_out, practice)


async def calendar_problem(task_class: str, regime: Regime = LOW_WRITER) -> LearningProblem:
    """A chapter 6 calendar class, its runs replayed from chapter 6's recordings as chapter 7 reads them."""
    from chapters.ch06.assistant.learning import (
        CLASSES,
        act,
        behavior_cases,
        gather_experience,
        recall_experience,
        remember,
    )
    from chapters.ch06.assistant.recording import Recording as BookRecording

    with BookRecording(CALENDAR_RUNS[task_class]).active():
        episodes = await gather_experience(efforts=regime.efforts, task_class=task_class)
    kept = {t["id"] for t in CLASSES[task_class].task_set("dev")[: regime.tasks]}
    tier = await remember([e for e in episodes if e.task_id in kept])
    recalled = await recall_experience(tier)
    tier.close()
    held_out = [t["id"] for t in behavior_cases("validation", task_class)]
    return LearningProblem(task_class, CLASSES[task_class].contract, recalled, act, held_out, act)
