"""Chapter 6's signals: outcome, safety, cost, routing regret, behavior diff,
and the objective that folds them. Each returns a GapMeasurement (SPEC §3.1)."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import litellm

from helix.artifact import Subtype, genesis
from helix.memory.base import MemoryContext
from helix.signal import GapMeasurement, SignalKind, SignalThreshold
from helix.signals.behavior_diff import BehaviorDiffSignal
from helix.signals.cost import CostSignal, CostWeights, call_dollars, run_cost
from helix.signals.metacognition import OBJECTIVES, MetacognitionObjective
from helix.signals.outcome import OutcomeSignal
from helix.signals.routing import BranchResult, RoutingSignal
from helix.signals.safety import SafetySignal
from helix.trajectory import Outcome, StepKind, Trajectory
# The scaffold tests' fixtures, for the end-to-end run.
from tests.test_scaffold import FakeDecider, _assistant, _reply

PROMPT = genesis("p.sys", Subtype.PROMPT, "You help with a calendar and an inbox.")
# Dollars per million input and output tokens.
PRICES = {"base-model": (1.0, 5.0), "deliberate-model": (3.0, 15.0)}


def _run(*, route=None, calls=(), model_calls=(), llm_calls=(), decider_calls=(), output="Done.",
         outcome=Outcome.COMPLETED, task="Reply to Dana that Thursday works.") -> Trajectory:
    trajectory = Trajectory(task=task)
    for model, usage, seconds in model_calls:
        trajectory.append(StepKind.MODEL_CALL, {"response": {}, "model": model, "params": {}, "usage": usage,
                                                "seconds": seconds})
    for name, arguments, result in calls:
        trajectory.append(StepKind.TOOL_CALL, {"name": name, "arguments": arguments})
        trajectory.append(StepKind.TOOL_RESULT, {"result": result})
    meta = {"decisions": [], "decider_calls": list(decider_calls), "llm_calls": list(llm_calls), "tools": [],
            "errors": []}
    if route:
        meta["route"] = route
    trajectory.metadata["metacognition"] = meta
    trajectory.complete(output, outcome)
    return trajectory


def _costly_run(**kwargs) -> Trajectory:
    return _run(
        model_calls=[("base-model", {"prompt_tokens": 1000, "completion_tokens": 200, "total_tokens": 1200}, 1.5)],
        llm_calls=[
            {"purpose": "revise", "model": "deliberate-model",
             "usage": {"prompt_tokens": 500, "completion_tokens": 100}, "seconds": 2.0},
            {"purpose": "reflect", "model": "deliberate-model",
             "usage": {"prompt_tokens": 400, "completion_tokens": 50}, "seconds": 3.0},
        ],
        decider_calls=[
            {"hook": "route", "calls": 1, "input_tokens": 480, "dollars": 0.00002, "seconds": 0.11, "version": "jev"},
            {"hook": "reflect", "calls": 1, "input_tokens": 480, "dollars": 0.00002, "seconds": 0.12, "version": "jev"},
        ],
        **kwargs,
    )


# ---------------------------------------------------------------- outcome


async def test_outcome_signal_scores_the_environment_reward_and_stays_silent_without_one():
    accepted = await OutcomeSignal(lambda t: 1.0).measure(PROMPT, trajectory=_run())
    assert (accepted.score, accepted.raw_value, accepted.confidence) == (1.0, 1.0, 1.0)
    pending = await OutcomeSignal(lambda t: None).measure(PROMPT, trajectory=_run())
    assert (pending.score, pending.confidence) == (None, 0.0)

    async def reward(trajectory):
        return 0.0, {"feedback": "booked over Dana's class", "accepted": False}

    signal = OutcomeSignal(reward, kind=SignalKind.ENVIRONMENT_REWARD)
    rejected = await signal.measure(PROMPT, trajectory=_run())
    assert (rejected.score, rejected.feedback, rejected.metadata["accepted"]) == (0.0, "booked over Dana's class", False)
    assert signal.kind is SignalKind.ENVIRONMENT_REWARD


# ---------------------------------------------------------------- safety


async def test_safety_signal_triggers_on_a_consequential_call_executed_without_confirmation():
    run = _run(calls=[("read_calendar", {}, {"events": []}), ("send_reply", {"to": "dana"}, {"sent": True})])
    m = await SafetySignal({"send_reply", "move_event"}).measure(PROMPT, trajectory=run)
    assert (m.score, m.triggered, m.raw_value) == (0.0, True, 1.0)
    assert "send_reply ran without confirmation" in m.feedback


async def test_cancelled_failed_and_confirmed_calls_are_not_violations():
    run = _run(calls=[
        ("send_reply", {"to": "sam"}, {"cancelled": True, "reason": "needs the user's confirmation"}),
        ("move_event", {"event": "e1"}, {"error": "slot taken"}),
        ("send_reply", {"to": "dana"}, {"sent": True}),
    ])
    signal = SafetySignal({"send_reply", "move_event"}, confirmed=lambda t, tool, args: args.get("to") == "dana")
    m = await signal.measure(PROMPT, trajectory=run)
    assert (m.score, m.triggered) == (1.0, False)
    executed = [(c["tool"], c["executed"], c["confirmed"]) for c in m.metadata["calls"]]
    assert executed == [("send_reply", False, False), ("move_event", False, False), ("send_reply", True, True)]


async def test_safety_records_every_call_for_confirmation_precision_and_recall():
    async def consequential(trajectory, tool, arguments):
        return tool == "move_event" and arguments.get("owner") != "me"

    run = _run(calls=[
        ("move_event", {"event": "e1", "owner": "me"}, {"cancelled": True}),  # a needless confirmation
        ("move_event", {"event": "e2", "owner": "sam"}, {"cancelled": True}),  # a needed one
    ])
    m = await SafetySignal(consequential).measure(PROMPT, trajectory=run)
    flags = [(c["consequential"], c["cancelled"], c["violation"]) for c in m.metadata["calls"]]
    assert flags == [(False, True, False), (True, True, False)]
    assert not m.triggered


# ---------------------------------------------------------------- cost


async def test_run_cost_sums_agent_system2_and_decision_model_calls():
    cost = run_cost(_costly_run(), PRICES)
    # 0.002 agent + 0.003 revise + 0.00195 reflect + 0.00004 decision model
    assert abs(cost.dollars - 0.00699) < 1e-12
    # Reflection happens after the reply, so the user waits 1.5 + 2.0 + 0.11 seconds.
    assert abs(cost.seconds - 3.61) < 1e-9
    assert cost.tokens == 1200 + 600 + 450 + 480 + 480
    assert {k: v["calls"] for k, v in cost.by_source.items()} == {"agent": 1, "system2": 2, "decider": 2}
    assert cost.unpriced == []


async def test_unpriced_models_are_listed_not_silently_counted_as_free():
    usage = {"prompt_tokens": 1000, "completion_tokens": 100}
    known = sum(litellm.cost_per_token(model="gpt-4o-mini", prompt_tokens=1000, completion_tokens=100))
    assert call_dollars("gpt-4o-mini", usage) == known
    assert call_dollars("scripted-model", usage) is None
    run = _run(model_calls=[("scripted-model", usage, 0.0)])
    assert run_cost(run).unpriced == ["scripted-model"]
    m = await CostSignal("dollars", SignalThreshold(baseline=0.01)).measure(PROMPT, trajectory=run)
    assert (m.raw_value, m.confidence) == (0.0, 0.5)


async def test_cost_signal_normalizes_dollars_and_seconds_against_a_baseline():
    threshold = SignalThreshold(baseline=0.004, threshold=0.002, direction="minimize", normalizer="ratio")
    m = await CostSignal("dollars", threshold, prices=PRICES).measure(PROMPT, trajectory=_costly_run())
    assert abs(m.raw_value - 0.00699) < 1e-12
    assert abs(m.score - (0.004 / 0.00699) / 2) < 1e-9
    assert m.triggered  # 0.00299 over the baseline exceeds the 0.002 allowance
    seconds = await CostSignal("seconds", SignalThreshold(baseline=4.0)).measure(PROMPT, trajectory=_costly_run())
    assert abs(seconds.raw_value - 3.61) < 1e-9 and seconds.score > 0.5


# ---------------------------------------------------------------- routing


async def test_routing_signal_measures_regret_against_the_hindsight_best_branch():
    table = {"fast": BranchResult(0.0, 0.001, 1.0), "deliberate": BranchResult(1.0, 0.004, 4.0),
             "ask": BranchResult(0.5, 0.001, 1.0)}
    signal = RoutingSignal(lambda t: table)
    wrong = await signal.measure(PROMPT, trajectory=_run(route="fast"))
    assert (wrong.raw_value, wrong.score, wrong.triggered, wrong.metadata["best"]) == (1.0, 0.0, True, "deliberate")
    right = await signal.measure(PROMPT, trajectory=_run(route="deliberate"))
    assert (right.raw_value, right.score, right.triggered) == (0.0, 1.0, False)
    unknown = await RoutingSignal(lambda t: None).measure(PROMPT, trajectory=_run(route="fast"))
    assert (unknown.score, unknown.confidence) == (None, 0.0)


async def test_routing_regret_prices_over_thinking_only_when_cost_is_weighted():
    table = {"fast": BranchResult(1.0, 0.001, 1.0), "deliberate": BranchResult(1.0, 0.004, 4.0)}
    free = await RoutingSignal(lambda t: table).measure(PROMPT, trajectory=_run(route="deliberate"))
    assert (free.raw_value, free.triggered, free.metadata["best"]) == (0.0, False, "deliberate")
    priced = RoutingSignal(lambda t: table, weights=CostWeights(dollars=10.0, seconds=0.02))
    m = await priced.measure(PROMPT, trajectory=_run(route="deliberate"))
    # fast is worth 1 - 0.01 - 0.02 and deliberate 1 - 0.04 - 0.08
    assert abs(m.raw_value - 0.09) < 1e-9 and m.triggered and m.metadata["best"] == "fast"


# ---------------------------------------------------------------- behavior diff


async def test_behavior_diff_flags_a_lesson_that_changes_nothing():
    calls = [("move_event", {"event": "e1", "to": "Thu 09:00"}, {"moved": True})]
    before = _run(route="fast", calls=calls, output="Moved your meeting with Dana to Thursday at 9.")
    after = _run(route="fast", calls=calls, output="Done: your Dana meeting is now Thursday, 9 AM.")
    signal = BehaviorDiffSignal(reward_fn=lambda t: 0.0)
    m = await signal.compare(before, after)
    assert (m.score, m.triggered) == (0.0, True)
    assert m.metadata["components"]["reply"] > 0  # the wording moved, the behavior did not
    assert m.feedback == "no change in route, tool calls, or outcome"


async def test_behavior_diff_measures_route_tool_and_outcome_changes():
    before = _run(route="fast", calls=[("move_event", {"event": "e1", "to": "Thu 09:00"}, {"moved": True})])
    after = _run(route="deliberate", calls=[("read_calendar", {}, {"events": []}),
                                            ("move_event", {"event": "e1", "to": "Thu 14:00"}, {"moved": True})])
    rewards = {before.id: 0.0, after.id: 1.0}
    signal = BehaviorDiffSignal(reward_fn=lambda t: rewards[t.id], baseline=lambda t: before)
    m = await signal.measure(PROMPT, trajectory=after)
    components = m.metadata["components"]
    assert (components["route"], components["outcome"]) == (1.0, 1.0) and components["tools"] > 0.5
    assert not m.triggered and m.score > 0.8
    assert "route fast -> deliberate" in m.feedback and "outcome 0.0 -> 1.0" in m.feedback
    for passed in (before, {"baseline": before}):
        same = await BehaviorDiffSignal(reward_fn=lambda t: rewards[t.id]).measure(PROMPT, after, ground_truth=passed)
        assert same.score == m.score
    alone = await BehaviorDiffSignal().measure(PROMPT, trajectory=after)
    assert (alone.score, alone.confidence) == (None, 0.0)


# ---------------------------------------------------------------- the objective


def _objective(reward, **kwargs) -> MetacognitionObjective:
    return MetacognitionObjective(OutcomeSignal(lambda t: reward), SafetySignal({"send_reply"}),
                                  weights=CostWeights(dollars=10.0, seconds=0.01), prices=PRICES, **kwargs)


async def test_metacognition_objective_folds_outcome_safety_dollars_and_seconds():
    m = await _objective(1.0).measure(PROMPT, trajectory=_costly_run())
    assert abs(m.score - (1.0 - 0.0699 - 0.0361)) < 1e-9
    objectives = m.metadata["objectives"]
    assert (objectives["outcome"], objectives["safe"]) == (1.0, True)
    assert abs(objectives["dollars"] - 0.00699) < 1e-12 and abs(objectives["seconds"] - 3.61) < 1e-9
    assert set(OBJECTIVES) <= set(objectives)
    assert [c["signal_id"] for c in m.metadata["components"]][0].startswith("OutcomeSignal")


async def test_an_unsafe_run_scores_zero_whatever_its_outcome():
    run = _costly_run(calls=[("send_reply", {"to": "dana"}, {"sent": True})])
    m = await _objective(1.0).measure(PROMPT, trajectory=run)
    assert (m.score, m.triggered, m.metadata["objectives"]["safe"]) == (0.0, True, False)
    assert "UNSAFE" in m.feedback


async def test_the_objective_reports_costs_while_the_reward_is_pending():
    m = await _objective(None).measure(PROMPT, trajectory=_costly_run())
    assert (m.score, m.confidence) == (None, 0.0)
    assert abs(m.metadata["objectives"]["dollars"] - 0.00699) < 1e-12


async def test_signals_read_what_a_scaffolded_run_records():
    decider = FakeDecider({"multi_event_limit": 0.9, "reply_complete": 0.2})
    agent, _ = _assistant(decider)
    revision = _reply("Thursday works, and the room is booked.")
    usage = type("Usage", (), {"prompt_tokens": 100, "completion_tokens": 20})()
    with patch("helix.agent.helix_acompletion", AsyncMock(return_value=_reply("Thursday works.", usage=usage))), \
            patch("helix.scaffold.common.acompletion", AsyncMock(return_value=revision)):
        _, trajectory = await agent.run("Does Thursday work, and can you book the room?",
                                        context=MemoryContext(user_id="u1"))
    cost = run_cost(trajectory, PRICES)
    assert {k: v["calls"] for k, v in cost.by_source.items()} == {"agent": 1, "system2": 1, "decider": 2}
    assert abs(cost.by_source["agent"]["dollars"] - (100 * 3.0 + 20 * 15.0) / 1e6) < 1e-12
    # The revision reported no token counts, so it is flagged rather than counted as free.
    assert cost.unpriced == ["deliberate-model"]
    m = await _objective(1.0).measure(PROMPT, trajectory=trajectory)
    assert isinstance(m, GapMeasurement) and m.metadata["objectives"]["safe"]
    routing = await RoutingSignal(lambda t: {"fast": BranchResult(1.0), "deliberate": BranchResult(1.0)}).measure(
        PROMPT, trajectory=trajectory)
    assert routing.metadata["route"] == "deliberate"
