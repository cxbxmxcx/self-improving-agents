"""The metacognitive scaffold: a decision model directs hook wrappers that
route, confirm, track progress, revise, and reflect. Chapter 6, SPEC §6.2
and §16.2.4. None of it adds a branch to the agent loop."""

from __future__ import annotations

import copy
import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from helix.agent import Agent
from helix.archive import SQLiteArchive
from helix.artifact import Subtype, genesis
from helix.deciders.base import Answer, Choice, DecisionUsage, Score
from helix.hooks import HookPoint, RequestPatch
from helix.improvement import ImproverPolicy, OnlineImprover
from helix.memory.base import MemoryContext
from helix.memory.episodic import EpisodicMemory
from helix.observability.bus import EventBus
from helix.scaffold import (
    ASK_NOTE,
    FINISH_NOTE,
    PLANNER_HEADING,
    MetacognitiveScaffold,
    Rule,
    check_config,
    config_artifact,
    decider_usage,
    thresholds_artifact,
)
from helix.tools import tool
from helix.trajectory import StepKind

CONFIG = {
    "questions": {
        "multi_event_limit": {"hook": "route", "type": "probability",
                              "instructions": "The request sets a limit that spans several events or people."},
        "ambiguous_reference": {"hook": "route", "type": "probability",
                                "instructions": "The request names a person or event that matches more than one item."},
        "consequence": {"hook": "tool", "type": "choice",
                        "instructions": "What does this tool call change for people other than the user?",
                        "criteria": {"none": "Only the user's own calendar or drafts",
                                     "other_event": "Another person's event", "sends": "A message sent for the user"}},
        "progress": {"hook": "progress", "type": "score", "instructions": "How much of the request is done?",
                     "levels": ["not started", "partly done", "done"]},
        "reply_complete": {"hook": "output", "type": "probability",
                           "instructions": "The reply answers every question the message asks."},
        "lesson_worth": {"hook": "reflect", "type": "probability",
                         "instructions": "The failure points to a lesson that would help similar requests."},
    },
    "rules": [
        {"question": "ambiguous_reference", "op": ">=", "threshold": "ask", "action": "ask"},
        {"question": "multi_event_limit", "op": ">=", "threshold": "deliberate", "action": "deliberate"},
        {"question": "consequence", "op": "not_in", "value": ["none"], "action": "confirm"},
        {"question": "progress", "op": ">=", "threshold": 1.0, "action": "finish"},
        {"question": "reply_complete", "op": "<", "threshold": "revise", "action": "revise"},
        {"question": "lesson_worth", "op": ">=", "threshold": 0.5, "action": "reflect"},
    ],
}
THRESHOLDS = {"ask": 0.5, "deliberate": 0.5, "revise": 0.5}
# A calm run: no route rule fires, calls are harmless, replies are complete.
CALM = {"multi_event_limit": 0.1, "ambiguous_reference": 0.1, "consequence": "none",
        "progress": "partly done", "reply_complete": 0.9, "lesson_worth": 0.9}


class FakeDecider:
    """Answers each question by name from a table, and remembers every call."""

    name = "fake"

    def __init__(self, values: dict[str, Any] | None = None, *, fail: bool = False) -> None:
        self.values = {**CALM, **(values or {})}
        self.fail = fail
        self.calls: list[tuple[Any, list[str]]] = []

    async def ask(self, state, questions):
        return (await self.ask_with_usage(state, questions))[0]

    async def ask_with_usage(self, state, questions):
        self.calls.append((state, list(questions)))
        if self.fail:
            raise RuntimeError("decision model unavailable")
        answers: dict[str, Answer] = {}
        for name, question in questions.items():
            value = self.values.get(name)
            if isinstance(question, Choice):
                rest = 0.1 / (len(question.criteria) - 1)
                answers[name] = Answer("choice", value, 0.9,
                                       {o: 0.9 if o == value else rest for o in question.criteria})
            elif isinstance(question, Score):
                index = question.levels.index(value)
                answers[name] = Answer("score", float(index), 0.8, {str(index): 0.8},
                                       index / (len(question.levels) - 1))
            else:
                answers[name] = Answer("probability", float(value))
        return answers, DecisionUsage(calls=1, input_tokens=100, dollars=0.00001, seconds=0.05, version="fake-1")

    def usage(self) -> DecisionUsage:
        return DecisionUsage()

    def hooks_asked(self) -> list[list[str]]:
        return [names for _, names in self.calls]


def _reply(content: str = "", *, call: str | None = None, arguments: str = "{}", usage=None):
    calls = None if call is None else [SimpleNamespace(
        id="call-1", type="function", function=SimpleNamespace(name=call, arguments=arguments))]
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=calls))],
                           usage=usage)


def _artifacts(config: dict | None = None, thresholds: dict | None = None):
    return {
        "config": config_artifact("meta.config", config or CONFIG),
        "thresholds": thresholds_artifact("meta.thresholds", thresholds or THRESHOLDS),
        "planner": genesis("meta.planner", Subtype.PLANNER, "List every constraint, check each, then act."),
        "reflector": genesis("meta.reflector", Subtype.REFLECTOR, "Write one lesson that would have avoided this."),
    }


def _scaffold(decider: FakeDecider, **kwargs) -> MetacognitiveScaffold:
    options = {**_artifacts(), "fast_model": "fast-model", "deliberate_model": "deliberate-model",
               "deliberate_params": {"reasoning_effort": "high"}, "watch_tools": {"send_reply"}}
    options.update(kwargs)
    return MetacognitiveScaffold(decider, **options)


class Completion:
    """A scripted model that snapshots each request: the loop hands every call
    the same live message list, so a mock's recorded arguments would show
    later edits."""

    def __init__(self, *replies) -> None:
        self.replies = list(replies)
        self.requests: list[dict] = []

    async def __call__(self, **kwargs):
        self.requests.append(copy.deepcopy(kwargs))
        return self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]


def _assistant(decider: FakeDecider, *, memory=None, bus=None, **kwargs):
    sent: list[dict] = []

    @tool(description="Send the drafted reply for the user.")
    async def send_reply(to: str) -> dict:
        sent.append({"to": to})
        return {"sent": True}

    @tool(description="Read the user's calendar.")
    async def read_calendar() -> dict:
        return {"events": []}

    agent = Agent(system_prompt=genesis("p.sys", Subtype.PROMPT, "You help with a calendar and an inbox."),
                  tools=[send_reply, read_calendar], model="base-model", bus=bus or EventBus(),
                  memory_tiers=memory, scaffold=_scaffold(decider, **kwargs))
    return agent, sent


def _system(call) -> str:
    content = call.kwargs["messages"][0]["content"]
    return content[0]["text"] if isinstance(content, list) else content


def _meta(trajectory) -> dict:
    return trajectory.metadata["metacognition"]


# ---------------------------------------------------------------- the router


async def test_router_takes_the_deliberate_path_and_appends_the_planner():
    decider = FakeDecider({"multi_event_limit": 0.9})
    agent, _ = _assistant(decider)
    completion = AsyncMock(return_value=_reply("Three meetings fit before Friday."))
    with patch("helix.agent.helix_acompletion", completion):
        _, trajectory = await agent.run("Fit three 30-minute meetings with Sam before Friday, under 2 hours total.")
    kwargs = completion.await_args.kwargs
    assert (kwargs["model"], kwargs["reasoning_effort"]) == ("deliberate-model", "high")
    assert PLANNER_HEADING in _system(completion.await_args)
    assert "List every constraint" in _system(completion.await_args)
    assert _meta(trajectory)["route"] == "deliberate"
    used = {ref for refs in trajectory.artifacts_used.values() for ref in refs}
    assert {("meta.config", 1), ("meta.thresholds", 1), ("meta.planner", 1)} <= used


async def test_router_patch_switches_the_model_for_the_rest_of_the_run():
    decider = FakeDecider({"multi_event_limit": 0.9})
    agent, _ = _assistant(decider)
    completion = AsyncMock(side_effect=[_reply(call="read_calendar"), _reply("Booked all three.")])
    with patch("helix.agent.helix_acompletion", completion):
        await agent.run("Fit three meetings with Sam before Friday.")
    assert [c.kwargs["model"] for c in completion.await_args_list] == ["deliberate-model", "deliberate-model"]
    assert decider.hooks_asked().count(["multi_event_limit", "ambiguous_reference"]) == 1


async def test_asking_beats_deliberating_and_turns_tool_calls_off():
    decider = FakeDecider({"multi_event_limit": 0.9, "ambiguous_reference": 0.8})
    agent, _ = _assistant(decider)
    completion = AsyncMock(return_value=_reply("Which Sam do you mean, Sam Ortiz or Sam Lee?"))
    with patch("helix.agent.helix_acompletion", completion):
        output, trajectory = await agent.run("Move my meeting with Sam to Thursday.")
    kwargs = completion.await_args.kwargs
    assert (kwargs["model"], kwargs["tool_choice"]) == ("fast-model", "none")
    assert ASK_NOTE.splitlines()[0] in _system(completion.await_args)
    assert _meta(trajectory)["route"] == "ask"
    # A question back is not checked for completeness.
    assert ["reply_complete"] not in decider.hooks_asked()
    assert output.startswith("Which Sam")


async def test_the_fast_path_is_taken_when_no_route_rule_fires():
    decider = FakeDecider()
    agent, _ = _assistant(decider)
    completion = AsyncMock(return_value=_reply("Booked 30 minutes at 10:00."))
    with patch("helix.agent.helix_acompletion", completion):
        _, trajectory = await agent.run("Book 30 minutes for writing tomorrow morning.")
    assert completion.await_args.kwargs["model"] == "fast-model"
    assert "reasoning_effort" not in completion.await_args.kwargs
    assert _meta(trajectory)["route"] == "fast"


# ---------------------------------------------------------------- the monitor


async def test_cancelled_tool_call_returns_the_monitors_reason():
    decider = FakeDecider({"consequence": "sends"})
    agent, sent = _assistant(decider)
    completion = AsyncMock(side_effect=[_reply(call="send_reply", arguments='{"to": "dana"}'),
                                        _reply("Shall I send it to Dana?")])
    with patch("helix.agent.helix_acompletion", completion):
        _, trajectory = await agent.run("Reply to Dana that Thursday works.")
    assert sent == []
    result = next(s.payload["result"] for s in trajectory.steps if s.kind == StepKind.TOOL_RESULT)
    assert result == {"cancelled": True,
                      "reason": "needs the user's confirmation before send_reply: A message sent for the user"}
    assert _meta(trajectory)["tools"][0]["verdict"] == "confirm"


async def test_a_confirmed_call_runs_without_asking_the_decision_model():
    decider = FakeDecider({"consequence": "sends"})
    agent, sent = _assistant(decider, confirmed=lambda trajectory, tool, args: tool == "send_reply")
    completion = AsyncMock(side_effect=[_reply(call="send_reply", arguments='{"to": "dana"}'), _reply("Sent.")])
    with patch("helix.agent.helix_acompletion", completion):
        _, trajectory = await agent.run("Yes, send it to Dana.")
    assert sent == [{"to": "dana"}]
    assert ["consequence"] not in decider.hooks_asked()
    assert _meta(trajectory)["tools"][0]["verdict"] == "confirmed"


async def test_unwatched_tools_are_never_checked():
    decider = FakeDecider({"consequence": "sends"})
    agent, _ = _assistant(decider)
    completion = AsyncMock(side_effect=[_reply(call="read_calendar"), _reply("You are free at 10.")])
    with patch("helix.agent.helix_acompletion", completion):
        _, trajectory = await agent.run("When am I free tomorrow?")
    assert ["consequence"] not in decider.hooks_asked()
    assert _meta(trajectory)["tools"] == []


async def test_a_retried_call_reuses_the_monitors_verdict():
    decider = FakeDecider({"consequence": "sends"})
    agent, sent = _assistant(decider)
    retry = _reply(call="send_reply", arguments='{"to": "dana"}')
    completion = AsyncMock(side_effect=[retry, retry, _reply("I need your confirmation to send it.")])
    with patch("helix.agent.helix_acompletion", completion):
        _, trajectory = await agent.run("Reply to Dana that Thursday works.")
    assert sent == []
    assert decider.hooks_asked().count(["consequence"]) == 1
    results = [s.payload["result"] for s in trajectory.steps if s.kind == StepKind.TOOL_RESULT]
    assert len(results) == 2 and all(r["cancelled"] for r in results)


async def test_confirmation_rule_ignores_the_online_thresholds():
    for extreme in (0.0, 1.0):
        decider = FakeDecider({"consequence": "sends"})
        scaffold = _scaffold(decider, thresholds=thresholds_artifact(
            "meta.thresholds", {name: extreme for name in THRESHOLDS}))
        agent = Agent(system_prompt=genesis("p.sys", Subtype.PROMPT, "You help."),
                      tools=_assistant(decider)[0].tools.values(), model="base-model", bus=EventBus(),
                      scaffold=scaffold)
        completion = AsyncMock(side_effect=[_reply(call="send_reply", arguments='{"to": "dana"}'), _reply("Confirm?")])
        # At 1.0 the revise rule fires too, so the System 2 call needs a stand-in.
        with patch("helix.agent.helix_acompletion", completion), \
                patch("helix.scaffold.common.acompletion", AsyncMock(return_value=_reply("Shall I send it?"))):
            _, trajectory = await agent.run("Reply to Dana.")
        assert _meta(trajectory)["tools"][0]["verdict"] == "confirm"
    # A confirmation rule may not even name an online threshold.
    unsafe = json.loads(json.dumps(CONFIG))
    unsafe["questions"]["sends_mail"] = {"hook": "tool", "type": "probability", "instructions": "The call sends mail."}
    unsafe["rules"].append({"question": "sends_mail", "op": ">=", "threshold": "ask", "action": "confirm"})
    assert "may not read the online thresholds" in check_config(unsafe)


async def test_monitor_revises_an_incomplete_reply_once_with_a_system2_call():
    decider = FakeDecider({"reply_complete": 0.2})
    agent, _ = _assistant(decider)
    revision = _reply("Thursday works, and the room is booked too.", usage=SimpleNamespace(prompt_tokens=80, completion_tokens=12))
    with patch("helix.agent.helix_acompletion", AsyncMock(return_value=_reply("Thursday works."))), \
            patch("helix.scaffold.common.acompletion", AsyncMock(return_value=revision)) as system2:
        output, trajectory = await agent.run("Does Thursday work, and can you book the room?")
    assert output == "Thursday works, and the room is booked too."
    assert trajectory.final_output == output
    assert system2.await_args.kwargs["model"] == "deliberate-model"
    meta = _meta(trajectory)
    assert meta["revised"] is True
    assert meta["llm_calls"] == [{"purpose": "revise", "model": "deliberate-model",
                                  "usage": {"prompt_tokens": 80, "completion_tokens": 12},
                                  "seconds": meta["llm_calls"][0]["seconds"]}]


async def test_a_complete_reply_goes_out_unchanged():
    decider = FakeDecider()
    agent, _ = _assistant(decider)
    with patch("helix.agent.helix_acompletion", AsyncMock(return_value=_reply("Thursday works."))), \
            patch("helix.scaffold.common.acompletion", AsyncMock()) as system2:
        output, trajectory = await agent.run("Does Thursday work?")
    assert output == "Thursday works."
    system2.assert_not_awaited()
    assert _meta(trajectory)["revised"] is False


# ---------------------------------------------------------------- task state


async def test_task_state_adds_a_finish_note_when_progress_is_done():
    decider = FakeDecider({"multi_event_limit": 0.9, "progress": "done"})
    agent, _ = _assistant(decider)
    completion = Completion(_reply(call="read_calendar"), _reply("All three are booked."))
    with patch("helix.agent.helix_acompletion", completion):
        _, trajectory = await agent.run("Book three meetings with Sam before Friday.")
    first, second = (SimpleNamespace(kwargs=r) for r in completion.requests)
    assert "tool_choice" not in first.kwargs and FINISH_NOTE.splitlines()[0] not in _system(first)
    # The finish patch keeps the route's model and adds only the switch.
    assert (second.kwargs["model"], second.kwargs["tool_choice"]) == ("deliberate-model", "none")
    assert FINISH_NOTE.splitlines()[0] in _system(second)
    assert (_meta(trajectory)["progress"], _meta(trajectory)["finish"]) == (1.0, True)


async def test_task_state_keeps_going_while_progress_is_partial():
    decider = FakeDecider({"progress": "partly done"})
    agent, _ = _assistant(decider)
    completion = AsyncMock(side_effect=[_reply(call="read_calendar"), _reply("Done.")])
    with patch("helix.agent.helix_acompletion", completion):
        _, trajectory = await agent.run("Book a meeting with Sam.")
    assert "tool_choice" not in completion.await_args_list[1].kwargs
    assert (_meta(trajectory)["progress"], _meta(trajectory)["finish"]) == (0.5, False)


# ---------------------------------------------------------------- the reflector


def _memory() -> dict:
    return {"episodic": EpisodicMemory(":memory:", embedding_fn=lambda texts: [[1.0, 0.0] for _ in texts])}


async def _run_with_reward(reward, *, values=None, memory=None, read_only=False):
    decider = FakeDecider(values)
    memory = memory or _memory()
    agent, _ = _assistant(decider, memory=memory, reward_fn=lambda trajectory: reward)
    if read_only:
        agent.memory_hooks.read_only = True
    lesson = _reply("When Dana asks for Thursday, book the afternoon; she teaches in the morning.")
    with patch("helix.agent.helix_acompletion", AsyncMock(return_value=_reply("Booked Thursday at 9."))), \
            patch("helix.scaffold.common.acompletion", AsyncMock(return_value=lesson)):
        _, trajectory = await agent.run("Book an hour with Dana on Thursday.",
                                        context=MemoryContext(user_id="u1", session_id="s1"))
    entries = await memory["episodic"].list_all()
    return trajectory, [e for e in entries if e.content.get("lesson")], decider


async def test_reflector_writes_a_lesson_only_after_a_rejected_outcome():
    trajectory, lessons, _ = await _run_with_reward(0.0)
    assert len(lessons) == 1
    lesson = lessons[0]
    assert lesson.content["lesson"].startswith("When Dana asks for Thursday")
    assert (lesson.metadata["kind"], lesson.metadata["created_by"]) == ("lesson", "reflector")
    assert lesson.scope_key.to_str() == "user:u1"
    assert _meta(trajectory)["lesson"]["entry_id"] == lesson.id
    assert ("meta.reflector", 1) in {ref for refs in trajectory.artifacts_used.values() for ref in refs}
    for reward in (1.0, None):
        _, lessons, _ = await _run_with_reward(reward)
        assert lessons == []


async def test_the_decision_model_can_judge_a_failure_not_worth_a_lesson():
    trajectory, lessons, decider = await _run_with_reward(0.0, values={"lesson_worth": 0.1})
    assert lessons == []
    assert ["lesson_worth"] in decider.hooks_asked()
    assert "lesson" not in _meta(trajectory)


async def test_a_new_lesson_keeps_its_initial_value_until_it_is_recalled():
    memory = _memory()
    _, lessons, _ = await _run_with_reward(0.0, memory=memory)
    assert lessons[0].score == 0.5  # the same starting value chapter 5 gives an episode
    decider = FakeDecider()
    agent, _ = _assistant(decider, memory=memory)
    completion = AsyncMock(return_value=_reply("Booked Thursday at 2."))
    with patch("helix.agent.helix_acompletion", completion):
        _, trajectory = await agent.run("Book an hour with Dana on Thursday.",
                                        context=MemoryContext(user_id="u1", session_id="s2"))
    assert "Lesson: When Dana asks for Thursday" in _system(completion.await_args)
    recalled = {r["entry_id"] for r in trajectory.metadata["memory_recalled"]}
    assert lessons[0].id in recalled
    # The router saw the recalled lesson too, so it can change the route.
    route_state = decider.calls[0][0]
    assert "When Dana asks for Thursday" in route_state["memory"]


async def test_read_only_runs_write_no_lessons():
    _, lessons, _ = await _run_with_reward(0.0, read_only=True)
    assert lessons == []


# ---------------------------------------------------------------- records and failures


async def test_decisions_are_recorded_per_question_with_usage_and_published():
    bus = EventBus()
    seen: list = []
    bus.subscribe("hook_fired", seen.append)
    decider = FakeDecider({"multi_event_limit": 0.9})
    agent, _ = _assistant(decider, bus=bus)
    with patch("helix.agent.helix_acompletion", AsyncMock(return_value=_reply("Done."))):
        _, trajectory = await agent.run("Fit three meetings before Friday.")
    meta = _meta(trajectory)
    rows = {(d["hook"], d["question"]): d for d in meta["decisions"]}
    assert rows[("route", "multi_event_limit")]["value"] == 0.9
    assert rows[("route", "multi_event_limit")]["actions"] == ["deliberate"]
    assert rows[("route", "ambiguous_reference")]["actions"] == []
    assert [c["hook"] for c in meta["decider_calls"]] == ["route", "output"]
    usage = decider_usage(trajectory)
    assert (usage.calls, usage.input_tokens, usage.version) == (2, 200, "fake-1")
    assert [(e.point, e.wrapper, e.verdict) for e in seen] == [
        ("pre_model", "router", "deliberate"), ("pre_output", "monitor", "keep")]
    json.dumps(trajectory.to_dict())  # the record must survive the archive


async def test_a_failed_decision_raises_unless_the_scaffold_is_told_to_play_safe():
    agent, _ = _assistant(FakeDecider(fail=True))
    with patch("helix.agent.helix_acompletion", AsyncMock(return_value=_reply("Done."))):
        with pytest.raises(RuntimeError, match="decision model unavailable"):
            await agent.run("Book a meeting.")
    agent, sent = _assistant(FakeDecider(fail=True), on_error="safe")
    completion = AsyncMock(side_effect=[_reply(call="send_reply", arguments='{"to": "dana"}'), _reply("Confirm?")])
    with patch("helix.agent.helix_acompletion", completion):
        _, trajectory = await agent.run("Reply to Dana.")
    meta = _meta(trajectory)
    # With no answers, only the safety rule fires: the send waits for the user.
    assert (meta["route"], meta["tools"][0]["verdict"], sent) == ("fast", "confirm", [])
    # The send was held, so its confirmation question is not checked for completeness.
    assert {e["hook"] for e in meta["errors"]} == {"route", "tool", "progress"}


async def test_state_fn_adds_application_state_to_every_decision():
    seen: list[str] = []

    async def state_fn(hook, state, trajectory):
        seen.append(hook)
        return {**state, "calendar": ["Sam Ortiz 1:1", "Sam Lee review"]}

    decider = FakeDecider()
    agent, _ = _assistant(decider, state_fn=state_fn)
    with patch("helix.agent.helix_acompletion", AsyncMock(return_value=_reply("Done."))):
        await agent.run("Move my meeting with Sam.")
    assert seen == ["route", "output"]
    assert all(state["calendar"] for state, _ in decider.calls)


# ---------------------------------------------------------------- artifacts, clones, and improvers


def test_the_scaffold_declares_a_metacognition_composite_without_the_thresholds():
    agent, _ = _assistant(FakeDecider())
    composite = agent.find_artifact("metacognition")
    assert (composite.subtype, composite.layer) == (Subtype.METACOGNITION, 3)
    assert agent.composite_constituent_ids() == {"meta.config", "meta.planner", "meta.reflector"}
    assert agent.find_artifact("meta.thresholds").layer == 1


async def test_clone_from_with_artifacts_keeps_the_scaffold_and_swaps_its_artifacts():
    decider = FakeDecider({"multi_event_limit": 0.7})
    agent, _ = _assistant(decider)
    lower = agent.scaffold.thresholds.mutate(json.dumps({**THRESHOLDS, "deliberate": 0.8}), created_by="online")
    clone = agent.with_artifacts({lower.id: lower})
    assert clone.scaffold is not agent.scaffold
    assert clone.scaffold.threshold_values["deliberate"] == 0.8
    assert agent.scaffold.threshold_values["deliberate"] == 0.5
    assert clone.find_artifact("metacognition") is agent.find_artifact("metacognition")
    routes = []
    for candidate in (agent, clone):
        with patch("helix.agent.helix_acompletion", AsyncMock(return_value=_reply("Done."))):
            _, trajectory = await candidate.run("Fit three meetings before Friday.")
        routes.append(_meta(trajectory)["route"])
    assert routes == ["deliberate", "fast"]


def test_swapping_a_constituent_rebinds_the_composite_as_a_derived_version():
    agent, _ = _assistant(FakeDecider())
    config_v2 = agent.scaffold.config.mutate(agent.scaffold.config.content, created_by="gepa")
    clone = agent.with_artifacts({config_v2.id: config_v2})
    composite = clone.find_artifact("metacognition")
    assert (composite.version, composite.parent_id, composite.created_by) == (2, ("metacognition", 1), "override")
    assert ("meta.config", 2) in composite.constituent_refs
    assert agent.find_artifact("metacognition").version == 1


def test_a_composite_override_brings_its_constituents():
    agent, _ = _assistant(FakeDecider())
    scaffold = agent.scaffold
    planner_v2 = scaffold.planner.mutate("Check each limit across all events first.", created_by="composed")
    derived = scaffold.with_overrides({planner_v2.id: planner_v2}).composite
    clone = agent.with_artifacts({derived.id: derived, planner_v2.id: planner_v2})
    assert clone.scaffold.planner is planner_v2
    with pytest.raises(LookupError, match="meta.planner:v2"):
        agent.with_artifacts({derived.id: derived})


def test_online_improver_refuses_the_metacognition_composite_and_its_members():
    agent, _ = _assistant(FakeDecider())
    for target, reason in (("metacognition", "L3"), ("meta.config", "constituent")):
        with pytest.raises(ValueError, match=reason):
            OnlineImprover(agent=agent, target_artifact_id=target, signal=object(), search=object(),
                           archive=SQLiteArchive(":memory:"), policy=ImproverPolicy())


def test_online_improver_accepts_the_standalone_thresholds_artifact():
    agent, _ = _assistant(FakeDecider())
    improver = OnlineImprover(agent=agent, target_artifact_id="meta.thresholds", signal=object(),
                              search=object(), archive=SQLiteArchive(":memory:"), policy=ImproverPolicy())
    assert improver.target_artifact_id == "meta.thresholds"


# ---------------------------------------------------------------- the configuration


def test_check_config_explains_why_a_configuration_is_unusable():
    def broken(mutate) -> dict:
        config = json.loads(json.dumps(CONFIG))
        mutate(config)
        return config

    assert check_config(json.dumps(CONFIG), THRESHOLDS) is None
    assert check_config("{not json").startswith("not valid JSON")
    cases = {
        "unknown question": lambda c: c["rules"].append({"question": "nope", "op": ">=", "threshold": 0.5, "action": "ask"}),
        "not available at the tool hook": lambda c: c["rules"].append(
            {"question": "consequence", "op": "in", "value": ["sends"], "action": "deliberate"}),
        "needs a choice question": lambda c: c["rules"].append(
            {"question": "reply_complete", "op": "in", "value": ["yes"], "action": "revise"}),
        "at least two distinct levels": lambda c: c["questions"]["progress"].update(levels=["done"]),
        "unknown options": lambda c: c["rules"].append(
            {"question": "consequence", "op": "in", "value": ["deletes"], "action": "confirm"}),
    }
    for reason, mutate in cases.items():
        assert reason in check_config(broken(mutate)), reason
    renamed = broken(lambda c: c["rules"][0].update(threshold="ask_more"))
    assert "do not exist: ['ask_more']" in check_config(renamed, THRESHOLDS)


def test_rules_read_score_positions_and_choice_option_probabilities():
    score = Answer("score", 2.0, 0.8, {"2": 0.8}, 1.0)
    assert Rule(question="progress", op=">=", threshold=1.0, action="finish").fires(score, {})
    choice = Answer("choice", "none", 0.7, {"none": 0.7, "sends": 0.3})
    unsure = Rule(question="consequence", op="<", threshold=0.9, option="none", action="confirm")
    assert unsure.fires(choice, {})
    assert not Rule(question="consequence", op="not_in", value=["none"], action="confirm").fires(choice, {})
    # An unreadable answer fires a safety rule and nothing else.
    assert unsure.fires(None, {})
    assert not Rule(question="progress", op=">=", threshold=1.0, action="finish").fires(None, {})


def test_the_scaffold_rejects_artifacts_in_the_wrong_slot_or_out_of_step():
    arts = _artifacts()
    with pytest.raises(ValueError, match="text:planner"):
        MetacognitiveScaffold(FakeDecider(), arts["config"], arts["thresholds"], planner=arts["reflector"])
    with pytest.raises(ValueError, match="does not hold"):
        MetacognitiveScaffold(FakeDecider(), arts["config"], thresholds_artifact("t", {"ask": 0.5}))
    same_id = thresholds_artifact("meta.config", THRESHOLDS)
    with pytest.raises(ValueError, match="their own artifact"):
        MetacognitiveScaffold(FakeDecider(), arts["config"], same_id)


async def test_a_hook_point_patch_from_the_router_is_a_request_patch():
    agent, _ = _assistant(FakeDecider())
    results = await agent.hooks.fire(HookPoint.PRE_MODEL, messages=[{"role": "system", "content": "s"},
                                                                    {"role": "user", "content": "u"}],
                                     trajectory=SimpleNamespace(metadata={}, steps=[], artifacts_used={},
                                                                record_artifact=lambda *a: None, task="u", id="t"))
    patches = [r for r in results if isinstance(r, RequestPatch)]
    assert [p.source for p in patches] == ["router:fast"]


async def test_a_reply_asking_to_confirm_a_held_call_is_not_revised():
    decider = FakeDecider({"consequence": "sends", "reply_complete": 0.1})
    agent, _ = _assistant(decider)
    completion = AsyncMock(side_effect=[_reply(call="send_reply", arguments='{"to": "dana"}'),
                                        _reply("Shall I send it to Dana?")])
    with patch("helix.agent.helix_acompletion", completion), \
            patch("helix.scaffold.common.acompletion", AsyncMock()) as system2:
        output, trajectory = await agent.run("Reply to Dana that Thursday works.")
    assert output == "Shall I send it to Dana?"
    system2.assert_not_awaited()
    assert ["reply_complete"] not in decider.hooks_asked()
