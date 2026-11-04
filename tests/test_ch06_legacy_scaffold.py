"""Chapter 6's assistant: the week, the checker, the request set, and the
scripted stand-in that lets every listing run without keys."""

from __future__ import annotations

from importlib import import_module

import pytest

from chapter_appendices.ch06_decision_scaffold_legacy.assistant.environment import (
    WATCHED_TOOLS,
    AssistantEnvironment,
    load_requests,
    preload_memory,
)
from chapter_appendices.ch06_decision_scaffold_legacy.assistant.scripted import model_mode
from helix.deciders.base import Answer, DecisionUsage
from helix.memory.base import MemoryContext
from helix.memory.episodic import EpisodicMemory
from helix.scaffold import record
from helix.scaffold.common import pinned
from helix.signals.safety import SafetySignal

listing = import_module("chapter_appendices.ch06_decision_scaffold_legacy.02_metacognitive_assistant")
REQUESTS = {r["id"]: r for r in load_requests()}


class CompleteReplies:
    """A stand-in decision model that finds every reply complete, so no run
    is revised; routes and confirmations are pinned by the tests."""

    name = "complete-replies"

    async def ask(self, state, questions):
        return (await self.ask_with_usage(state, questions))[0]

    async def ask_with_usage(self, state, questions):
        answers = {name: Answer("probability", 0.95) for name in questions}
        return answers, DecisionUsage(calls=1)

    def usage(self):
        return DecisionUsage()


def _tier() -> EpisodicMemory:
    from chapters.ch05.personal.encoder import embed
    return EpisodicMemory(":memory:", utility_weight=0.35, recency_weight=0.0, embedding_fn=embed)


async def _run(request, path, *, tool="allow", memory=True):
    env = AssistantEnvironment()
    tier = _tier()
    if memory:
        await preload_memory(tier, request)
    agent = listing.build_assistant(env, tier, CompleteReplies())
    agent.memory_hooks.read_only = True
    env.begin(request)
    with model_mode(False), pinned(route=path, tool=tool):
        answer, trajectory = await agent.run(
            request["text"], context=MemoryContext(user_id=request["user"], session_id=request["session"]))
    return env, trajectory


@pytest.mark.parametrize("request_id", sorted(REQUESTS))
async def test_every_request_scores_as_documented_on_each_path(request_id):
    request = REQUESTS[request_id]
    for path in ("fast", "deliberate", "ask"):
        env, trajectory = await _run(request, path)
        assert env.reward(trajectory) == request["expected"][path], (
            request_id, path, env.check(trajectory.final_output), trajectory.final_output)


def test_the_request_set_has_forty_training_and_twenty_held_out_requests_across_five_categories():
    splits = {}
    for r in REQUESTS.values():
        splits[r["split"]] = splits.get(r["split"], 0) + 1
    assert (splits["train"], splits["test"]) == (40, 20)
    assert {r["category"] for r in REQUESTS.values()} == {"fast", "deliberate", "ask", "confirm", "memory"}
    assert len({r["text"] for r in REQUESTS.values()}) == len(REQUESTS)


async def test_memory_requests_fail_without_the_memory_they_rely_on():
    for rid in ("M01", "M03", "M05", "M07", "M08", "M10"):
        env, trajectory = await _run(REQUESTS[rid], "fast", memory=False)
        assert env.reward(trajectory) == 0.0, rid


async def _safety(env, trajectory):
    return await SafetySignal(env.consequential, confirmed=env.confirmed).measure(None, trajectory=trajectory)


async def test_the_monitor_turns_unconfirmed_sends_and_moves_into_questions():
    for rid in ("C01", "C03"):
        env, acted = await _run(REQUESTS[rid], "fast", tool="allow")
        assert (await _safety(env, acted)).triggered, rid
        env, held = await _run(REQUESTS[rid], "fast", tool="confirm")
        assert not (await _safety(env, held)).triggered, rid
        assert env.reward(held) == 1.0 and "?" in held.final_output


async def test_a_request_that_is_the_users_go_ahead_skips_the_monitor():
    env, trajectory = await _run(REQUESTS["C09"], "fast", tool="confirm")
    assert record(trajectory)["tools"][0]["verdict"] == "confirmed"
    assert env.reward(trajectory) == 1.0 and env.state["sent"]
    assert env.confirmed(trajectory, "send_reply", {}) and not env.confirmed(trajectory, "move_event", {})


def test_only_sends_and_moves_of_other_peoples_events_are_consequential():
    env = AssistantEnvironment()
    env.begin(REQUESTS["F05"])
    assert WATCHED_TOOLS == {"send_reply", "move_event"}
    assert env.consequential(None, "send_reply", {"message_id": "M1"})
    assert env.consequential(None, "move_event", {"event_id": "E09"})  # Priya organized it
    assert not env.consequential(None, "move_event", {"event_id": "E11"})  # Maya's own lunch
    assert not env.consequential(None, "book_time", {})


def test_the_decision_model_sees_the_week_the_inbox_and_each_calls_target():
    env = AssistantEnvironment()
    env.begin(REQUESTS["A01"])
    route = env.decision_state("route", {"request": "Move my meeting with Sam to Thursday at 09:00."}, None)
    assert "Sam Lee (designer)" in route["people"] and "Sam Ortiz (engineering lead)" in route["people"]
    assert "Tue 11:00-11:30 1:1 with Sam Ortiz (organizer Maya Rivera)" in route["calendar"]
    assert "Mon/Tue/Wed/Thu/Fri 10:00-10:15 Team standup (organizer Priya Shah)" in route["calendar"]
    assert "M6 from Sam Lee: Lunch" in route["inbox"]
    tool = env.decision_state("tool", {"tool": "move_event", "arguments": {"event_id": "E07"}}, None)
    assert tool["event"]["organizer"] == "Sam Lee"


def test_the_checker_rejects_conflicts_missing_attendees_and_wrong_counts():
    env = AssistantEnvironment()
    env.begin(REQUESTS["D10"])  # 60 minutes with Leo on Thursday afternoon
    assert env.book_time("Review", "thu", "11:00", 60, ["Leo Chen"])["error"].startswith("Maya Rivera is busy")
    env.book_time("Review", "thu", "13:00", 60, ["Leo Chen"])
    assert env.check()["reason"] == "Leo Chen is busy then (Dentist)"
    env.reset()
    env.begin(REQUESTS["D10"])
    env.book_time("Review", "thu", "14:00", 60, [])
    assert env.check()["reason"] == "Leo Chen is not invited"
    with pytest.raises(ValueError, match="Sam Ortiz or Sam Lee"):
        env.resolve("Sam")


def test_snapshots_restore_the_week_and_forks_keep_shadow_runs_apart():
    env = AssistantEnvironment()
    env.begin(REQUESTS["F01"])
    saved = env.snapshot()
    shadow = env.fork()
    shadow.book_time("Writing", "mon", "09:00", 45)
    assert len(shadow.state["events"]) == len(env.state["events"]) + 1
    env.book_time("Writing", "mon", "09:00", 45)
    env.restore(saved)
    assert env.state == saved and env.actions == []


async def test_listing_one_replays_jevs_recorded_answers(capsys):
    await import_module("chapter_appendices.ch06_decision_scaffold_legacy.01_decision_model").main()
    printed = capsys.readouterr().out
    assert "Decision model: replay/jev-latest" in printed
    # The recalled lesson lowers the ambiguity judgment for the same request.
    assert "ambiguous_reference = 0.57" in printed and "ambiguous_reference = 0.24" in printed
    assert "consequence = sends" in printed


async def test_the_walkthrough_replays_its_recorded_run_without_keys(tmp_path):
    report = await listing.run_walkthrough(tmp_path / "run")
    rows = {r["request"]: r for r in report["requests"]}
    routes = [r["metacognition"]["route"] for r in report["requests"]]
    assert routes == ["fast", "deliberate", "ask", "fast", "fast", "fast", "fast"]
    assert [r["reward"] for r in report["requests"]] == [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0]
    assert rows["C01"]["metacognition"]["tools"][0]["verdict"] == "confirm"
    assert rows["D08"]["metacognition"]["revised"] is True
    # The rejection wrote a lesson, and the follow-up recalled it.
    assert "09:00 for writing" in rows["W01"]["metacognition"]["lesson"]["text"]
    assert "09:00" in rows["M01"]["answer"]


async def test_the_comparison_replays_its_recorded_run_without_keys(tmp_path):
    evals = import_module("chapter_appendices.ch06_decision_scaffold_legacy.03_metacognition_evals")
    report = await evals.evaluate("replay", tmp_path / "evals")
    summary = {name: c["summary"] for name, c in report["conditions"].items()}
    assert (summary["always fast"]["accepted"], summary["always fast"]["unsafe"]) == (12, 3)
    assert summary["always deliberate"]["accepted"] == 20
    seed, searched = summary["seed configuration"], summary["searched configuration"]
    # The search kept every request safe, accepted more, and spent less.
    assert searched["unsafe"] == 0 and searched["accepted"] > seed["accepted"]
    assert searched["dollars"] < seed["dollars"] < summary["always deliberate"]["dollars"]
    lessons = {row["request"]: row for row in report["behavior"]}
    assert lessons["M04 theater"]["flat"] and not lessons["M06"]["flat"]


async def test_the_search_replays_its_recorded_run_without_keys(tmp_path):
    search = import_module("chapter_appendices.ch06_decision_scaffold_legacy.04_decision_search")
    report = await search.search("replay", tmp_path / "search")
    offline = report["offline"]
    assert offline["final_score"] > offline["seed_score"]
    assert offline["thresholds"] == {"ask": 0.4, "deliberate": 0.4, "revise": 0.5}
    assert report["online"]["proposed"] == 4 and report["online"]["error"] is None
    from helix.archive import SQLiteArchive
    archive = SQLiteArchive(tmp_path / "search" / "archive.sqlite")
    composite = await archive.live_champion("metacognition")
    assert composite.version == 2 and composite.parent_id == ("metacognition", 1)
    history = await archive.promotion_history("metacognition")
    assert history[-1].approver == "person:reviewer"
