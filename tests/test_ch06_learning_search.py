"""Chapter 6, stages 2 and 3: experience, learning mechanisms, behavior match, and the search."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from chapters.ch06.assistant.generator import task_set
from chapters.ch06.assistant.learning import (CONTRACT, act, bootstrapped_source, match_signal, recall_experience,
                                              remember, run_on_case)
from helix.artifact import ArtifactKind, genesis
from helix.memory.learning_mechanisms import (ALPHA, CONTROL, INITIAL_VALUE, MECHANISMS, Episode, Reflection,
                                              ValueLearning, Writer, extract_source)
from helix.search.candidates import CandidateSearch
from helix.signal import GapMeasurement

# What a writer produces without seeing these calendars: every listed time read
# as the user's, every event blocking, no buffer, and no working hours.
FACE_VALUE = '''def find_shared_starts(api, attendees: list[str], days: list[str], minutes: int,
                       earliest: str = "", latest_end: str = "") -> dict:
    def to_minutes(hhmm):
        hours, mins = hhmm.split(":")
        return int(hours) * 60 + int(mins)

    def to_hhmm(total):
        return f"{total // 60:02d}:{total % 60:02d}"

    low = to_minutes(earliest) if earliest else 9 * 60
    high = to_minutes(latest_end) if latest_end else 17 * 60
    starts = {}
    for day in days:
        busy = []
        key = day
        for person in [""] + list(attendees):
            page = api.read_calendar(person, day)
            key = page["days"][0]["day"]
            busy += [(to_minutes(e["start"]), to_minutes(e["end"])) for e in page["days"][0]["events"]]
        found = []
        start = max(low, 9 * 60)
        start += (15 - start % 15) % 15
        while start + minutes <= min(high, 17 * 60):
            if all(start + minutes <= a or b <= start for a, b in busy):
                found.append(to_hhmm(start))
            start += 15
        if found:
            starts[key] = found
    return {"starts": starts}
'''


def _episodes(n=4):
    tasks = task_set("dev")[:n]
    return [Episode(t["id"], t["text"], [{"tool": "read_calendar", "arguments": {"person": "", "day": "mon"},
                                         "result": {"days": []}}], i % 2 == 0, "accepted" if i % 2 == 0 else "busy")
            for i, t in enumerate(tasks)]


def _scripted_writer(rule):
    """A writer stand-in: `rule(prompt)` decides which function it writes."""
    prompts = []

    async def model(**kwargs):
        prompt = kwargs["messages"][-1]["content"]
        prompts.append(prompt)
        reply = rule(prompt)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=reply, tool_calls=None))],
                               usage=None)

    return model, prompts


def _block(source):
    return f"Here it is.\n```python\n{source}```"


def test_the_bootstrapped_memory_is_accepted_on_every_development_task():
    rewards = [asyncio.run(act(bootstrapped_source(), t["id"]))[0] for t in task_set("dev")]
    assert rewards == [1.0] * 16


def test_a_function_written_blind_is_rejected_on_every_development_task():
    rewards = [asyncio.run(act(FACE_VALUE, t["id"]))[0] for t in task_set("dev")]
    assert rewards == [0.0] * 16


def test_behavior_match_is_exact_for_the_reference_and_partial_for_a_blind_function():
    signal = match_signal()
    cases = task_set("validation")
    same = asyncio.run(signal.measure(bootstrapped_source(), ground_truth=cases))
    assert same.score == 1.0 and not same.triggered
    blind = asyncio.run(signal.measure(FACE_VALUE, ground_truth=cases))
    assert blind.score < 1.0 and blind.triggered and blind.feedback


def test_a_crashing_function_mismatches_instead_of_failing_the_measurement():
    crash = "def find_shared_starts(api, attendees: list[str], days: list[str], minutes: int,\n" \
            "                       earliest: str = '', latest_end: str = '') -> dict:\n    return 1 // 0\n"
    result = asyncio.run(match_signal().measure(crash, ground_truth=task_set("validation")[:2]))
    assert result.score == 0.0


def test_experience_round_trips_through_episodic_memory_in_order():
    episodes = _episodes()

    async def go():
        return await recall_experience(await remember(episodes))

    recalled = asyncio.run(go())
    assert [e.task_id for e in recalled] == [e.task_id for e in episodes]
    assert recalled[1].reason == "busy" and recalled[0].steps == episodes[0].steps


def test_the_writer_extracts_the_function_and_retries_once_when_the_guard_refuses():
    replies = iter([_block("import os\n" + bootstrapped_source()), _block(bootstrapped_source())])
    model, prompts = _scripted_writer(lambda prompt: next(replies))
    with patch("helix.memory.learning_mechanisms.helix_acompletion", side_effect=model):
        source = asyncio.run(Writer().write("write it", CONTRACT))
    assert source == extract_source(_block(bootstrapped_source()))
    assert "refused" in prompts[1]


def test_the_control_writes_from_the_contract_alone():
    model, prompts = _scripted_writer(lambda prompt: _block(FACE_VALUE))
    with patch("helix.memory.learning_mechanisms.helix_acompletion", side_effect=model):
        learned = asyncio.run(CONTROL.learn(_episodes(), act, Writer(), CONTRACT))
    assert learned.writer_calls == 1 and "Request:" not in prompts[0]
    assert "def find_shared_starts(api" in prompts[0]


def test_value_learning_keeps_the_candidate_chapter_5s_rule_values_highest():
    episodes = [Episode(t["id"], t["text"], [], True, "accepted") for t in task_set("dev")[:4]]
    good = {episodes[1].request}
    # The candidate written from the second run is the right method; the rest read at face value.
    model, _ = _scripted_writer(lambda prompt: _block(bootstrapped_source() if any(r in prompt for r in good)
                                                      else FACE_VALUE))
    with patch("helix.memory.learning_mechanisms.helix_acompletion", side_effect=model):
        learned = asyncio.run(ValueLearning(epochs=1).learn(episodes, act, Writer(), CONTRACT))
    assert learned.source == extract_source(_block(bootstrapped_source()))
    first = episodes[0].task_id
    assert learned.values[first] == INITIAL_VALUE + ALPHA * (0.0 - INITIAL_VALUE)
    assert learned.values[episodes[1].task_id] > INITIAL_VALUE


def test_reflection_writes_a_lesson_and_a_revision_after_a_rejection():
    def rule(prompt):
        if prompt.startswith("A calendar assistant used the function"):
            return "Calendars are in their owner's local time; convert with the UTC offset."
        if "Lessons learned" in prompt:
            return _block(bootstrapped_source())
        return _block(FACE_VALUE)

    model, prompts = _scripted_writer(rule)
    episodes = [Episode(t["id"], t["text"], [], True, "accepted") for t in task_set("dev")[:3]]
    with patch("helix.memory.learning_mechanisms.helix_acompletion", side_effect=model):
        learned = asyncio.run(Reflection().learn(episodes, act, Writer(), CONTRACT))
    assert learned.notes == ["Calendars are in their owner's local time; convert with the UTC offset."]
    assert learned.source == extract_source(_block(bootstrapped_source()))


def test_mechanisms_are_proposed_with_chapter_5s_rl_first():
    assert list(MECHANISMS)[0] == "value_learning"
    assert set(MECHANISMS) == {"value_learning", "reflection", "success_distillation",
                               "transition_distillation", "skill_synthesis"}


def test_the_search_keeps_the_incumbent_on_a_tie_and_replaces_it_only_on_strict_improvement():
    scores = {"a": 0.5, "b": 0.5, "c": 0.75, "d": 0.6}

    async def evaluate(artifact):
        return GapMeasurement(score=scores[artifact.content["name"]]), {}

    async def go():
        seed = genesis("mechanism.test", ArtifactKind.CODE, {"name": "a"})
        search = CandidateSearch([{"name": n} for n in "abcd"], evaluate)
        variants = [v async for v in search.propose(seed, None, None, None)]
        return variants, await search.select(variants)

    variants, winner = asyncio.run(go())
    assert winner.content["name"] == "c"
    assert all(v.artifact.parent_id == ("mechanism.test", 1) for v in variants[1:])
    assert variants[0].artifact.parent_id is None


def test_a_learned_function_runs_under_the_guard_on_a_case():
    result = asyncio.run(run_on_case(bootstrapped_source(), task_set("validation")[0]))
    assert set(result) == {"starts"}


def test_act_reports_a_crash_as_a_crash_not_as_the_users_rejection():
    crash = "def find_shared_starts(api, attendees: list[str], days: list[str], minutes: int,\n" \
            "                       earliest: str = '', latest_end: str = '') -> dict:\n    return type(1)\n"
    reward, reason, _ = asyncio.run(act(crash, task_set("dev")[0]["id"]))
    assert reward == 0.0 and reason.startswith("the function crashed when run") and "type" in reason
    _, refused, _ = asyncio.run(act(FACE_VALUE, task_set("dev")[0]["id"]))
    assert refused.startswith("booking its earliest start was rejected by the user")


def test_the_contract_lists_the_builtins_a_learned_function_may_use():
    from helix.memory.code_skills import SAFE_BUILTINS
    from helix.memory.learning_mechanisms import contract_text

    text = contract_text(CONTRACT)
    assert all(name in text for name in SAFE_BUILTINS) and "type," not in text


def test_reflection_stops_revising_after_its_cap():
    model, prompts = _scripted_writer(lambda prompt: "A lesson." if prompt.startswith("A calendar assistant")
                                      else _block(FACE_VALUE))
    episodes = [Episode(t["id"], t["text"], [], True, "accepted") for t in task_set("dev")]
    with patch("helix.memory.learning_mechanisms.helix_acompletion", side_effect=model):
        learned = asyncio.run(Reflection(max_revisions=2).learn(episodes, act, Writer(), CONTRACT))
    assert len(learned.notes) == 2 and learned.writer_calls == 1 + 2 * 2


# ---------------------------------------------------------------- the second task class

FACE_VALUE_COMMON = FACE_VALUE.replace("def find_shared_starts(", "def find_common_starts(").replace(
    '''    return {"starts": starts}''',
    '''    days_found = [set(v) for v in starts.values()]
    if len(days_found) < len(days):
        return {"starts": []}
    return {"starts": sorted(set.intersection(*days_found))}''')


def test_the_second_class_has_a_common_solution_and_a_face_value_trap():
    from chapters.ch06.assistant.generator_same_time import SPLITS as SAME_SPLITS, solutions as common_solutions
    from chapters.ch06.assistant.generator_same_time import task_set as same_tasks

    for split in SAME_SPLITS:
        for task in same_tasks(split):
            assert 1 <= len(common_solutions(task)) <= 4
            assert task["truth"]["same_start"] and task["truth"]["count"] == len(task["truth"]["days"])


def test_the_second_bootstrapped_memory_is_accepted_and_a_blind_function_is_not():
    from chapters.ch06.assistant.generator_same_time import task_set as same_tasks

    source = bootstrapped_source("same_time")
    tasks = same_tasks("dev")
    assert [asyncio.run(act(source, t["id"]))[0] for t in tasks] == [1.0] * 16
    assert [asyncio.run(act(FACE_VALUE_COMMON, t["id"]))[0] for t in tasks] == [0.0] * 16


def test_the_checker_refuses_meetings_at_different_times_on_a_same_time_request():
    from chapters.ch06.assistant.environment import SchedulingEnvironment
    from chapters.ch06.assistant.generator_same_time import task_set as same_tasks

    task = same_tasks("dev")[0]
    env = SchedulingEnvironment(task)
    truth = task["truth"]
    for i, day in enumerate(truth["days"]):
        env.book_time("Sync", day, "09:00" if i == 0 else "09:15", truth["minutes"], truth["attendees"])
    verdict = env.check()
    assert not verdict["success"]


def test_a_function_without_type_hints_is_called_with_the_contracts_arguments():
    plain = bootstrapped_source()
    for hinted, bare in (("attendees: list[str]", "attendees"), ("days: list[str]", "days"),
                         ("minutes: int", "minutes")):
        assert hinted in plain
        plain = plain.replace(hinted, bare, 1)
    task = task_set("validation")[0]
    assert asyncio.run(run_on_case(plain, task)) == asyncio.run(run_on_case(bootstrapped_source(), task))


def test_a_function_the_guard_refuses_is_reported_not_raised():
    result = asyncio.run(run_on_case("import os\n", task_set("validation")[0]))
    assert "could not be called" in result["error"]
