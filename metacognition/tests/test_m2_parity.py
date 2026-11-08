"""M2 acceptance: the study's E5 online loop, replayed through Metacognition on the SDK surface.

The starting memory is what chapter 6's learning search kept (value learning
on E1's medium and high runs), relearned here with metacog's own port of the
mechanism and the book's prompts. The stream is the study's: six test
requests, then fourteen after the client's note changes to "half an hour".
Every agent call and writer call must match the book's recordings.
"""

from __future__ import annotations

import pytest
from chapters.ch06.assistant import generator, generator_drift
from chapters.ch06.assistant.environment import SchedulingEnvironment
from chapters.ch06.assistant.learning import CONTRACT, act, gather_experience, recall_experience, remember
from helix.memory.learning_mechanisms import Writer
from helix.metacognition import CALENDAR, Controller, Gate, Metacognition, ValueLearning
from helix.metacognition.pricing import PRICES

from examples.calendar.agent import FIND_SHARED_STARTS, assistant, check
from metacog import Recording, RunConfig, Runner

WRITER = {"model": "gpt-5.4-mini", "params": {"reasoning_effort": "medium"}}

# The study's trace: (task, procedure offered, effort, accepted).
TRACE = (
    [(f"S20{i}", True, "none", True) for i in range(1, 7)]
    + [("D701", True, "none", False), ("D702", True, "none", False), ("D703", False, "high", False)]
    + [(f"D70{i}", False, "high", True) for i in range(4, 7)]
    + [(f"D7{i:02d}", True, "none", True) for i in range(7, 15)]
)


async def kept_memory(ch06_recording) -> str:
    """Listing 6.3's kept memory: value learning on E1's medium and high runs, recalled from episodic memory."""
    with Recording(ch06_recording("reasoning_floor.jsonl")).active():
        episodes = await gather_experience()
    experience = await recall_experience(await remember(episodes))
    with Recording(ch06_recording("learning_search.jsonl")).active():
        learned = await ValueLearning(CALENDAR).learn(experience, act, Writer(**WRITER), CONTRACT)
    assert learned.writer_calls == 16
    return learned.source


@pytest.fixture
def meta():
    gate = Gate(cases=[t["id"] for t in generator_drift.task_set("gate")])
    return Metacognition(
        [FIND_SHARED_STARTS], verifier=check, learner=ValueLearning(CALENDAR), writer_model=WRITER["model"],
        writer_params=WRITER["params"], practice=act, gate=gate, controller=Controller({"memory": "none",
        "no_memory": "high"}), recall_k=1, relearn_after=4, learning="inline", prices=PRICES)


async def test_e5_online_loop_replays_through_metacognition(ch06_recording, meta):
    meta.install("find_shared_starts", await kept_memory(ch06_recording), created_by="value_learning")
    agent = assistant(metacognition=meta)
    stream = generator.task_set("test")[:6] + generator_drift.task_set("stream")[:14]
    trace, events = [], {}
    with Recording(ch06_recording("online_loop.jsonl")).active():
        for n, task in enumerate(stream, 1):
            env = SchedulingEnvironment(task)
            result = await Runner.run(agent, task["text"], context=env, max_turns=16,
                                      run_config=RunConfig(max_tool_calls=24, case_id=task["id"]))
            report = result.metacognition
            trace.append((task["id"], bool(report.offered), report.effort, report.verdict.accepted))
            if report.events:
                events[n] = report.events
    assert trace == TRACE
    assert list(events) == [8, 12]
    assert events[8][0].startswith("detector: quarantined find_shared_starts@1 (Dana Park asked for 30 minutes")
    assert events[12] == ["learner: value_learning wrote find_shared_starts from 4 runs; gate 6/6 passed, installed"]

    summary = meta.summary()
    none, high = summary["by_effort"]["none"], summary["by_effort"]["high"]
    assert (none["runs"], none["accepted"], round(none["mean_dollars"], 4)) == (16, 14, 0.0021)
    assert (high["runs"], high["accepted"], round(high["mean_reasoning_tokens"])) == (4, 3, 5133)
    assert round(high["mean_dollars"], 4) == 0.0258
    assert (summary["learning"]["writer_calls"], round(summary["learning"]["dollars"], 4)) == (3, 0.1805)
    assert round(summary["break_even_requests"], 1) == 7.6
    assert summary["procedures"]["find_shared_starts"] == [(1, "quarantined", "value_learning"),
                                                           (2, "trusted", "value_learning")]
