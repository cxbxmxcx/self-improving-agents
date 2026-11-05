"""M1 acceptance: the chapter 6 calendar assistant, written with the SDK surface, replays E1.

Agent, function_tool, ModelSettings, and Runner must compile to requests
identical to the book's (same messages, tool schemas, and parameters), or the
recording has no reply and replay fails.
"""

from __future__ import annotations

from calendar_sdk import assistant
from chapters.ch06.assistant.environment import SchedulingEnvironment
from chapters.ch06.assistant.generator import task_set
from helix.eval.effort import EFFORTS, sweep

from metacog import ModelSettings, Reasoning, Recording, RunConfig, Runner
from metacog.pricing import PRICES

ASSISTANT = assistant()

# Study table E1, no memory: solved of 16, mean reasoning tokens, mean dollars.
EXPECTED = {"none": (2, 0, 0.0032), "low": (4, 1000, 0.0076), "medium": (12, 2077, 0.0125),
            "high": (13, 2810, 0.0162)}


async def run(task: dict, effort: str):
    env = SchedulingEnvironment(task)
    config = RunConfig(model_settings=ModelSettings(reasoning=Reasoning(effort=effort)), max_tool_calls=24)
    result = await Runner.run(ASSISTANT, task["text"], context=env, max_turns=16, run_config=config)
    verdict = env.check()
    return verdict["success"], verdict["reason"], result.trajectory


async def test_e1_no_memory_arm_replays_through_the_sdk(ch06_recording):
    with Recording(ch06_recording("reasoning_floor.jsonl")).active():
        result = await sweep("no_memory", task_set("dev"), run, efforts=EFFORTS, prices=PRICES)
    for effort, (solved, reasoning, dollars) in EXPECTED.items():
        cell = result.cells[effort]
        assert (cell.solved, cell.total) == (solved, 16), effort
        assert round(cell.mean("reasoning_tokens")) == reasoning, effort
        assert round(cell.mean("dollars"), 4) == dollars, effort
        # Replay restores each call's recorded latency, as the study reports it.
        assert cell.mean("seconds") > 1.0, effort
