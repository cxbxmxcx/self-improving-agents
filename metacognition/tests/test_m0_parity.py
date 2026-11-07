"""M0 acceptance: the study's E1 reasoning floors, replayed through metacog's recording layer.

The agent side is still the book's chapter 6 harness; only the model calls
go through `metacog.recording`. Matching the study's table here proves the
replay layer serves the book's recordings exactly, which every later parity
test (and the SDK itself) depends on.
"""

from __future__ import annotations

from helix.eval.effort import EFFORTS, SOLID, sweep

from metacog.pricing import PRICES
from metacog.recording import Recording

# Study table E1: solved of 16, mean reasoning tokens, and mean dollars per task.
EXPECTED = {
    "no_memory": {"none": (2, 0, 0.0032), "low": (4, 1000, 0.0076),
                  "medium": (12, 2077, 0.0125), "high": (13, 2810, 0.0162)},
    "bootstrapped_memory": {"none": (16, 0, 0.0021), "low": (16, 82, 0.0025),
                            "medium": (16, 152, 0.0028), "high": (16, 253, 0.0033)},
}


async def test_e1_reasoning_floors_replay_through_metacog(ch06_recording):
    from chapters.ch06.assistant.generator import task_set
    from chapters.ch06.assistant.harness import BOOTSTRAPPED_MEMORY, install_memory, run_task

    tasks = task_set("dev")
    arms = {"no_memory": None, "bootstrapped_memory": BOOTSTRAPPED_MEMORY}
    with Recording(ch06_recording("reasoning_floor.jsonl"), mode="replay").active():
        for arm, entry in arms.items():
            memory = await install_memory(entry)

            async def run(task, effort, memory=memory):
                return await run_task(task, effort, memory=memory)

            result = await sweep(arm, tasks, run, efforts=EFFORTS, prices=PRICES)
            for effort, (solved, reasoning, dollars) in EXPECTED[arm].items():
                cell = result.cells[effort]
                assert (cell.solved, cell.total) == (solved, 16), (arm, effort)
                assert round(cell.mean("reasoning_tokens")) == reasoning, (arm, effort)
                assert round(cell.mean("dollars"), 4) == dollars, (arm, effort)
            expected_floor = None if arm == "no_memory" else "none"
            assert result.floor(SOLID) == expected_floor
