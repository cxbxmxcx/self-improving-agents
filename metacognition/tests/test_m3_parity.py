"""M3 acceptance: the study's offline experiments, replayed through metacog's toolkit.

E1  calibrate          both arms of the reasoning-floor sweep, and the controller table they imply
E2  search_procedure   GEPA over the procedure's code (attempt D), with the book's evaluator and prompts
E3  search_learners    all five mechanisms and the control on the same experience, kept by behavior match
"""

from __future__ import annotations

import random

from calendar_sdk import FIND_SHARED_STARTS, assistant, check
from chapters.ch06.assistant import context_search as cs
from chapters.ch06.assistant.environment import SchedulingEnvironment
from chapters.ch06.assistant.generator import task_set
from chapters.ch06.assistant.harness import BOOTSTRAPPED_MEMORY, MEMORIES
from chapters.ch06.assistant.learning import (
    act,
    behavior_cases,
    bootstrapped_source,
    gather_experience,
    recall_experience,
    remember,
    run_on_case,
)

from metacog import Recording, RunConfig
from metacog.meta import CALENDAR, Metacognition
from metacog.offline import Case, calibrate, search_learners, search_procedure


async def test_e1_floors_and_the_controller_table_through_calibrate(ch06_recording):
    meta = Metacognition([FIND_SHARED_STARTS], verifier=check, recall_k=1, learning="inline")
    meta.install("find_shared_starts", (MEMORIES / BOOTSTRAPPED_MEMORY["file"]).read_text(encoding="utf-8"))
    cases = [Case(t["id"], t["text"], t) for t in task_set("dev")]
    with Recording(ch06_recording("reasoning_floor.jsonl")).active():
        floors = await calibrate(assistant(metacognition=meta), cases, verifier=check, world=SchedulingEnvironment,
                                 max_turns=16, run_config=RunConfig(max_tool_calls=24))
    solved = {arm: [floors.sweeps[arm].cells[e].solved for e in ("none", "low", "medium", "high")]
              for arm in ("no_memory", "memory")}
    assert solved == {"no_memory": [2, 4, 12, 13], "memory": [16, 16, 16, 16]}
    assert (floors.floor("no_memory"), floors.floor("memory")) == (None, "none")
    assert floors.table() == {"memory": "none", "no_memory": "high"}  # the study's E5 controller
    assert (floors.signal("no_memory").score, floors.signal("memory").score) == (0.0, 1.0)
    assert meta.store.count() == 0  # calibration runs are frozen: nothing recorded


async def test_e2_code_search_keeps_v15(ch06_recording):
    seed = (MEMORIES / "find_shared_starts.contract_only.py").read_text(encoding="utf-8")
    with Recording(ch06_recording("context_search.jsonl"), distinct_repeats=True).active():
        found = await search_procedure(
            FIND_SHARED_STARTS, seed_source=seed, evaluator=cs.evaluator, mutation_prompt=cs.mutation_prompt(),
            crossover_prompt=cs.crossover_prompt(), proposer_model="gpt-5.4-mini",
            proposer_params={"reasoning_effort": "medium"}, population=4, generations=4, rng=random.Random(1165002))
    assert found.version == 15
    parents = {row["version"]: row["parent"] for row in found.lineage}
    accepted = {row["version"]: row["metadata"]["accepted"] for row in found.lineage}
    assert (parents[15], parents[12], parents[17]) == (12, 6, 10)
    assert (accepted[12], accepted[15], accepted[17]) == (12, 16, 12)
    for split in ("validation", "test"):
        assert (await cs.acceptance(found.source, split))[:2] == (16, 16), split


async def test_e3_learning_mechanism_search(ch06_recording):
    with Recording(ch06_recording("reasoning_floor.jsonl")).active():
        episodes = await gather_experience()
    experience = await recall_experience(await remember(episodes))
    with Recording(ch06_recording("learning_search.jsonl")).active():
        found = await search_learners(
            FIND_SHARED_STARTS, experience, practice=act, reference=bootstrapped_source(), run_case=run_on_case,
            cases=behavior_cases("validation"), test_cases=behavior_cases("test"), prompts=CALENDAR,
            case_id=lambda task: task["id"])
    table = {t.mechanism: (t.matched, t.writer_calls) for t in found.trials}
    assert (found.control.matched, found.control.writer_calls) == (0, 1)
    assert table == {"value_learning": (16, 16), "reflection": (16, 7), "success_distillation": (10, 1),
                     "transition_distillation": (16, 1), "skill_synthesis": (16, 1)}
    assert found.kept == "value_learning" and found.test == (16, 16)
