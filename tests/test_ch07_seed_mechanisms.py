"""Chapter 7, step 1: the seed mechanisms reproduce chapter 6 from its recordings.

The search starts from chapter 6's five mechanisms rewritten as source, so
those rewrites must learn exactly what the classes learned: the same prompts,
the same writer calls, the same functions, and the same validation matches
reported in chapter 6. Every model reply is replayed; no key is needed.
"""

from __future__ import annotations

import ast
import asyncio

import pytest

from chapters.ch06.assistant.learning import (CLASSES, act, behavior_cases, gather_experience, match_signal,
                                              recall_experience, remember)
from chapters.ch06.assistant.recording import Recording
from chapters.ch07.mechanisms import CONTROL, SEEDS, seed_source
from helix.memory import learning_mechanisms as ch6
from helix.memory.mechanism_runtime import Lab, run_mechanism

RECORDED = "chapters/ch06/assistant/recorded/"
WRITER = {"model": "gpt-5.4-mini", "params": {"reasoning_effort": "medium"}}
# Validation matches out of 16 in chapter 6's second search (table in section 6.3.4).
CHAPTER_6_MATCHES = {"no_experience": 0, "value_learning": 16, "reflection": 16, "success_distillation": 10,
                     "transition_distillation": 16, "skill_synthesis": 16}
# Listing 6.3: the kept mechanism and the control on the same-time class.
CHAPTER_6_SAME_TIME_MATCHES = {"no_experience": 0, "value_learning": 16}


def _writer():
    return ch6.Writer(model=WRITER["model"], params=dict(WRITER["params"]))


def _class_mechanism(name):
    return ch6.MECHANISMS.get(name) or ch6.CONTROL


async def _experience(task_class):
    return await recall_experience(await remember(await gather_experience(task_class=task_class)))


async def _both(name, experience, contract):
    chapter_6 = await _class_mechanism(name).learn(experience, act, _writer(), contract)
    seed = await run_mechanism(seed_source(name), Lab(experience, act, contract, _writer()), name=name)
    return chapter_6, seed


@pytest.fixture(scope="module")
def multi_attendee():
    async def go():
        with Recording(RECORDED + "reasoning_floor.jsonl").active():
            experience = await _experience("multi_attendee")
        with Recording(RECORDED + "learning_search.jsonl").active():
            results = {name: await _both(name, experience, CLASSES["multi_attendee"].contract)
                       for name in (*SEEDS, CONTROL)}
        signal, validation = match_signal(), behavior_cases("validation")
        matches = {name: (await signal.measure(seed.source, ground_truth=validation)).metadata["matched"]
                   for name, (_, seed) in results.items()}
        return results, matches

    return asyncio.run(go())


@pytest.mark.parametrize("name", [*SEEDS, CONTROL])
def test_seed_prompts_are_chapter_six_prompts_word_for_word(name):
    tree = ast.parse(seed_source(name))
    constants = {target.id: ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
                 for target in node.targets if target.id.endswith("_PROMPT")}
    assert constants
    for constant, text in constants.items():
        assert text == getattr(ch6, constant), constant


@pytest.mark.parametrize("name", [*SEEDS, CONTROL])
def test_seed_mechanisms_reproduce_chapter_six_learned_sources(multi_attendee, name):
    results, _ = multi_attendee
    chapter_6, seed = results[name]
    assert seed.source == chapter_6.source
    assert seed.writer_calls == chapter_6.writer_calls
    assert seed.notes == chapter_6.notes
    assert seed.values == chapter_6.values


def test_seed_mechanisms_reproduce_chapter_six_validation_matches(multi_attendee):
    _, matches = multi_attendee
    assert matches == CHAPTER_6_MATCHES


def test_the_value_learning_seed_reproduces_the_same_time_function_from_listing_6_3():
    async def go():
        contract = CLASSES["same_time"].contract
        with Recording(RECORDED + "generalization.jsonl").active():
            experience = await _experience("same_time")
            results = {name: await _both(name, experience, contract) for name in CHAPTER_6_SAME_TIME_MATCHES}
        signal, validation = match_signal("same_time"), behavior_cases("validation", "same_time")
        matches = {name: (await signal.measure(seed.source, ground_truth=validation)).metadata["matched"]
                   for name, (_, seed) in results.items()}
        return results, matches

    results, matches = asyncio.run(go())
    for name, (chapter_6, seed) in results.items():
        assert seed.source == chapter_6.source and seed.writer_calls == chapter_6.writer_calls, name
    assert matches == CHAPTER_6_SAME_TIME_MATCHES
