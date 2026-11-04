"""Chapter 7, step 1: mechanisms as source, run through a metered lab under a guard."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from chapters.ch06.assistant.learning import CONTRACT, bootstrapped_source
from chapters.ch07.mechanisms import CONTROL, SEEDS, seed_source
from helix.memory.learning_mechanisms import Episode, Writer
from helix.memory.mechanism_runtime import (Lab, MechanismBudget, MechanismStopped, UnsafeMechanism,
                                            check_mechanism, run_mechanism)


def _block(source):
    return f"Here it is.\n```python\n{source}```"


def _scripted_writer(reply):
    calls = []

    async def model(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=reply, tool_calls=None))],
                               usage=None)

    return model, calls


async def _accept(source, task_id):
    return 1.0, "accepted", ""


def _lab(budget=None):
    episodes = [Episode("S001", "Book 30 minutes with Dana on Monday.", [], True, "accepted")]
    return Lab(episodes, _accept, CONTRACT, Writer(), budget)


def _run(source, lab=None, reply=None):
    model, calls = _scripted_writer(reply or _block(bootstrapped_source()))
    with patch("helix.memory.learning_mechanisms.helix_acompletion", side_effect=model):
        return asyncio.run(run_mechanism(source, lab or _lab(), name="test")), calls


@pytest.mark.parametrize("name", [*SEEDS, CONTROL])
def test_every_seed_mechanism_passes_the_mechanism_guard(name):
    check_mechanism(seed_source(name))


@pytest.mark.parametrize("source, refusal", [
    ("async def learn(lab):\n    import os\n    return ''\n", "may not import"),
    ("async def learn(lab):\n    return lab._writer\n", "may not name '_writer'"),
    ("async def learn(lab):\n    return ''.__class__\n", "may not name '__class__'"),
    ("async def learn(lab):\n    return '{x}'.format(x=1)\n", "lab.fill"),
    ("async def learn(lab):\n    try:\n        return ''\n    except:\n        return ''\n", "bare except"),
    ("def learn(lab):\n    return ''\n", "async def learn"),
    ("async def learn(lab, extra):\n    return ''\n", "exactly one argument"),
    ("PROMPT = open('x').read()\nasync def learn(lab):\n    return ''\n", "literals"),
])
def test_check_mechanism_refuses_imports_private_names_format_and_bare_except(source, refusal):
    with pytest.raises(UnsafeMechanism, match=refusal):
        check_mechanism(source)


def test_a_mechanism_returns_the_learned_function_with_its_writer_calls_and_notes():
    source = ("async def learn(lab):\n"
              "    lab.note('wrote one')\n"
              "    return await lab.write(lab.fill('{contract}', contract=lab.contract))\n")
    learned, calls = _run(source)
    # A provenance comment may precede the function in the reference file.
    assert any(line.startswith("def find_shared_starts(") for line in learned.source.splitlines())
    assert learned.writer_calls == 1 and learned.notes == ["wrote one"]
    assert "def find_shared_starts(api" in calls[0]["messages"][0]["content"]


def test_mechanism_runtime_stops_a_design_that_exceeds_its_writer_budget():
    source = ("async def learn(lab):\n"
              "    first = await lab.ask('one')\n"
              "    return await lab.ask('two')\n")
    with pytest.raises(MechanismStopped, match="1 writer calls") as stopped:
        _run(source, _lab(MechanismBudget(writer_calls=1)))
    assert stopped.value.writer_calls == 1


def test_a_design_that_catches_exceptions_cannot_swallow_its_budget_stop():
    source = ("async def learn(lab):\n"
              "    for attempt in range(5):\n"
              "        try:\n"
              "            await lab.practice('def f(): pass', 'S001')\n"
              "        except Exception:\n"
              "            pass\n"
              "    return 'def f(): pass'\n")
    with pytest.raises(MechanismStopped, match="2 practice calls"):
        _run(source, _lab(MechanismBudget(practice_calls=2)))


def test_mechanism_runtime_stops_a_design_that_loops_past_its_deadline():
    source = ("async def learn(lab):\n"
              "    count = 0\n"
              "    while True:\n"
              "        count += 1\n")
    with pytest.raises(MechanismStopped, match="seconds"):
        _run(source, _lab(MechanismBudget(seconds=0.2)))


def test_a_harness_failure_reaches_the_caller_instead_of_scoring_the_design_zero():
    async def missing(**kwargs):
        raise KeyError("no recorded reply for this call")

    source = ("async def learn(lab):\n"
              "    try:\n"
              "        return await lab.ask('x')\n"
              "    except Exception:\n"
              "        return 'def f(): pass'\n")
    with patch("helix.memory.learning_mechanisms.helix_acompletion", side_effect=missing):
        with pytest.raises(KeyError, match="no recorded reply"):
            asyncio.run(run_mechanism(source, _lab(), name="test"))


def test_a_failing_design_stops_with_its_own_error_as_the_reason():
    with pytest.raises(MechanismStopped, match="ZeroDivisionError"):
        _run("async def learn(lab):\n    return 1 // 0\n")


def test_a_design_that_returns_no_source_stops_instead_of_storing_an_empty_memory():
    with pytest.raises(MechanismStopped, match="no function source"):
        _run("async def learn(lab):\n    return '   '\n")


def test_fill_refuses_attribute_and_item_lookups_in_a_template():
    lab = _lab()
    assert lab.fill("{contract} and {more}", contract="a", more="b") == "a and b"
    for template in ("{x.attr}", "{x[0]}", "{}"):
        with pytest.raises(ValueError, match="plain names"):
            lab.fill(template, x="value")


def test_writer_effort_reaches_the_model_call_and_is_limited_to_the_searchable_settings():
    source = ("async def learn(lab):\n"
              "    return await lab.write(lab.fill('{contract}', contract=lab.contract), effort='low')\n")
    learned, calls = _run(source)
    assert calls[0]["reasoning_effort"] == "low"
    with pytest.raises(MechanismStopped, match="writer effort must be one of"):
        _run("async def learn(lab):\n    return await lab.ask('x', effort='high')\n")


def test_without_an_effort_the_writer_keeps_its_configured_setting():
    learned, calls = _run(seed_source(CONTROL))
    assert calls[0]["reasoning_effort"] == "medium"
