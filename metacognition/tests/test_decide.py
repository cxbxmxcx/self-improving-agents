"""helix.metacognition.decide: decision programs serve requests with no model; Jev answers judgments (no key)."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from helix.deciders import MissingAnswer, ReplayDecider
from helix.metacognition.decide import LOCAL_SECONDS, Desk, Kit, UnsafeProgram, check_program, run_program, serve

from examples.decisions import learn, study

JEV = Path(__file__).parents[1] / "examples" / "decisions" / "recorded" / "jev.jsonl"


@pytest.mark.parametrize(("source", "refusal"), [
    ("import re\nasync def decide(request, kit):\n    return {}", "not Import"),
    ("async def decide(request, kit):\n    import re\n    return {}", "may not import"),
    ("async def decide(request, kit):\n    return kit._jev", "may not name '_jev'"),
    ("def decide(request, kit):\n    return {}", "does not define async def decide"),
    ("X = open('f')\nasync def decide(request, kit):\n    return {}", "top-level values must be literals"),
])
def test_the_guard_refuses_what_a_mechanism_may_not_do(source, refusal):
    with pytest.raises(UnsafeProgram, match=refusal):
        check_program(source)


async def test_a_program_reads_finds_and_calls_procedures_through_the_kit():
    async def payable(invoice_id):
        return {"invoice_id": invoice_id, "amount_usd": 10.0}

    kit = Kit(reads={"read_vendor": lambda vendor: {"email": "ap@" + vendor}}, procedures={"payable": payable})
    source = '''
async def decide(request, kit):
    ids = kit.find(r"INV-[0-9]+", request)
    answer = await kit.procedure("payable", invoice_id=ids[0])
    return {"action": "pay", "arguments": answer, "reply": kit.read("read_vendor", vendor="acme")["email"]}
'''
    decision = await run_program(source, "pay INV-7, not INV-7 twice", kit)
    assert decision == {"action": "pay", "arguments": {"invoice_id": "INV-7", "amount_usd": 10.0},
                        "reply": "ap@acme"}
    assert await run_program("async def decide(request, kit):\n    return 3", "x", kit) == {
        "handoff": "the program returned 3, neither an action nor a hand-off"}


async def test_a_failing_program_hands_the_request_off_and_a_handoff_is_not_judged():
    acted = []
    desk = Desk(world=lambda task: {}, reads=lambda world: {}, procedures=lambda world: {},
                actions={"act": lambda world, **arguments: acted.append(arguments)},
                verdict=lambda world, reply: {"success": True, "reason": "fine"})
    crash = await serve("async def decide(request, kit):\n    return 1 / 0", {"id": "c1", "text": "x"}, desk, None)
    assert (crash.handled, crash.right) == (False, None) and "ZeroDivisionError" in crash.reason
    served = await serve('async def decide(request, kit):\n    return {"action": "act", "arguments": {"a": 1}}',
                         {"id": "c2", "text": "x"}, desk, None)
    assert (served.handled, served.right, acted) == (True, True, [{"a": 1}])


async def test_what_a_served_request_cost_in_time_never_depends_on_the_clock():
    """A gap report quotes these seconds to the builder. With wall-clock time in them, a loaded machine changed a
    recorded search's question, and the search was paid for twice."""

    async def slow(**arguments):
        await asyncio.sleep(0.05)  # a procedure that takes real time, as one run in a child process does
        return {"ok": True}

    desk = Desk(world=lambda task: {}, reads=lambda world: {}, procedures=lambda world: {"look": slow},
                actions={"act": lambda world, **arguments: None},
                verdict=lambda world, reply: {"success": True, "reason": "fine"})
    program = ('async def decide(request, kit):\n    await kit.procedure("look")\n'
               '    return {"action": "act", "arguments": {}}')
    first = await serve(program, {"id": "c1", "text": "x"}, desk, None)
    second = await serve(program, {"id": "c1", "text": "x"}, desk, None)
    assert first.handled and first.seconds == second.seconds == LOCAL_SECONDS


async def test_jev_replays_from_its_recording_and_refuses_an_unrecorded_call(tmp_path):
    if not JEV.exists():
        pytest.skip("no Jev recording")
    kit = Kit({}, {}, ReplayDecider(JEV))
    with pytest.raises(MissingAnswer, match="no recorded answer"):
        await kit.ask({"request": "never asked"}, {"q": {"type": "noul", "instructions": "?"}})


@pytest.mark.parametrize(("domain", "handled", "right", "handoffs"), [("d1", 16, 16, 0), ("d3", 14, 14, 2)])
async def test_the_reference_programs_serve_the_held_out_requests_with_no_model(domain, handled, right, handoffs,
                                                                               tmp_path):
    if not JEV.exists():
        pytest.skip("no Jev recording")
    summary = await study.main(study.parse(["--domain", domain, "--output", str(tmp_path)]))
    assert (summary["handled"], summary["right"], summary["handoffs"]) == (handled, right, handoffs)
    assert summary["dollars_per_request"] < 0.0001  # Jev's input tokens only, at $0.042 per million


LEARNED = Path(__file__).parents[1] / "examples" / "decisions" / "learned"


async def run(domain: str, split: str, tmp_path: Path, mechanism: str | None = None) -> dict:
    argv = ["--domain", domain, "--wordings", "--split", split, "--quiet", "--output", str(tmp_path)]
    if mechanism:
        argv += ["--program", str(LEARNED / f"{domain}_{mechanism}.py")]
    return await study.main(study.parse(argv))


@pytest.mark.parametrize(("domain", "handled", "right", "handoffs"), [("d1", 64, 64, 4), ("d3", 56, 56, 6)])
async def test_a_program_that_lets_jev_judge_intent_serves_the_varied_wordings(domain, handled, right, handoffs,
                                                                              tmp_path):
    """Learned from the agent's runs on the development wordings; these wordings are held out."""
    if not JEV.exists():
        pytest.skip("no Jev recording")
    summary = await run(domain, "test", tmp_path, "program_judgment")
    assert (summary["handled"], summary["right"], summary["handoffs"]) == (handled, right, handoffs)
    assert summary["by_style"]["shorthand"]["right"] == summary["by_style"]["shorthand"]["requests"]


@pytest.mark.parametrize(("domain", "mechanism", "wrong"), [
    ("d1", None, 0), ("d3", None, 0), ("d1", "program_synthesis", 20), ("d3", "program_synthesis", 16),
    ("d1", "program_judgment", 0), ("d3", "program_judgment", 2)])
async def test_near_misses_show_who_understands_the_request(domain, mechanism, wrong, tmp_path):
    """The held-out near misses use the job's own words to ask for something else: all should be handed off.
    Word lists act on nearly all of them; Jev's judgment acts on almost none."""
    if not JEV.exists():
        pytest.skip("no Jev recording")
    summary = await run(domain, "stress2", tmp_path, mechanism)
    assert summary["handled"] - summary["right"] == wrong


async def test_the_learned_program_replays_from_its_recordings(tmp_path):
    if not JEV.exists():
        pytest.skip("no Jev recording")
    result = await learn.main(learn.parse(["--domain", "d3", "--mechanism", "program_judgment", "--experience",
                                           "dev+near", "--output", str(tmp_path)]))
    assert (result["trial"]["accepted"], result["trial"]["total"], result["trial"]["writer_calls"]) == (62, 62, 3)
    replayed = (tmp_path / "d3_program_judgment.py").read_text(encoding="utf-8")
    assert replayed == (LEARNED / "d3_program_judgment.py").read_text(encoding="utf-8")
