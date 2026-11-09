"""A5 live (DOMAINS.md): learner tournaments (L2) and a learner design search (L3) on learning problems built
from the domains' recordings, replayed from those recordings (no API key)."""

from __future__ import annotations

import pytest

from examples.learning import study
from metacog import Recording

LEARNING = study.HERE / "recorded" / "learning.jsonl"
DESIGNED = study.HERE / "designed"
# Held-out cases accepted (of 16) by each seed's function, per training problem, as recorded on 2026-11-06.
TOURNAMENTS = {
    "low_writer": {
        "expenses": {"value_learning": 16, "reflection": 16, "success_distillation": 16,
                     "transition_distillation": 16, "skill_synthesis": 16},
        "refunds": {"value_learning": 0, "reflection": 0, "success_distillation": 16,
                    "transition_distillation": 16, "skill_synthesis": 16},
    },
    "weak_writer": {
        "expenses": {"value_learning": 16, "reflection": 16, "success_distillation": 0,
                     "transition_distillation": 16, "skill_synthesis": 16},
        "refunds": {"value_learning": 0, "reflection": 0, "success_distillation": 8,
                    "transition_distillation": 0, "skill_synthesis": 0},
    },
}


async def test_the_recorded_learning_study_replays(tmp_path):
    if not LEARNING.exists():
        pytest.skip("no recorded learning study")
    args = study.parse(["--output", str(tmp_path), "--population", "4", "--generations", "3"])
    problems = await study.build_problems(args, study.Spend(0.0))  # the same runs serve both regimes
    with Recording(LEARNING).active():
        # L2: no seed learns both training problems best, so the loop's tournament picks per problem: the
        # cheapest of those that learned the most.
        for regime, expected in TOURNAMENTS.items():
            args.regime = regime
            report = await study.phase_tournaments(problems, args)
            assert {problem: {t["mechanism"]: t["accepted"] for t in result["trials"]}
                    for problem, result in report.items()} == expected
            assert {problem: result["winner"] for problem, result in report.items()} == {
                "expenses": "reflection", "refunds": "success_distillation"}
        # L3 under the weak writer: GEPA from transition distillation, three rewrites to a design that learns
        # both training problems; on chapter 6's calendar classes it only ties the seed.
        args.regime = "weak_writer"
        design = await study.phase_design(problems, args)
    assert design["best_seed"] == "transition_distillation"
    assert [round(row["score"], 3) for row in design["lineage"]] == [0.594, 0.75, 1.0]
    # On the calendar classes it only tied the seed (0.09 each): under the stricter gate a tie is no sign of
    # transfer, so it is refused there. On D3 it does transfer (test_the_design_transfers_to_d3s_invoices).
    assert (design["accepted"], design["standing"], design["train"], round(design["test"], 3)) == (
        False, "refused", 1.0, 0.094)
    assert [(s["problem"], s["accepted"]) for s in design["scores"]["test"]] == [("multi_attendee", 2),
                                                                                 ("same_time", 1)]
    replayed = (tmp_path / f"{design['name']}.py").read_text(encoding="utf-8")
    assert replayed == (DESIGNED / f"{design['name']}.py").read_text(encoding="utf-8")


async def test_the_design_transfers_to_d3s_invoices(tmp_path):
    """Item 1 of A5's follow-ups: on D3's intake and payment, which no design saw, every seed fails under the
    weak writer and the design searched on D1 and D2 learns most cases."""
    if not LEARNING.exists():
        pytest.skip("no recorded learning study")
    args = study.parse(["--output", str(tmp_path), "--regime", "weak_writer", "--test", "invoices_intake",
                        "invoices_payment", "--designs", "designed_ee25cd476662"])
    problems = await study.build_problems(args, study.Spend(0.0))
    with Recording(LEARNING).active():
        report = await study.phase_transfer(problems, args)
    seeds = {"value_learning": 0, "reflection": 0, "success_distillation": 0, "transition_distillation": 0}
    accepted = {problem: {t["mechanism"]: t["accepted"] for t in result["trials"]}
                for problem, result in report.items()}
    assert accepted == {
        "invoices_intake": {**seeds, "skill_synthesis": 2, "designed_ee25cd476662": 7},
        "invoices_payment": {**seeds, "skill_synthesis": 0, "designed_ee25cd476662": 12}}


async def test_repeated_design_searches_vary(tmp_path):
    """Item 3 of A5's follow-ups: two more searches, each with fresh designer samples (a label), from other random
    starting points. Both beat the seeds on training; on D3 one transfers a little and one not at all."""
    if not LEARNING.exists():
        pytest.skip("no recorded learning study")
    found = {}
    for seed in (11, 23):
        args = study.parse(["--output", str(tmp_path), "--regime", "weak_writer", "--test", "invoices_intake",
                            "invoices_payment", "--seed", str(seed), "--label", f"s{seed}", "--generations", "3"])
        if not found:
            problems = await study.build_problems(args, study.Spend(0.0))
        with Recording(LEARNING).active():
            design = await study.phase_design(problems, args)
        found[seed] = ([round(row["score"], 3) for row in design["lineage"]], design["train"],
                       [(s["problem"], s["accepted"]) for s in design["scores"]["test"]], design["reason"])
    assert found[11][:3] == ([0.5, 0.75, 0.5, 0.5], 0.75, [("invoices_intake", 5), ("invoices_payment", 0)])
    assert found[23][:3] == ([0.906, 0.0], 0.90625, [("invoices_intake", 0), ("invoices_payment", 0)])
    assert found[23][3].endswith("ties it on the test problems (0.00 vs 0.00), so there is no sign it transfers")


async def test_a_search_with_validation_is_judged_on_a_class_it_never_touched(tmp_path):
    """Every candidate is also scored on D3 intake while the search runs; D3 payment stays held out. The pick
    transfers to intake (5 of 12, where the seed learns none) but ties the seed at zero on payments, so the
    stricter gate refuses it."""
    if not LEARNING.exists():
        pytest.skip("no recorded learning study")
    args = study.parse(["--output", str(tmp_path), "--regime", "weak_writer", "--validation", "invoices_intake",
                        "--test", "invoices_payment", "--seed", "31", "--label", "v31", "--generations", "3"])
    problems = await study.build_problems(args, study.Spend(0.0))
    with Recording(LEARNING).active():
        design = await study.phase_design(problems, args)
    assert [(round(row["score"], 3), round(row["validation"], 3)) for row in design["lineage"]] == [
        (0.0, 0.0), (0.5, 0.0), (0.0, 0.0), (0.906, 0.417)]
    assert (design["standing"], design["accepted"], design["train"], round(design["validation"], 3),
            design["test"]) == ("refused", False, 0.90625, 0.417, 0.0)


async def test_a_capped_recording_stops_before_a_call_could_pass_the_budget(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import helix.llm_call
    from helix.metacognition import recording

    async def model(**kwargs):
        prompt = 100_000 if kwargs["reasoning_effort"] == "medium" else 100  # 7.5 cents a call at medium
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok", tool_calls=None))],
            usage=SimpleNamespace(prompt_tokens=prompt, completion_tokens=0,
                                  completion_tokens_details=SimpleNamespace(reasoning_tokens=0),
                                  prompt_tokens_details=SimpleNamespace(cached_tokens=0)))

    recording.install()
    monkeypatch.setattr(recording, "_original", model)
    spend = study.Spend(0.2)
    capped = study.CappedRecording(tmp_path / "r.jsonl", mode="record", spend=spend)

    def call(n: int, effort: str) -> dict:
        return {"model": "gpt-5.4-mini", "messages": [{"role": "user", "content": str(n)}], "tools": None,
                "reasoning_effort": effort}

    with capped.active():
        await helix.llm_call.acompletion(**call(1, "medium"))
        await helix.llm_call.acompletion(**call(2, "medium"))
        with pytest.raises(study.BudgetSpent):  # a third would make 22.5 cents
            await helix.llm_call.acompletion(**call(3, "medium"))
        await helix.llm_call.acompletion(**call(4, "none"))  # a call this small still fits
    assert capped.live_calls == 3 and spend.total <= spend.limit
