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
        # both training problems; on chapter 6's calendar classes it neither helps nor hurts.
        args.regime = "weak_writer"
        design = await study.phase_design(problems, args)
    assert design["best_seed"] == "transition_distillation"
    assert [round(row["score"], 3) for row in design["lineage"]] == [0.594, 0.75, 1.0]
    assert (design["accepted"], design["train"], round(design["test"], 3)) == (True, 1.0, 0.094)
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
