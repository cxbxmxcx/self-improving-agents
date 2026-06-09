"""Deterministic tests for the Chapter 3 revenue data-investigation task.

These exercise the ground-truth pipeline and the scorers, which are plain code with
no LLM, so they run fast and pin down the task the search methods optimize against.
"""
from types import SimpleNamespace

from agents.revenue import (
    ground_truth, reconstruct_answer, score_revenue, score_smooth,
    score_smooth_with_feedback, score_with_feedback,
)

CANONICAL = [{"rep_name": n, "revenue": v} for n, v in ground_truth()]


def test_ground_truth_is_the_net_revenue_leaderboard():
    assert ground_truth() == [("Cy", 358.0), ("Ben", 296.0), ("Dee", 220.0)]


def test_exact_score_is_one_for_the_canonical_answer():
    assert score_revenue(CANONICAL) == 1.0


def test_exact_score_falls_when_a_rep_total_is_wrong():
    wrong = [{"rep_name": "Cy", "revenue": 408.0},
             {"rep_name": "Ben", "revenue": 296.0},
             {"rep_name": "Dee", "revenue": 220.0}]
    assert score_revenue(wrong) < 1.0


def test_smooth_score_is_one_for_the_canonical_answer():
    assert score_smooth(CANONICAL) == 1.0


def test_smooth_score_gives_partial_credit_for_one_missing_rule():
    # Missing the refund rule overstates Cy by 50; the other two reps stay correct.
    missing_refund = [{"rep_name": "Cy", "revenue": 408.0},
                      {"rep_name": "Ben", "revenue": 296.0},
                      {"rep_name": "Dee", "revenue": 220.0}]
    score = score_smooth(missing_refund)
    assert 0.5 < score < 1.0


def test_smooth_score_decreases_as_more_rules_are_missed():
    one_wrong = [{"rep_name": "Cy", "revenue": 408.0},
                 {"rep_name": "Ben", "revenue": 296.0},
                 {"rep_name": "Dee", "revenue": 220.0}]
    three_wrong = [{"rep_name": "Cy", "revenue": 558.0},
                   {"rep_name": "Ben", "revenue": 446.0},
                   {"rep_name": "Dee", "revenue": 260.0}]
    assert score_smooth(three_wrong) < score_smooth(one_wrong) < 1.0


def test_a_missing_rep_scores_zero_for_that_slot():
    two_reps = [{"rep_name": "Cy", "revenue": 358.0},
                {"rep_name": "Ben", "revenue": 296.0}]
    # Dee is absent, so its slot earns nothing: 2 of 3 perfect -> 2/3.
    assert abs(score_smooth(two_reps) - 2 / 3) < 1e-9


def test_feedback_names_the_wrong_positions_without_revealing_numbers():
    too_high = [{"rep_name": "Cy", "revenue": 408.0},
                {"rep_name": "Ben", "revenue": 296.0},
                {"rep_name": "Dee", "revenue": 220.0}]
    _, feedback = score_with_feedback(too_high)
    assert "Cy" in feedback and "358" not in feedback


def test_smooth_feedback_reports_direction_of_each_error():
    too_high = [{"rep_name": "Cy", "revenue": 508.0},
                {"rep_name": "Ben", "revenue": 296.0},
                {"rep_name": "Dee", "revenue": 220.0}]
    _, feedback = score_smooth_with_feedback(too_high)
    assert "too high" in feedback


def test_reconstruct_answer_reads_the_last_summarize_result():
    steps = [
        SimpleNamespace(kind="tool_call", payload={"name": "summarize"}),
        SimpleNamespace(kind="tool_result", payload={"result": {"answer": [["Cy", 358.0]]}}),
    ]
    trajectory = SimpleNamespace(steps=steps)
    assert reconstruct_answer(trajectory) == [["Cy", 358.0]]


def test_reconstruct_answer_is_empty_when_nothing_was_summarized():
    steps = [SimpleNamespace(kind="tool_call", payload={"name": "load_table"})]
    assert reconstruct_answer(SimpleNamespace(steps=steps)) == []
