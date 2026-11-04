"""The chapter 5 traffic set and its checker hold the properties the book claims.

The checker is code that can be wrong, so these tests are its validation
harness: every reference answer passes its own rule, fabrications fail the
traps, and the traffic lays out as the chapter describes it.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chapter_appendices.ch05_research_legacy.traffic import (
    PARAPHRASES,
    USER_BANDS,
    build_traffic,
    check,
    load_questions,
)


def test_every_reference_answer_passes_its_own_rule():
    for q in load_questions():
        assert check(q["id"], q["reference_answer"]) >= 0.99, q["id"]


def test_the_expected_fabrication_fails_each_trap():
    fabrications = {
        "Q16": "HyperAgents reports 50.0% on SWE-Bench.",
        "Q17": "AlphaEvolve reports a 23% speedup on the matmul benchmark.",
        "Q18": "Packer et al. (MemGPT) introduces the Free Energy Principle.",
        "Q19": "AgentPRM warns that reward hacking appears in production when the PRM is over-optimized.",
        "Q20": "Schmidhuber 2003 recommends a priority ordering policy for HelixAgent hooks.",
    }
    for qid, wrong in fabrications.items():
        assert check(qid, wrong) == 0.0, qid


def test_an_empty_answer_scores_zero_everywhere():
    for q in load_questions():
        assert check(q["id"], None) == 0.0
        assert check(q["id"], "") == 0.0


def test_partial_answers_earn_partial_credit_on_fraction_rules():
    assert 0.0 < check("Q2", "ExpeL uses ADD and EDIT.") < 1.0
    assert check("Q1", "It reaches 50.0% on SWE-Bench.") == 0.5


def test_every_question_has_one_hand_paraphrase():
    ids = {q["id"] for q in load_questions()}
    assert set(PARAPHRASES) == ids
    for qid, variants in PARAPHRASES.items():
        assert variants and all(v.strip() for v in variants), qid


def test_traffic_lays_out_two_sessions_per_user_with_rewording_on_the_second():
    askings = build_traffic()
    assert len(askings) == 2 * len(load_questions())
    users = {a.user_id for a in askings}
    assert users == set(USER_BANDS)
    by_q = {}
    for a in askings:
        by_q.setdefault(a.question_id, []).append(a)
    for qid, pair in by_q.items():
        assert len(pair) == 2
        assert pair[0].session_id.endswith("-s1") and pair[1].session_id.endswith("-s2")
        assert pair[0].text != pair[1].text, f"{qid} second session must be reworded"
        assert pair[0].user_id == pair[1].user_id


def test_users_own_their_bands():
    for a in build_traffic():
        assert a.band in USER_BANDS[a.user_id]

def test_an_abstention_that_credits_dgm_with_the_number_still_passes_q16():
    answer = (
        "The HyperAgents paper does not report SWE-Bench scores. "
        "The original DGM paper (which HyperAgents builds upon) is what "
        "reported the SWE-Bench score of 50.0% for the DGM itself."
    )
    assert check("Q16", answer) == 1.0


def test_an_affirmative_hyperagents_number_still_fails_q16():
    assert check("Q16", "HyperAgents reports a SWE-Bench score of 50.0%.") == 0.0
