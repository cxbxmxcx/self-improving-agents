"""Tests for the panel introspection helpers."""

from __future__ import annotations

import pytest

from helix.chat_ui._introspect import (
    describe_archive,
    describe_eval_source,
    describe_policy,
    describe_search,
    describe_signal,
)


def test_describe_signal_renders_swap_and_agree_wrapper():
    from helix.signals.pairwise_judge import PairwiseJudge, SwapAndAgree
    sig = SwapAndAgree(PairwiseJudge(model="claude-sonnet-4-6"))
    out = describe_signal(sig)
    assert "SwapAndAgree" in out
    assert "PairwiseJudge" in out
    assert "claude-sonnet-4-6" in out


def test_describe_signal_renders_composite():
    from helix.signal import CompositeSignal
    from helix.signals.pairwise_judge import PairwiseJudge

    composite = CompositeSignal(
        signals=[
            PairwiseJudge(model="claude-haiku-4-5"),
            PairwiseJudge(model="claude-sonnet-4-6"),
        ],
        aggregator="weighted_mean",
        weights=[0.7, 0.3],
    )
    out = describe_signal(composite)
    assert "CompositeSignal" in out
    assert "weighted_mean" in out
    assert "0.7" in out
    assert "0.3" in out


def test_describe_signal_renders_bare_signal():
    from helix.signals.pairwise_judge import PairwiseJudge
    sig = PairwiseJudge(model="claude-haiku-4-5")
    out = describe_signal(sig)
    assert "PairwiseJudge" in out
    assert "claude-haiku-4-5" in out


def test_describe_search_renders_spo():
    from helix.search.spo import SPO
    s = SPO(proposer_model="claude-sonnet-4-6", rounds=3)
    out = describe_search(s)
    assert "SPO" in out
    assert "claude-sonnet-4-6" in out
    assert "rounds=3" in out


def test_describe_search_renders_strategy_chain():
    from helix.search import StrategyChain
    from helix.search.spo import SPO

    chain = StrategyChain(
        strategies=[
            SPO(proposer_model="claude-haiku-4-5"),
            SPO(proposer_model="claude-sonnet-4-6"),
        ],
        max_failures_per_strategy=2,
    )
    out = describe_search(chain)
    assert "StrategyChain" in out
    assert "max_failures=2" in out
    assert "SPO" in out


def test_describe_eval_source_uses_label():
    from helix.eval.dataset import EvalQuestion, EvalSet
    from helix.eval.source import FixedEvalSet
    es = FixedEvalSet(EvalSet(questions=[
        EvalQuestion(id="Q1", band=1, question="?", reference_answer="x")
    ]))
    out = describe_eval_source(es)
    assert "fixed_eval_set" in out or "FixedEvalSet" in out


def test_describe_policy_returns_field_dict():
    from helix.improvement import ImproverPolicy, Schedule
    p = ImproverPolicy(
        schedule=Schedule.INTERVAL,
        interval_sec=300.0,
        max_concurrent_questions=2,
    )
    out = describe_policy(p)
    assert out["schedule"] == "interval"
    assert out["interval_sec"] == 300.0
    assert out["max_concurrent_questions"] == 2


def test_describe_archive_includes_path():
    from helix.archive import SQLiteArchive
    arc = SQLiteArchive(":memory:")
    out = describe_archive(arc)
    assert "SQLiteArchive" in out
    assert ":memory:" in out
