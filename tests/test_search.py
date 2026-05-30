"""Search protocol conformance and budget tests."""

from __future__ import annotations

import pytest

from helix.search.base import Search, SearchBudget, SearchKind, Variant
from helix.search.hillclimb import HillClimb
from helix.search.spo import SPO
from helix.search.gepa import GEPA


def test_budget_remaining_and_exhausted():
    b = SearchBudget(max_candidates=3)
    assert not b.exhausted()
    b.saw_candidate()
    b.saw_candidate()
    b.saw_candidate()
    assert b.exhausted()


def test_budget_charges_tokens_and_dollars():
    from helix.signal import Cost
    b = SearchBudget(max_tokens=1000, max_dollars=0.10)
    b.charge(Cost(tokens=400, dollars=0.04))
    rem = b.remaining()
    assert rem["tokens"] == 600
    assert rem["dollars"] == pytest.approx(0.06)
    assert not b.exhausted()
    b.charge(Cost(tokens=700))
    assert b.exhausted()


def test_hillclimb_has_correct_kind():
    h = HillClimb()
    assert h.kind == SearchKind.HILL_CLIMB
    assert isinstance(h, Search)


def test_spo_has_correct_kind():
    s = SPO()
    assert s.kind == SearchKind.PAIRWISE
    assert isinstance(s, Search)


def test_gepa_has_correct_kind():
    # GEPA requires agent and eval_source for internal population
    # evaluation (SPEC §15.2 — agent.with_artifacts replaces factory).
    from helix.eval.dataset import EvalQuestion, EvalSet
    from helix.eval.source import FixedEvalSet

    class _DummyAgent:
        def with_artifacts(self, overrides):
            return self

    es = FixedEvalSet(EvalSet(questions=[EvalQuestion(id="X", band=1, question="x", reference_answer="x")]))
    g = GEPA(
        agent=_DummyAgent(),
        eval_source=es,
    )
    assert g.kind == SearchKind.GENETIC_PARETO
    assert isinstance(g, Search)


def test_variant_dataclass_carries_parent_and_search_method():
    from helix.artifact import genesis, ArtifactKind, Subtype
    parent = genesis("p", Subtype.PROMPT, "v1")
    child = parent.mutate("v2", created_by="spo")
    v = Variant(artifact=child, parent=parent, search_method="spo")
    assert v.artifact.parent_id == parent.ref
    assert v.search_method == "spo"
    assert v.measurement is None
