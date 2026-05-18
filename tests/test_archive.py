"""SQLiteArchive round-trip tests."""

from __future__ import annotations

import pytest

from helix.archive import SQLiteArchive, SamplingStrategy
from helix.artifact import ArtifactKind, genesis
from helix.search.base import Variant
from helix.signal import GapMeasurement, Preference


@pytest.mark.asyncio
async def test_record_and_retrieve_by_id():
    arc = SQLiteArchive(":memory:")
    a = genesis("p", ArtifactKind.PROMPT, "hello")
    await arc.record(
        Variant(artifact=a, parent=a, search_method="human"),
        GapMeasurement(score=0.5),
    )
    fetched = await arc.by_id("p", version=1)
    assert fetched is not None
    assert fetched.artifact.content == "hello"
    assert fetched.measurement.score == 0.5


@pytest.mark.asyncio
async def test_best_orders_by_score_descending():
    arc = SQLiteArchive(":memory:")
    a = genesis("p", ArtifactKind.PROMPT, "v1")
    b = a.mutate("v2", created_by="spo")
    c = b.mutate("v3", created_by="spo")
    await arc.record(Variant(artifact=a, parent=a, search_method="human"), GapMeasurement(score=0.3))
    await arc.record(Variant(artifact=b, parent=a, search_method="spo"), GapMeasurement(score=0.9))
    await arc.record(Variant(artifact=c, parent=b, search_method="spo"), GapMeasurement(score=0.6))
    top = await arc.best(k=3)
    scores = [v.measurement.score for v in top]
    assert scores == [0.9, 0.6, 0.3]


@pytest.mark.asyncio
async def test_lineage_reconstructs_genesis_to_leaf():
    arc = SQLiteArchive(":memory:")
    a = genesis("p", ArtifactKind.PROMPT, "v1")
    b = a.mutate("v2", created_by="spo")
    c = b.mutate("v3", created_by="spo")
    await arc.record(Variant(artifact=a, parent=a, search_method="human"), GapMeasurement(score=0.3))
    await arc.record(Variant(artifact=b, parent=a, search_method="spo"), GapMeasurement(score=0.5))
    await arc.record(Variant(artifact=c, parent=b, search_method="spo"), GapMeasurement(score=0.7))
    chain = await arc.lineage(c)
    versions = [art.version for art in chain]
    assert versions == [1, 2, 3]


@pytest.mark.asyncio
async def test_descendants_returns_immediate_children():
    # Two sibling branches share parent (p, v1) but use distinct artifact ids
    # so they get their own (id, version) primary keys. Forks across ids is
    # the realistic shape: SPO and GEPA produce different *named* artifacts.
    arc = SQLiteArchive(":memory:")
    a = genesis("p", ArtifactKind.PROMPT, "v1")
    b_kind = a.mutate("v2-a", created_by="spo")
    # Simulate a second-line fork by using a different id but pointing parent
    # back at a. In practice GEPA would do this when forking off a known seed.
    from helix.artifact import Artifact
    c_fork = Artifact(
        id="p_fork", version=1, kind=a.kind, content="v2-b",
        parent_id=a.ref, created_by="gepa",
    )
    await arc.record(Variant(artifact=a, parent=a, search_method="human"), GapMeasurement(score=0.3))
    await arc.record(Variant(artifact=b_kind, parent=a, search_method="spo"), GapMeasurement(score=0.5))
    await arc.record(Variant(artifact=c_fork, parent=a, search_method="gepa"), GapMeasurement(score=0.6))
    desc = await arc.descendants(a)
    assert len(desc) == 2
    assert {d.created_by for d in desc} == {"spo", "gepa"}


@pytest.mark.asyncio
async def test_sample_best_returns_top_scorer():
    arc = SQLiteArchive(":memory:")
    a = genesis("p", ArtifactKind.PROMPT, "v1")
    b = a.mutate("v2", created_by="spo")
    await arc.record(Variant(artifact=a, parent=a, search_method="human"), GapMeasurement(score=0.3))
    await arc.record(Variant(artifact=b, parent=a, search_method="spo"), GapMeasurement(score=0.9))
    v = await arc.sample(SamplingStrategy.BEST)
    assert v.measurement.score == 0.9


@pytest.mark.asyncio
async def test_pareto_front_single_objective_returns_max():
    arc = SQLiteArchive(":memory:")
    a = genesis("p", ArtifactKind.PROMPT, "v1")
    b = a.mutate("v2", created_by="spo")
    c = b.mutate("v3", created_by="spo")
    await arc.record(Variant(artifact=a, parent=a, search_method="human"), GapMeasurement(score=0.3))
    await arc.record(Variant(artifact=b, parent=a, search_method="spo"), GapMeasurement(score=0.9))
    await arc.record(Variant(artifact=c, parent=b, search_method="spo"), GapMeasurement(score=0.7))
    front = await arc.pareto_front(["score"])
    assert len(front) == 1
    assert front[0].measurement.score == 0.9


@pytest.mark.asyncio
async def test_all_artifacts_with_measurements_returns_complete_set():
    arc = SQLiteArchive(":memory:")
    a = genesis("p", ArtifactKind.PROMPT, "v1")
    b = a.mutate("v2", created_by="spo")
    await arc.record(Variant(artifact=a, parent=a, search_method="human"), GapMeasurement(score=0.4))
    await arc.record(Variant(artifact=b, parent=a, search_method="spo"), GapMeasurement(score=0.6))
    everything = await arc.all_artifacts_with_measurements()
    versions = sorted(v.artifact.version for v in everything)
    assert versions == [1, 2]
    scores = {v.artifact.version: v.measurement.score for v in everything}
    assert scores == {1: 0.4, 2: 0.6}


@pytest.mark.asyncio
async def test_measurement_history_returns_chronological_list():
    arc = SQLiteArchive(":memory:")
    a = genesis("p", ArtifactKind.PROMPT, "seed")
    variant = Variant(artifact=a, parent=a, search_method="human")
    # Record three measurements over time.
    await arc.record(variant, GapMeasurement(score=0.3))
    await arc.record(variant, GapMeasurement(score=0.5))
    await arc.record(variant, GapMeasurement(score=0.7))
    history = await arc.get_measurement_history("p", 1)
    assert [m.score for m in history] == [0.3, 0.5, 0.7]


@pytest.mark.asyncio
async def test_lineage_tree_returns_nested_forest():
    arc = SQLiteArchive(":memory:")
    a = genesis("p", ArtifactKind.PROMPT, "v1")
    b = a.mutate("v2", created_by="spo")
    c = b.mutate("v3", created_by="spo")
    await arc.record(Variant(artifact=a, parent=a, search_method="human"), GapMeasurement(score=0.4))
    await arc.record(Variant(artifact=b, parent=a, search_method="spo"), GapMeasurement(score=0.6))
    await arc.record(Variant(artifact=c, parent=b, search_method="spo"), GapMeasurement(score=0.8))
    forest = await arc.lineage_tree()
    assert len(forest) == 1
    root = forest[0]
    assert root["artifact"].version == 1
    assert len(root["children"]) == 1
    assert root["children"][0]["artifact"].version == 2
    assert len(root["children"][0]["children"]) == 1
    assert root["children"][0]["children"][0]["artifact"].version == 3


@pytest.mark.asyncio
async def test_diversity_metrics_counts_unique_hashes():
    arc = SQLiteArchive(":memory:")
    a = genesis("p", ArtifactKind.PROMPT, "v1")
    b = a.mutate("v2", created_by="spo")
    await arc.record(Variant(artifact=a, parent=a, search_method="human"), GapMeasurement(score=0.3))
    await arc.record(Variant(artifact=b, parent=a, search_method="spo"), GapMeasurement(score=0.6))
    metrics = await arc.diversity_metrics()
    assert metrics.n_variants == 2
    assert metrics.n_unique_content_hashes == 2
    assert metrics.score_range == (0.3, 0.6)
