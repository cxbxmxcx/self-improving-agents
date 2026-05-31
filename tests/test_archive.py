"""SQLiteArchive round-trip tests."""

from __future__ import annotations

import pytest

from helix.archive import SQLiteArchive, SamplingStrategy
from helix.artifact import ArtifactKind, genesis, Subtype
from helix.search.base import Variant
from helix.signal import GapMeasurement, Preference


@pytest.mark.asyncio
async def test_legacy_v01_artifact_row_migrates_on_read():
    """A row written under v0.1 (kind='prompt', no subtype) reads back as a
    v0.2 (text, prompt) artifact via the on-read migration. SPEC §1.2, §14."""
    arc = SQLiteArchive(":memory:")
    arc._conn.execute(
        "INSERT INTO artifacts "
        "(artifact_id, version, kind, subtype, content, content_is_json, "
        " parent_id, parent_version, created_by, created_at, content_hash, "
        " artifact_metadata) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("legacy.p", 1, "prompt", None, "old content", 0, None, None,
         "human", "2025-01-01T00:00:00+00:00", "deadbeef", "{}"),
    )
    arc._conn.commit()

    v = await arc.by_id("legacy.p", 1)
    assert v is not None
    assert v.artifact.kind is ArtifactKind.TEXT
    assert v.artifact.subtype is Subtype.PROMPT
    assert v.artifact.layer == 1


@pytest.mark.asyncio
async def test_duplicate_content_collapses_within_id_and_notifies():
    """A mutation reproducing an earlier version's content collapses onto it,
    emits ArtifactDuplicate, and routes its measurement to the canonical
    version (SPEC §1.3, fix #3)."""
    from helix.observability.bus import EventBus

    bus = EventBus()
    arc = SQLiteArchive(":memory:", bus=bus)
    dups = []
    bus.subscribe("artifact_duplicate", lambda ev: dups.append(ev))

    a1 = genesis("p", Subtype.PROMPT, "same content")
    r1 = await arc.record(Variant(artifact=a1, parent=a1, search_method="human"),
                          GapMeasurement(score=0.5))
    assert r1.inserted and not r1.was_duplicate

    a2 = a1.mutate("same content", created_by="spo")  # v2, identical content
    r2 = await arc.record(Variant(artifact=a2, parent=a1, search_method="spo"),
                          GapMeasurement(score=0.8))
    assert r2.was_duplicate
    assert not r2.inserted
    assert r2.canonical_ref == ("p", 1)        # collapsed onto v1
    assert await arc.by_id("p", 2) is None      # v2 was never stored
    assert len(await arc.get_measurement_history("p", 1)) == 2  # both enrich v1
    assert len(dups) == 1 and dups[0].canonical_version == 1

    # A different id with identical content is NOT collapsed (within-id scope).
    b = genesis("q", Subtype.PROMPT, "same content")
    r3 = await arc.record(Variant(artifact=b, parent=b, search_method="human"),
                          GapMeasurement(score=0.3))
    assert r3.inserted and not r3.was_duplicate
    assert await arc.by_id("q", 1) is not None


@pytest.mark.asyncio
async def test_put_artifact_persists_with_dedup_and_returns_canonical():
    """put_artifact is the public persist-without-measurement path GEPA uses
    instead of private storage internals (fix #5), with the same dedup (#3)."""
    arc = SQLiteArchive(":memory:")
    a1 = genesis("p", Subtype.PROMPT, "body")
    r1 = await arc.put_artifact(a1)
    assert r1.inserted and r1.canonical_ref == ("p", 1)

    a2 = a1.mutate("body", created_by="gepa")  # identical content
    r2 = await arc.put_artifact(a2)
    assert r2.was_duplicate and r2.canonical_ref == ("p", 1)
    assert await arc.by_id("p", 2) is None


@pytest.mark.asyncio
async def test_best_and_measurements_for_signal_attach_the_right_measurement():
    """best(signal_id=...) and measurements_for_signal attach the measurement
    taken under that signal, not a signal-agnostic latest (fix #6)."""
    arc = SQLiteArchive(":memory:")
    a = genesis("p", Subtype.PROMPT, "hi")
    v = Variant(artifact=a, parent=a, search_method="human")
    await arc.record(v, GapMeasurement(score=0.9, signal_id="judge", signal_version=1))
    await arc.record(v, GapMeasurement(score=0.2, signal_id="metric", signal_version=1))

    best_judge = await arc.best(k=1, signal_id="judge")
    assert best_judge[0].measurement.signal_id == "judge"
    assert best_judge[0].measurement.score == 0.9

    best_metric = await arc.best(k=1, signal_id="metric")
    assert best_metric[0].measurement.signal_id == "metric"
    assert best_metric[0].measurement.score == 0.2

    for_judge = await arc.measurements_for_signal("judge")
    assert len(for_judge) == 1
    assert for_judge[0][1].signal_id == "judge"
    assert for_judge[0][1].score == 0.9


@pytest.mark.asyncio
async def test_composite_round_trips_through_archive():
    """A composite stores and reads back with its constituents and layer intact."""
    from helix.artifact import compose

    arc = SQLiteArchive(":memory:")
    p = genesis("p", Subtype.PLANNER, "plan it")
    m = genesis("m", ArtifactKind.MEMORY_ENTRY, {"k": 1})
    comp = compose("meta", [(p, "planner"), (m, "memory")], subtype=Subtype.METACOGNITION)
    await arc.record(
        Variant(artifact=comp, parent=comp, search_method="human"),
        GapMeasurement(score=0.5),
    )

    v = await arc.by_id("meta", 1)
    assert v is not None
    assert v.artifact.kind is ArtifactKind.COMPOSITE
    assert v.artifact.subtype is Subtype.METACOGNITION
    assert v.artifact.layer == 3
    assert v.artifact.constituent_refs == [("p", 1), ("m", 1)]


@pytest.mark.asyncio
async def test_record_and_retrieve_by_id():
    arc = SQLiteArchive(":memory:")
    a = genesis("p", Subtype.PROMPT, "hello")
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
    a = genesis("p", Subtype.PROMPT, "v1")
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
    a = genesis("p", Subtype.PROMPT, "v1")
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
    a = genesis("p", Subtype.PROMPT, "v1")
    b_kind = a.mutate("v2-a", created_by="spo")
    # Simulate a second-line fork by using a different id but pointing parent
    # back at a. In practice GEPA would do this when forking off a known seed.
    from helix.artifact import Artifact
    c_fork = Artifact(
        id="p_fork", version=1, kind=a.kind, subtype=a.subtype, content="v2-b",
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
    a = genesis("p", Subtype.PROMPT, "v1")
    b = a.mutate("v2", created_by="spo")
    await arc.record(Variant(artifact=a, parent=a, search_method="human"), GapMeasurement(score=0.3))
    await arc.record(Variant(artifact=b, parent=a, search_method="spo"), GapMeasurement(score=0.9))
    v = await arc.sample(SamplingStrategy.BEST)
    assert v.measurement.score == 0.9


@pytest.mark.asyncio
async def test_pareto_front_single_objective_returns_max():
    arc = SQLiteArchive(":memory:")
    a = genesis("p", Subtype.PROMPT, "v1")
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
    a = genesis("p", Subtype.PROMPT, "v1")
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
    a = genesis("p", Subtype.PROMPT, "seed")
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
    a = genesis("p", Subtype.PROMPT, "v1")
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
    a = genesis("p", Subtype.PROMPT, "v1")
    b = a.mutate("v2", created_by="spo")
    await arc.record(Variant(artifact=a, parent=a, search_method="human"), GapMeasurement(score=0.3))
    await arc.record(Variant(artifact=b, parent=a, search_method="spo"), GapMeasurement(score=0.6))
    metrics = await arc.diversity_metrics()
    assert metrics.n_variants == 2
    assert metrics.n_unique_content_hashes == 2
    assert metrics.score_range == (0.3, 0.6)


# ---------------- signal attribution (SPEC §5.2.1) ----------------


@pytest.mark.asyncio
async def test_record_persists_signal_identity_and_extra_fields():
    """signal_id, signal_version, raw_value, triggered survive a round trip."""
    arc = SQLiteArchive(":memory:")
    a = genesis("p", Subtype.PROMPT, "v1")
    m = GapMeasurement(
        score=0.7,
        signal_id="JudgeX:abc123",
        signal_version=2,
        triggered=True,
        raw_value=4200.0,
    )
    await arc.record(Variant(artifact=a, parent=a, search_method="human"), m)
    fetched = await arc.by_id("p", version=1)
    assert fetched is not None
    assert fetched.measurement.signal_id == "JudgeX:abc123"
    assert fetched.measurement.signal_version == 2
    assert fetched.measurement.triggered is True
    assert fetched.measurement.raw_value == 4200.0


@pytest.mark.asyncio
async def test_best_without_signal_id_returns_all_measurements():
    """The signal_id kwarg defaults to None; behavior matches pre-attribution."""
    arc = SQLiteArchive(":memory:")
    a = genesis("p", Subtype.PROMPT, "v1")
    b = a.mutate("v2", created_by="spo")
    await arc.record(
        Variant(artifact=a, parent=a, search_method="human"),
        GapMeasurement(score=0.3, signal_id="JudgeA:x"),
    )
    await arc.record(
        Variant(artifact=b, parent=a, search_method="spo"),
        GapMeasurement(score=0.8, signal_id="JudgeB:y"),
    )
    top = await arc.best(k=2)
    assert [v.measurement.score for v in top] == [0.8, 0.3]


@pytest.mark.asyncio
async def test_best_filters_by_signal_id():
    """When signal_id is provided, only matching measurements compete."""
    arc = SQLiteArchive(":memory:")
    a = genesis("p", Subtype.PROMPT, "v1")
    b = a.mutate("v2", created_by="spo")
    c = b.mutate("v3", created_by="spo")
    # a scored under judge A at 0.9 (would otherwise win)
    await arc.record(
        Variant(artifact=a, parent=a, search_method="human"),
        GapMeasurement(score=0.9, signal_id="JudgeA:x"),
    )
    # b and c scored under judge B
    await arc.record(
        Variant(artifact=b, parent=a, search_method="spo"),
        GapMeasurement(score=0.6, signal_id="JudgeB:y"),
    )
    await arc.record(
        Variant(artifact=c, parent=b, search_method="spo"),
        GapMeasurement(score=0.75, signal_id="JudgeB:y"),
    )
    # Filter to JudgeB: only b and c qualify; c wins with 0.75
    top = await arc.best(k=2, signal_id="JudgeB:y")
    assert len(top) == 2
    assert top[0].artifact.version == 3
    assert top[1].artifact.version == 2
    # Filter to JudgeA: only a qualifies
    top_a = await arc.best(k=5, signal_id="JudgeA:x")
    assert len(top_a) == 1
    assert top_a[0].artifact.version == 1


@pytest.mark.asyncio
async def test_measurements_for_signal_returns_matching_rows():
    arc = SQLiteArchive(":memory:")
    a = genesis("p", Subtype.PROMPT, "v1")
    b = a.mutate("v2", created_by="spo")
    await arc.record(
        Variant(artifact=a, parent=a, search_method="human"),
        GapMeasurement(score=0.4, signal_id="JudgeA:x", signal_version=1),
    )
    await arc.record(
        Variant(artifact=b, parent=a, search_method="spo"),
        GapMeasurement(score=0.7, signal_id="JudgeA:x", signal_version=1),
    )
    await arc.record(
        Variant(artifact=b, parent=a, search_method="spo"),
        GapMeasurement(score=0.5, signal_id="JudgeB:y", signal_version=1),
    )
    rows = await arc.measurements_for_signal("JudgeA:x")
    assert len(rows) == 2
    ids_versions = {(v.artifact.id, v.artifact.version) for v, _ in rows}
    assert ids_versions == {("p", 1), ("p", 2)}


@pytest.mark.asyncio
async def test_measurements_for_signal_filters_by_version():
    arc = SQLiteArchive(":memory:")
    a = genesis("p", Subtype.PROMPT, "v1")
    await arc.record(
        Variant(artifact=a, parent=a, search_method="human"),
        GapMeasurement(score=0.4, signal_id="JudgeA:x", signal_version=1),
    )
    b = a.mutate("v2", created_by="spo")
    await arc.record(
        Variant(artifact=b, parent=a, search_method="spo"),
        GapMeasurement(score=0.7, signal_id="JudgeA:x", signal_version=2),
    )
    only_v2 = await arc.measurements_for_signal("JudgeA:x", signal_version=2)
    assert len(only_v2) == 1
    assert only_v2[0][0].artifact.version == 2
    both = await arc.measurements_for_signal("JudgeA:x")
    assert len(both) == 2


@pytest.mark.asyncio
async def test_measurements_for_signal_empty_when_no_match():
    arc = SQLiteArchive(":memory:")
    a = genesis("p", Subtype.PROMPT, "v1")
    await arc.record(
        Variant(artifact=a, parent=a, search_method="human"),
        GapMeasurement(score=0.4, signal_id="JudgeA:x"),
    )
    rows = await arc.measurements_for_signal("NonExistent:zzz")
    assert rows == []
