"""Spec conformance for the Artifact primitive (Spec §1)."""

from __future__ import annotations

import pytest

from helix.artifact import Artifact, ArtifactKind, genesis, Subtype


def test_genesis_artifact_has_no_parent_and_version_one():
    art = genesis("prompt.test", Subtype.PROMPT, "hello")
    assert art.parent_id is None
    assert art.version == 1
    assert art.created_by == "human"


def test_mutate_increments_version_and_records_parent_pointer():
    art = genesis("prompt.test", Subtype.PROMPT, "hello")
    child = art.mutate("hello v2", created_by="spo")

    assert child.version == 2
    assert child.parent_id == ("prompt.test", 1)
    assert child.created_by == "spo"
    assert child.id == art.id
    assert child.kind == art.kind


def test_mutate_chains_preserve_full_lineage():
    a1 = genesis("prompt.test", Subtype.PROMPT, "v1")
    a2 = a1.mutate("v2", created_by="spo")
    a3 = a2.mutate("v3", created_by="gepa")

    assert a3.version == 3
    assert a3.parent_id == ("prompt.test", 2)
    assert a2.parent_id == ("prompt.test", 1)


def test_artifact_is_frozen_against_in_place_edit():
    art = genesis("prompt.test", Subtype.PROMPT, "hello")
    with pytest.raises(Exception):
        art.content = "this should fail"  # type: ignore[misc]


def test_content_hash_is_stable_for_identical_content():
    a = genesis("p", Subtype.PROMPT, "same content")
    b = genesis("p", Subtype.PROMPT, "same content")
    assert a.content_hash == b.content_hash


def test_content_hash_differs_when_content_differs():
    a = genesis("p", Subtype.PROMPT, "alpha")
    b = a.mutate("beta", created_by="human")
    assert a.content_hash != b.content_hash


def test_dict_content_hash_is_key_order_independent():
    a = Artifact(id="r", version=1, kind=ArtifactKind.TEXT, subtype=Subtype.RUBRIC, content={"a": 1, "b": 2})
    b = Artifact(id="r", version=1, kind=ArtifactKind.TEXT, subtype=Subtype.RUBRIC, content={"b": 2, "a": 1})
    assert a.content_hash == b.content_hash


# ---------------------------------------------------------------------------
# v0.2 kind/subtype model (SPEC §1.2, §18.2)
# ---------------------------------------------------------------------------

def test_genesis_from_subtype_infers_text_kind():
    art = genesis("p.x", Subtype.SKILL, "do a thing")
    assert art.kind is ArtifactKind.TEXT
    assert art.subtype is Subtype.SKILL
    assert art.layer == 1


def test_genesis_from_base_kind_has_no_subtype():
    art = genesis("c.x", ArtifactKind.CODE, "async def f(): ...")
    assert art.kind is ArtifactKind.CODE
    assert art.subtype is None
    assert art.layer == 4


def test_mutate_preserves_kind_and_subtype():
    art = genesis("t.x", Subtype.TOOL_DESCRIPTION, "describes a tool")
    child = art.mutate("describes it better", created_by="spo")
    assert child.kind is ArtifactKind.TEXT
    assert child.subtype is Subtype.TOOL_DESCRIPTION


def test_metacognition_composite_floors_at_l3_above_its_constituents():
    meta = genesis("meta", ArtifactKind.COMPOSITE, [], subtype=Subtype.METACOGNITION)
    assert meta.layer == 3  # the subtype floor lifts it above L2 constituents


def test_resolve_kind_rejects_subtype_that_does_not_belong_to_kind():
    with pytest.raises(ValueError):
        genesis("x", ArtifactKind.CODE, "...", subtype=Subtype.PROMPT)


def test_migrate_kind_maps_retired_v01_kinds_to_text_subtypes():
    from helix.artifact import migrate_kind

    assert migrate_kind("prompt") == (ArtifactKind.TEXT, Subtype.PROMPT)
    assert migrate_kind("planner") == (ArtifactKind.TEXT, Subtype.PLANNER)
    # a v0.2 row passes through unchanged
    assert migrate_kind("text", "skill") == (ArtifactKind.TEXT, Subtype.SKILL)
    assert migrate_kind("code", None) == (ArtifactKind.CODE, None)


# ---------------------------------------------------------------------------
# Composite artifacts (SPEC §18)
# ---------------------------------------------------------------------------

def test_compose_builds_metacognition_composite_floored_at_l3():
    from helix.artifact import compose

    planner = genesis("plan", Subtype.PLANNER, "decompose the task")
    monitor = genesis("mon", Subtype.MONITOR, "watch for loops")
    mem = genesis("mem", ArtifactKind.MEMORY_ENTRY, {"state": "ok"})
    meta = compose(
        "meta",
        [(planner, "planner"), (monitor, "monitor"), (mem, "memory")],
        subtype=Subtype.METACOGNITION,
    )
    assert meta.kind is ArtifactKind.COMPOSITE
    assert meta.subtype is Subtype.METACOGNITION
    assert meta.layer == 3  # subtype floor L3 beats the L2 constituent max
    assert meta.constituent_refs == [("plan", 1), ("mon", 1), ("mem", 1)]


def test_composite_layer_takes_max_of_floor_and_constituents():
    from helix.artifact import compose

    code = genesis("c", ArtifactKind.CODE, "async def f(): ...")
    prompt = genesis("p", Subtype.PROMPT, "hi")
    # No subtype floor, but an L4 code constituent makes the bundle L4.
    bundle = compose("b", [code, prompt])
    assert bundle.layer == 4


def test_composite_content_hash_is_stable_across_equal_bindings():
    from helix.artifact import compose

    p = genesis("p", Subtype.PROMPT, "hi")
    a = compose("x", [p], subtype=Subtype.METACOGNITION)
    b = compose("x", [p], subtype=Subtype.METACOGNITION)
    assert a.content_hash == b.content_hash
