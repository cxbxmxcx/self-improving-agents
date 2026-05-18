"""Spec conformance for the Artifact primitive (Spec §1)."""

from __future__ import annotations

import pytest

from helix.artifact import Artifact, ArtifactKind, genesis


def test_genesis_artifact_has_no_parent_and_version_one():
    art = genesis("prompt.test", ArtifactKind.PROMPT, "hello")
    assert art.parent_id is None
    assert art.version == 1
    assert art.created_by == "human"


def test_mutate_increments_version_and_records_parent_pointer():
    art = genesis("prompt.test", ArtifactKind.PROMPT, "hello")
    child = art.mutate("hello v2", created_by="spo")

    assert child.version == 2
    assert child.parent_id == ("prompt.test", 1)
    assert child.created_by == "spo"
    assert child.id == art.id
    assert child.kind == art.kind


def test_mutate_chains_preserve_full_lineage():
    a1 = genesis("prompt.test", ArtifactKind.PROMPT, "v1")
    a2 = a1.mutate("v2", created_by="spo")
    a3 = a2.mutate("v3", created_by="gepa")

    assert a3.version == 3
    assert a3.parent_id == ("prompt.test", 2)
    assert a2.parent_id == ("prompt.test", 1)


def test_artifact_is_frozen_against_in_place_edit():
    art = genesis("prompt.test", ArtifactKind.PROMPT, "hello")
    with pytest.raises(Exception):
        art.content = "this should fail"  # type: ignore[misc]


def test_content_hash_is_stable_for_identical_content():
    a = genesis("p", ArtifactKind.PROMPT, "same content")
    b = genesis("p", ArtifactKind.PROMPT, "same content")
    assert a.content_hash == b.content_hash


def test_content_hash_differs_when_content_differs():
    a = genesis("p", ArtifactKind.PROMPT, "alpha")
    b = a.mutate("beta", created_by="human")
    assert a.content_hash != b.content_hash


def test_dict_content_hash_is_key_order_independent():
    a = Artifact(id="r", version=1, kind=ArtifactKind.RUBRIC, content={"a": 1, "b": 2})
    b = Artifact(id="r", version=1, kind=ArtifactKind.RUBRIC, content={"b": 2, "a": 1})
    assert a.content_hash == b.content_hash
