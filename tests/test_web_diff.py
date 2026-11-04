"""Diff endpoints return opcode payloads a client can render."""

from __future__ import annotations


def test_content_diff_reports_opcodes_and_stats(observatory_client):
    body = observatory_client.get(
        "/api/archives/helix_archive/diff",
        params={
            "a_id": "prompt.fixture.system",
            "a_version": 1,
            "b_id": "prompt.fixture.system",
            "b_version": 2,
        },
    ).json()
    assert body["a"]["label"] == "prompt.fixture.system v1"
    assert body["a"]["lines"] == ["You are a careful assistant."]
    assert body["b"]["lines"] == ["Assistant, round 2."]
    assert body["opcodes"][0]["tag"] == "replace"
    assert body["stats"] == {"added": 1, "removed": 1, "unchanged": 0}
    assert body["truncated"] is False


def test_content_diff_404s_on_a_missing_side(observatory_client):
    resp = observatory_client.get(
        "/api/archives/helix_archive/diff",
        params={
            "a_id": "prompt.fixture.system",
            "a_version": 1,
            "b_id": "prompt.fixture.system",
            "b_version": 99,
        },
    )
    assert resp.status_code == 404


def test_trajectory_diff_compares_final_outputs(observatory_client):
    body = observatory_client.get(
        "/api/archives/helix_archive/diff/trajectory",
        params={
            "artifact_id": "prompt.fixture.system",
            "a_version": 2,
            "b_version": 2,
            "question_id": "q1",
        },
    ).json()
    assert body["a"]["lines"] == body["b"]["lines"] == ["The answer is 42."]
    assert body["opcodes"] == [{"tag": "equal", "a1": 0, "a2": 1, "b1": 0, "b2": 1}]
