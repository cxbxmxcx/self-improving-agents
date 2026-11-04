"""Core read endpoints of the Observatory against the fixture archive."""

from __future__ import annotations

import pytest

from helix_web.server.normalize import parse_ref, ref


def test_ref_roundtrips_and_rejects_malformed_values():
    assert parse_ref(ref("prompt.fixture.system", 4)) == ("prompt.fixture.system", 4)
    with pytest.raises(ValueError):
        parse_ref("no-version-marker")


def test_summary_counts_the_fixture_archive(observatory_client):
    body = observatory_client.get("/api/archives/helix_archive/summary").json()
    totals = body["totals"]
    assert totals["artifacts"] == 5
    assert totals["measured"] == 5
    assert totals["top_score"] == 0.92
    assert totals["trajectories"] == 3
    assert totals["failed"] == 1
    assert totals["purgeable"] == 2
    assert body["failed_by_type"][0]["error_type"] == "RateLimitError"
    assert "rate limit" in body["failed_by_type"][0]["likely_cause"].lower()


def test_variant_list_marks_the_champion_and_live_version(observatory_client):
    body = observatory_client.get("/api/archives/helix_archive/variants").json()
    assert body["total"] == 5
    by_version = {v["version"]: v for v in body["items"]}
    assert by_version[5]["is_champion"] is True
    assert by_version[4]["is_champion"] is False
    assert all(v["live_version"] == 2 for v in body["items"])
    assert by_version[2]["search_method"] == "spo"


def test_variant_detail_returns_content_and_unified_history(observatory_client):
    body = observatory_client.get(
        "/api/archives/helix_archive/variants/prompt.fixture.system/2"
    ).json()
    assert body["content"] == "Assistant, round 2."
    assert body["is_best"] is False
    history = body["measurement_history"]
    assert len(history) == 1
    assert history[0]["score"] == 0.55
    assert history[0]["cost"] == {"tokens": 0, "dollars": 0.0, "wall_clock_sec": 0.0}
    assert history[0]["metadata"]["per_question"][0]["question_id"] == "q1"


def test_lineage_forest_nests_children_and_reports_the_crossover_edge(
    observatory_client,
):
    body = observatory_client.get("/api/archives/helix_archive/lineage").json()
    assert body["stats"] == {
        "n_nodes": 5,
        "n_roots": 1,
        "max_depth": 4,
        "truncated": False,
    }
    root = body["roots"][0]
    assert root["version"] == 1
    child_versions = sorted(c["version"] for c in root["children"])
    assert child_versions == [2]
    v2 = root["children"][0]
    assert sorted(c["version"] for c in v2["children"]) == [3, 5]
    assert body["extra_edges"] == [
        {
            "from": "prompt.fixture.system__v2",
            "to": "prompt.fixture.system__v4",
            "type": "crossover",
        }
    ]
    assert body["champions"] == ["prompt.fixture.system__v5"]


def test_champions_endpoint_reports_a_pending_promotion(observatory_client):
    body = observatory_client.get("/api/archives/helix_archive/champions").json()
    assert len(body["families"]) == 1
    fam = body["families"][0]
    assert fam["best"]["version"] == 5
    assert fam["live_version"] == 2
    assert fam["pending"] is True
    assert [p["version"] for p in fam["lineage_path"]] == [1, 2, 5]
    assert fam["promotions"][0]["approver"] == "tests"
    assert fam["top_wins"] == []


def test_etag_lets_the_client_skip_an_unchanged_payload(observatory_client):
    first = observatory_client.get("/api/archives/helix_archive/lineage")
    etag = first.headers["etag"]
    second = observatory_client.get(
        "/api/archives/helix_archive/lineage", headers={"If-None-Match": etag}
    )
    assert second.status_code == 304


def test_unknown_archive_and_missing_variant_return_404(observatory_client):
    assert observatory_client.get("/api/archives/nope/summary").status_code == 404
    assert (
        observatory_client.get(
            "/api/archives/helix_archive/variants/prompt.fixture.system/99"
        ).status_code
        == 404
    )
