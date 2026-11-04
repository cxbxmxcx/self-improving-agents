"""Verdicts, trajectories, and rounds payloads for the remaining panels."""

from __future__ import annotations


def test_verdicts_flatten_per_question_metadata_with_facets(observatory_client):
    body = observatory_client.get("/api/archives/helix_archive/verdicts").json()
    assert body["grand_total"] == 2
    assert body["facets"]["preferences"] == ["left", "tie"]
    assert body["counts_by_preference"] == {"left": 1, "tie": 1}
    q1 = next(r for r in body["items"] if r["question_id"] == "q1")
    assert q1["version"] == 2
    assert q1["candidate_failed"] is False
    assert q1["role"] == "candidate"


def test_verdicts_filter_server_side_by_preference(observatory_client):
    body = observatory_client.get(
        "/api/archives/helix_archive/verdicts", params={"preference": "left"}
    ).json()
    assert body["total"] == 1
    assert body["items"][0]["preference"] == "left"
    assert body["grand_total"] == 2


def test_verdict_heatmap_keeps_the_latest_cell_per_question_and_version(
    observatory_client,
):
    body = observatory_client.get(
        "/api/archives/helix_archive/verdicts/heatmap"
    ).json()
    assert body["questions"] == ["q1", "q2"]
    assert body["versions"] == [2]
    assert body["cells"]["q1|2"] == "left"


def test_trajectory_list_reports_the_failed_purgeable_split(observatory_client):
    body = observatory_client.get("/api/archives/helix_archive/trajectories").json()
    assert body["total"] == 3
    assert body["n_failed"] == 1
    assert body["n_purgeable"] == 2
    failed = next(r for r in body["items"] if r["status"] == "failed")
    assert failed["error_type"] == "RateLimitError"


def test_trajectory_detail_returns_steps_and_classification(observatory_client):
    body = observatory_client.get(
        "/api/archives/helix_archive/trajectories/prompt.fixture.system/2/q1"
    ).json()
    assert body["status"] == "ok"
    assert body["final_output"] == "The answer is 42."
    assert body["n_steps"] == 1
    assert body["steps"][0]["kind"] == "model_call"


def test_rounds_fall_back_to_an_archive_series_when_no_jsonl_exists(
    observatory_client,
):
    body = observatory_client.get("/api/archives/helix_archive/rounds").json()
    assert body["source"] == "archive"
    assert body["rounds"] == []
    series = body["fallback_series"]
    assert [s["version"] for s in series] == [1, 2, 3, 4, 5]
    assert body["summary"]["best_score"] == 0.92


def test_rounds_prefer_the_jsonl_side_channel(observatory_client, populated_archive):
    jsonl = populated_archive.parent / "spo_rounds.jsonl"
    jsonl.write_text(
        '{"round": 1, "mean_score": 0.5, "promoted": false}\n'
        "not json at all\n"
        '{"round": 2, "mean_score": 0.7, "promoted": true}\n',
        encoding="utf-8",
    )
    body = observatory_client.get("/api/archives/helix_archive/rounds").json()
    assert body["source"] == "jsonl"
    assert [r["round"] for r in body["rounds"]] == [1, 2]
    assert body["summary"]["n_promotions"] == 1
    assert body["summary"]["best_score"] == 0.7
