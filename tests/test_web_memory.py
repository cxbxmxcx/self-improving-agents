"""The Observatory memory endpoints over MEMORY_ENTRY artifacts (chapter 5)."""

from __future__ import annotations

import re
from pathlib import Path

from helix_web.server.routers.memory import preview_of


def test_memory_router_never_imports_the_archive_class():
    source = Path("helix_web/server/routers/memory.py").read_text(encoding="utf-8")
    assert not re.search(r"SQLiteArchive", source)


def test_memory_list_folds_each_entry_to_its_latest_version_ranked_by_utility(
    memory_observatory_client,
):
    body = memory_observatory_client.get("/api/archives/helix_archive/memory").json()
    assert body["total"] == 2
    assert body["grand_total"] == 2
    assert body["facets"] == {
        "tiers": {"episodic": 1, "semantic": 1},
        "scopes": {"user:alice": 1, "org:acme": 1},
    }
    first, second = body["items"]
    assert first["artifact_id"] == "memory.episodic.e1"
    assert first["entry_id"] == "e1"
    assert first["version"] == 2
    assert first["n_versions"] == 2
    assert first["utility"] == 0.8
    assert first["operator"] == "upvote"
    assert first["created_by"] == "memrl"
    assert first["parent"] == {"id": "memory.episodic.e1", "version": 1}
    assert first["latest_measurement"]["signal_id"] == "utility"
    assert first["latest_measurement"]["score"] == 0.8
    assert first["preview"] == "vpn keeps dropping"
    assert second["artifact_id"] == "memory.semantic.f1"
    assert second["n_versions"] == 1
    assert second["operator"] == "add"
    assert second["parent"] is None
    assert second["latest_measurement"] is None
    assert second["preview"] == "vpn_profile: wired"


def test_memory_list_filters_by_tier_but_keeps_whole_archive_facets(
    memory_observatory_client,
):
    body = memory_observatory_client.get(
        "/api/archives/helix_archive/memory", params={"tier": "semantic"}
    ).json()
    assert body["total"] == 1
    assert body["grand_total"] == 2
    assert body["items"][0]["tier"] == "semantic"
    assert body["facets"]["tiers"] == {"episodic": 1, "semantic": 1}


def test_memory_list_is_empty_for_an_archive_without_memory_entries(
    observatory_client,
):
    body = observatory_client.get("/api/archives/helix_archive/memory").json()
    assert body["items"] == []
    assert body["grand_total"] == 0
    assert body["facets"] == {"tiers": {}, "scopes": {}}


def test_memory_detail_returns_the_version_chain_and_utility_history(
    memory_observatory_client,
):
    body = memory_observatory_client.get(
        "/api/archives/helix_archive/memory/memory.episodic.e1"
    ).json()
    assert body["content"]["entry"]["user_message"] == "vpn keeps dropping"
    assert body["content"]["utility"] == 0.8
    assert [v["version"] for v in body["versions"]] == [1, 2]
    assert [v["operator"] for v in body["versions"]] == ["add", "upvote"]
    assert [v["utility"] for v in body["versions"]] == [0.5, 0.8]
    assert body["versions"][1]["parent"] == {"id": "memory.episodic.e1", "version": 1}
    assert body["versions"][0]["parent"] is None
    assert "content" not in body["versions"][0]
    assert len(body["utility_history"]) == 1
    assert body["utility_history"][0]["feedback"] == "recalled and helped"


def test_memory_detail_404s_for_an_unknown_entry(memory_observatory_client):
    resp = memory_observatory_client.get(
        "/api/archives/helix_archive/memory/memory.episodic.missing"
    )
    assert resp.status_code == 404


def test_preview_prefers_insight_then_falls_back_to_key_value_and_name():
    assert preview_of({"insight": "prefer wired", "user_message": "x"}) == "prefer wired"
    assert preview_of({"key": "k", "value": "v"}) == "k: v"
    assert preview_of({"name": "summarize", "code": "..."}) == "summarize"
    assert preview_of({}) == ""
