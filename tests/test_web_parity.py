"""Parity of the new read layer against the real chapter archives.

These tests pin the Observatory to what the recorded runs actually contain:
the populated ch03 SPO run and the hollow ch04 GEPA run, which predates the
loops' switch to archive.record and so has artifacts only.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from helix_web.server.app import create_app
from helix_web.server.config import REPO_ROOT, Settings

SPO_ARCHIVE = REPO_ROOT / "chapters" / "ch03" / "runs" / "revenue_spo.sqlite"
GEPA_ARCHIVE = REPO_ROOT / "chapters" / "ch04" / "runs" / "revenue_gepa.sqlite"


@pytest.fixture(scope="module")
def repo_client():
    with TestClient(create_app(Settings())) as client:
        yield client


@pytest.mark.skipif(not SPO_ARCHIVE.exists(), reason="ch03 SPO run not present")
def test_the_spo_run_reads_with_scores_and_a_champion(repo_client):
    body = repo_client.get(
        "/api/archives/chapters.ch03.runs.revenue_spo/lineage"
    ).json()
    assert body["stats"]["n_nodes"] == 11
    assert body["stats"]["n_roots"] == 1
    assert len(body["champions"]) == 1
    root = body["roots"][0]
    assert root["created_by"] == "human"
    assert root["measurement"]["score"] == pytest.approx(0.6308, abs=1e-3)


@pytest.mark.skipif(not GEPA_ARCHIVE.exists(), reason="ch04 GEPA run not present")
def test_the_regenerated_gepa_run_reads_with_scores_and_crossovers(repo_client):
    body = repo_client.get(
        "/api/archives/chapters.ch04.runs.revenue_gepa/lineage"
    ).json()
    assert body["stats"]["n_roots"] == 1
    assert len(body["champions"]) == 1
    root = body["roots"][0]
    assert root["created_by"] == "human"
    assert root["measurement"]["score"] is not None
    assert any(e["type"] == "crossover" for e in body["extra_edges"])


def test_an_artifacts_only_archive_degrades_without_errors(tmp_path):
    import asyncio

    from fastapi.testclient import TestClient

    from helix.archive.sqlite_backend import SQLiteArchive
    from helix.artifact import Subtype, genesis

    path = tmp_path / "hollow.sqlite"

    async def build() -> None:
        archive = SQLiteArchive(str(path))
        seed = genesis(
            id="prompt.hollow.system",
            kind=Subtype.PROMPT,
            content="recorded before measurements existed",
            created_by="human",
        )
        await archive.put_artifact(seed)
        await archive.put_artifact(
            seed.mutate("older loop, artifacts only", created_by="gepa", version=2)
        )

    asyncio.run(build())

    from helix_web.server.app import create_app
    from helix_web.server.config import Settings

    settings = Settings(
        repo_root=tmp_path, scan_globs=("*.sqlite",), web_dir=tmp_path / "no-web"
    )
    with TestClient(create_app(settings)) as client:
        body = client.get("/api/archives/hollow/lineage").json()
        assert body["stats"]["n_nodes"] == 2
        assert body["champions"] == []
        assert body["roots"][0]["measurement"] is None
        assert body["roots"][0]["search_method"] == "unknown"
        summary = client.get("/api/archives/hollow/summary").json()
        assert summary["totals"]["measured"] == 0
        assert summary["totals"]["top_score"] is None
        assert summary["champion"] is None
