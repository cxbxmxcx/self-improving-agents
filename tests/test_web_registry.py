"""Archive registry and discovery endpoints of the Observatory."""

from __future__ import annotations

import re
from pathlib import Path

from helix_web.server.config import REPO_ROOT, Settings
from helix_web.server.registry import ArchiveRegistry, key_for


def test_registry_discovers_the_chapter_run_archives():
    registry = ArchiveRegistry(Settings())
    registry.discover()
    keys = registry.keys()
    for expected in (
        "chapters.ch02.runs.helix_archive",
        "chapters.ch03.runs.revenue_hillclimb",
        "chapters.ch03.runs.revenue_spo",
        "chapters.ch04.runs.revenue_dgm",
        "chapters.ch04.runs.revenue_gepa",
    ):
        assert expected in keys


def test_registry_keys_are_stable_dotted_relative_paths(tmp_path: Path):
    inside = REPO_ROOT / "chapters" / "ch03" / "runs" / "revenue_spo.sqlite"
    assert key_for(inside, REPO_ROOT) == "chapters.ch03.runs.revenue_spo"
    outside = tmp_path / "other.sqlite"
    key = key_for(outside, REPO_ROOT)
    assert re.fullmatch(r"other-[0-9a-f]{8}", key)


def test_api_lists_the_fixture_archive_with_counts(observatory_client):
    payload = observatory_client.get("/api/archives").json()
    rows = payload["archives"]
    assert len(rows) == 1
    row = rows[0]
    assert row["key"] == "helix_archive"
    assert row["valid"] is True
    assert row["n_artifacts"] == 5
    assert row["n_measured"] == 5
    assert row["top_score"] == 0.92


def test_adding_a_non_archive_file_is_rejected(observatory_client, tmp_path: Path):
    junk = tmp_path / "junk.sqlite"
    junk.write_text("this is not a database")
    resp = observatory_client.post("/api/archives", json={"path": str(junk)})
    assert resp.status_code == 400


def test_health_reports_the_archive_count(observatory_client):
    body = observatory_client.get("/api/health").json()
    assert body["ok"] is True
    assert body["n_archives"] == 1


def test_read_layer_never_imports_sqlitearchive():
    server_dir = Path(__file__).resolve().parent.parent / "helix_web" / "server"
    import_pattern = re.compile(
        r"^\s*(from\s+helix\.archive\b.*import|import\s+helix\.archive\b)", re.M
    )
    offenders = []
    for path in server_dir.rglob("*.py"):
        if path.name == "writes.py":
            continue
        if import_pattern.search(path.read_text(encoding="utf-8")):
            offenders.append(path.name)
    assert offenders == []
