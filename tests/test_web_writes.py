"""The Observatory's two write operations against a temp archive."""

from __future__ import annotations


def test_promote_records_a_promotion_and_flips_the_cursor(observatory_client):
    before = observatory_client.get("/api/archives/helix_archive/lineage").headers[
        "etag"
    ]
    resp = observatory_client.post(
        "/api/archives/helix_archive/promote",
        json={
            "artifact_id": "prompt.fixture.system",
            "version": 5,
            "approver": "tests",
            "reason": "crowning the dgm branch",
        },
    )
    assert resp.status_code == 200
    assert resp.json()["version"] == 5
    after = observatory_client.get("/api/archives/helix_archive/lineage").headers[
        "etag"
    ]
    assert after != before
    champs = observatory_client.get("/api/archives/helix_archive/champions").json()
    fam = champs["families"][0]
    assert fam["live_version"] == 5
    assert fam["pending"] is False


def test_promoting_a_missing_version_is_a_client_error(observatory_client):
    resp = observatory_client.post(
        "/api/archives/helix_archive/promote",
        json={"artifact_id": "prompt.fixture.system", "version": 99},
    )
    assert resp.status_code == 400


def test_purge_requires_confirmation_then_deletes_purgeable_rows(
    observatory_client,
):
    refused = observatory_client.post(
        "/api/archives/helix_archive/trajectories/purge", json={}
    )
    assert refused.status_code == 400

    done = observatory_client.post(
        "/api/archives/helix_archive/trajectories/purge", json={"confirm": True}
    )
    assert done.status_code == 200
    # The backend's predicate also removes the zero-step row that still had
    # a final output, which is exactly the display-vs-purge gap we surface.
    assert done.json()["deleted"] == 2

    remaining = observatory_client.get(
        "/api/archives/helix_archive/trajectories"
    ).json()
    assert remaining["total"] == 1
    assert remaining["n_failed"] == 0
