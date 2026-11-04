"""The live event stream: cursor codec, backfill, and end-to-end delivery."""

from __future__ import annotations

import asyncio
from pathlib import Path

from helix_web.server import cursor as cursor_mod


def test_cursor_roundtrips_and_rejects_garbage():
    marks = {
        "artifacts": (5, 5),
        "measurements": (7, 7),
        "trajectories": (3, 3),
        "promotions": (1, 1),
    }
    encoded = cursor_mod.encode(marks, 42)
    assert cursor_mod.decode(encoded) == (marks, 42)
    assert cursor_mod.decode("nonsense") is None
    assert cursor_mod.decode("a1:1|m2:2") is None


def _parse_sse(text: str) -> list[tuple[str, str]]:
    events = []
    for block in text.split("\n\n"):
        lines = [l for l in block.splitlines() if l and not l.startswith(":")]
        etype = next((l[7:] for l in lines if l.startswith("event: ")), None)
        data = next((l[6:] for l in lines if l.startswith("data: ")), None)
        if etype:
            events.append((etype, data or ""))
    return events


def test_stream_says_hello_and_delivers_a_new_measurement(
    observatory_client, populated_archive: Path
):
    import json

    # Snapshot the current cursor, then append one artifact + measurement the
    # way a running loop would, and ask the stream to replay past the cursor.
    cur = None

    with observatory_client.stream(
        "GET", "/api/archives/helix_archive/events", params={"max_events": 0}
    ) as resp:
        text = "".join(resp.iter_text())
    events = _parse_sse(text)
    assert events[0][0] == "hello"
    hello = json.loads(events[0][1])
    cur = hello["cursor"]
    assert hello["archive"]["label"] == "helix_archive"

    async def append() -> None:
        from helix.archive.sqlite_backend import SQLiteArchive
        from helix.artifact import Subtype, genesis
        from helix.search.base import Variant
        from helix.signal import GapMeasurement

        archive = SQLiteArchive(str(populated_archive))
        seed = genesis(
            id="prompt.fixture.system2",
            kind=Subtype.PROMPT,
            content="A second family appears mid-stream.",
            created_by="human",
        )
        await archive.record(
            Variant(artifact=seed, parent=seed, search_method="human"),
            GapMeasurement(score=0.33),
        )

    asyncio.run(append())

    with observatory_client.stream(
        "GET",
        "/api/archives/helix_archive/events",
        params={"since": cur, "max_events": 0},
    ) as resp:
        text = "".join(resp.iter_text())
    replay = _parse_sse(text)
    types = [t for t, _ in replay]
    assert types[0] == "hello"
    assert "artifact" in types
    assert "measurement" in types
    artifact_ev = json.loads(next(d for t, d in replay if t == "artifact"))
    assert artifact_ev["node"]["id"] == "prompt.fixture.system2"
    assert artifact_ev["node"]["measurement"]["score"] == 0.33


def test_stream_resyncs_on_a_malformed_cursor(observatory_client):
    with observatory_client.stream(
        "GET",
        "/api/archives/helix_archive/events",
        params={"since": "garbage", "max_events": 0},
    ) as resp:
        text = "".join(resp.iter_text())
    events = _parse_sse(text)
    assert [t for t, _ in events][:2] == ["hello", "resync"]
