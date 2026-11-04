"""Shared fixtures: a fully populated temp archive and an Observatory client.

The populated_archive fixture builds the smallest archive that exercises every
shape the dashboard renders: a genesis, four search children including a
crossover with a second parent, a promotion, and three trajectory flavors.
Nothing here is autouse, so existing tests are unaffected.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from helix.archive.sqlite_backend import SQLiteArchive
from helix.artifact import Subtype, genesis
from helix.search.base import Variant
from helix.signal import GapMeasurement


class _StubTrajectory:
    """Minimal object satisfying store_trajectory's to_dict contract."""

    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def to_dict(self) -> dict:
        return self._payload


async def _build(path: Path) -> None:
    archive = SQLiteArchive(str(path))

    seed = genesis(
        id="prompt.fixture.system",
        kind=Subtype.PROMPT,
        content="You are a careful assistant.",
        created_by="human",
    )
    await archive.record(
        Variant(artifact=seed, parent=seed, search_method="human"),
        GapMeasurement(score=0.40, feedback="baseline"),
    )

    v2 = seed.mutate("Assistant, round 2.", created_by="spo_round_1", version=2)
    await archive.record(
        Variant(artifact=v2, parent=seed, search_method="spo"),
        GapMeasurement(
            score=0.55,
            feedback="better tone",
            metadata={
                "role": "candidate",
                "per_question": [
                    {
                        "question_id": "q1",
                        "band": 1,
                        "preference": "left",
                        "confidence": 0.9,
                        "feedback": "candidate answered cleanly",
                    },
                    {
                        "question_id": "q2",
                        "band": 2,
                        "preference": "tie",
                        "confidence": 0.5,
                        "feedback": "both adequate",
                    },
                ],
            },
        ),
    )

    v3 = v2.mutate("Assistant, reflected fix.", created_by="gepa_mutation", version=3)
    await archive.record(
        Variant(artifact=v3, parent=v2, search_method="gepa"),
        GapMeasurement(score=0.70),
    )

    v4 = v3.mutate(
        "Assistant, merged best of both.",
        created_by="gepa_crossover",
        metadata={"other_parent_version": 2},
        version=4,
    )
    await archive.record(
        Variant(artifact=v4, parent=v3, search_method="gepa"),
        GapMeasurement(score=0.80),
    )

    v5 = v2.mutate("Assistant, archive branch.", created_by="dgm_round_3", version=5)
    await archive.record(
        Variant(artifact=v5, parent=v2, search_method="dgm"),
        GapMeasurement(score=0.92),
    )

    await archive.promote(
        "prompt.fixture.system", 2, approver="tests", reason="fixture promotion"
    )

    await archive.store_trajectory(
        v2,
        "q1",
        _StubTrajectory(
            {
                "task": "answer q1",
                "outcome": "completed",
                "final_output": "The answer is 42.",
                "steps": [
                    {"kind": "model_call", "payload": {"content": "The answer is 42."}},
                ],
                "metadata": {},
            }
        ),
        answer="The answer is 42.",
    )
    # Zero steps but a final output: not failed for display, yet purgeable.
    # This is the case where the two predicates disagree on purpose.
    await archive.store_trajectory(
        v2,
        "q2",
        _StubTrajectory(
            {
                "task": "answer q2",
                "outcome": "completed",
                "final_output": "Cached answer.",
                "steps": [],
                "metadata": {},
            }
        ),
        answer="Cached answer.",
    )
    await archive.store_trajectory(
        v3,
        "q1",
        _StubTrajectory(
            {
                "task": "answer q1",
                "outcome": "failed",
                "final_output": None,
                "steps": [],
                "metadata": {"error": "RateLimitError: too many requests"},
            }
        ),
        answer=None,
    )


async def _add_memory(path: Path) -> None:
    """Layer two MEMORY_ENTRY artifacts onto the fixture archive (chapter 5).

    One episodic entry gets an upvote child version plus a utility measurement
    so the memory panel has a chain and a history to render; one semantic
    entry stays at genesis. Kept out of _build so the counts other web tests
    assert on the base archive do not move.
    """
    from helix.memory.archive_bridge import record_genesis, record_version
    from helix.memory.base import MemoryEntry, Scope, ScopeKey

    archive = SQLiteArchive(str(path))
    episodic = MemoryEntry(
        id="e1",
        scope_key=ScopeKey(Scope.USER, "alice"),
        content={"user_message": "vpn keeps dropping", "final_output": "use the wired profile"},
        score=0.5,
    )
    ref = await record_genesis(archive, "episodic", episodic)
    parent = (await archive.by_id(ref[0])).artifact
    upvoted = dict(parent.content)
    upvoted["utility"] = 0.8
    child_ref = await record_version(archive, parent, upvoted, "memrl", "upvote")
    child = (await archive.by_id(child_ref[0], child_ref[1])).artifact
    await archive.record(
        Variant(artifact=child, parent=parent, search_method="memrl"),
        GapMeasurement(score=0.8, signal_id="utility", feedback="recalled and helped"),
    )

    semantic = MemoryEntry(
        id="f1",
        scope_key=ScopeKey(Scope.ORG, "acme"),
        content={"key": "vpn_profile", "value": "wired"},
        score=0.3,
    )
    await record_genesis(archive, "semantic", semantic)


@pytest.fixture
def memory_archive(tmp_path: Path) -> Path:
    """The populated archive plus memory-entry artifacts for the memory panel."""
    path = tmp_path / "helix_archive.sqlite"
    asyncio.run(_build(path))
    asyncio.run(_add_memory(path))
    return path


@pytest.fixture
def memory_observatory_client(memory_archive: Path):
    """A TestClient over an app whose registry sees only the memory archive."""
    from fastapi.testclient import TestClient

    from helix_web.server.app import create_app
    from helix_web.server.config import Settings

    settings = Settings(
        repo_root=memory_archive.parent,
        scan_globs=("*.sqlite",),
        web_dir=memory_archive.parent / "no-web-dir",
    )
    with TestClient(create_app(settings)) as client:
        yield client


@pytest.fixture
def populated_archive(tmp_path: Path) -> Path:
    """A helix_archive.sqlite covering every shape the dashboard renders."""
    path = tmp_path / "helix_archive.sqlite"
    asyncio.run(_build(path))
    return path


@pytest.fixture
def observatory_client(populated_archive: Path):
    """A TestClient over an app whose registry sees only the temp archive."""
    from fastapi.testclient import TestClient

    from helix_web.server.app import create_app
    from helix_web.server.config import Settings

    settings = Settings(
        repo_root=populated_archive.parent,
        scan_globs=("*.sqlite",),
        web_dir=populated_archive.parent / "no-web-dir",
    )
    with TestClient(create_app(settings)) as client:
        yield client
