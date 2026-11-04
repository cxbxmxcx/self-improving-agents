"""Run the chapter's learner wiring without model calls or historical writes."""

import importlib.util
import json
from pathlib import Path

import pytest

from chapter_appendices.ch05_research_legacy import common
from helix.archive import SQLiteArchive


@pytest.mark.parametrize("filename,stem,output,rows", [
    ("04_memrl_search.py", "memrl_search", "comparison", 60),
    ("06_online_memory_loop.py", "online_memory", "stream", 40),
])
async def test_chapter_runs_learn_through_the_harness_in_an_isolated_store(
    filename, stem, output, rows, tmp_path, monkeypatch, capsys,
):
    monkeypatch.setattr(common, "RUNS_DIR", tmp_path)
    monkeypatch.setattr(common, "SMOKE", True)
    monkeypatch.setattr("helix.memory.episodic._embed", lambda texts: [[1.0, 0.0] for _ in texts])
    path = Path(__file__).resolve().parents[1] / "chapter_appendices/ch05_research_legacy" / filename
    spec = importlib.util.spec_from_file_location("chapter_memory_harness_demo", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with common.maybe_smoke():
        await module.main()

    records = [json.loads(line) for line in (tmp_path/f"{stem}_{output}.jsonl").read_text().splitlines()]
    assert len(records) == rows
    archive = SQLiteArchive(tmp_path/f"{stem}.sqlite")
    learned = [v for v in await archive.all_artifacts_with_measurements()
               if v.measurement and v.measurement.metadata.get("learning_algorithm") == "one_step_value_update"]
    assert learned, "the script must drive the learner, not merely write new interactions"
    assert all(v.measurement.metadata["alpha"] == module.ALPHA for v in learned)
    assert module.UTILITY_WEIGHT == 0.35
    console = capsys.readouterr().out
    assert "[smoke]" in console
    if filename.startswith("06"):
        assert "40 trajectories, 40 updates" in console
