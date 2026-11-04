"""The studio scenario exercises real retrieval, hooks, and persisted learning."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

from chapter_appendices.ch05_studio_legacy.studio.corpus import CORPUS_DIR, StudioCorpus, embed
from importlib import import_module

support = import_module("chapter_appendices.ch05_studio_legacy.02_studio_agent")
SupportSimulator, load_tickets = support.SupportSimulator, support.load_tickets
from helix.memory.base import MemoryContext, MemoryEntry, MemoryQuery, Scope, ScopeKey
from helix.memory.episodic import EpisodicMemory


def load_walkthrough():
    path = Path(__file__).resolve().parents[1] / "chapter_appendices/ch05_studio_legacy/03_memory_learning.py"
    spec = importlib.util.spec_from_file_location("studio_walkthrough", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_numbered_agent_listing_runs_from_outside_the_repository(tmp_path):
    script = Path(__file__).resolve().parents[1] / "chapter_appendices/ch05_studio_legacy/02_studio_agent.py"
    result = subprocess.run(
        [sys.executable, str(script)], cwd=tmp_path,
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "Runbook: 01_orbitlink_vpn.md" in result.stdout
    assert "Observed: OrbitLink connected; the asset sync succeeded." in result.stdout
    assert "Reward: 1.0" in result.stdout


@pytest.mark.parametrize("query,source", [
    ("OrbitLink NET01 disconnected after sleep", "01_orbitlink_vpn.md"),
    ("CrateSync AUTH17 expired session refresh", "02_cratesync_assets.md"),
    ("BuildDock QUEUED job duplicate", "03_builddock_queue.md"),
    ("StageLink USB headset microphone input test", "04_stagelink_audio.md"),
    ("AccessDesk PERM04 project membership approval", "05_accessdesk_permissions.md"),
    ("handoff previous case observed outcome", "06_support_handoffs.md"),
])
def test_each_studio_problem_retrieves_an_approved_source(query, source):
    hits = StudioCorpus().search(query)
    assert source in [hit["source"] for hit in hits]
    for hit in hits:
        assert (CORPUS_DIR / hit["source"]).is_file()
        assert "expected_action" not in hit["text"]


def test_studio_search_handles_empty_queries_and_limits():
    corpus = StudioCorpus()
    assert corpus.search("the and") == []
    assert corpus.search("CrateSync", k=0) == []
    assert len(corpus.search("CrateSync", k=1)) == 1
    assert embed(["asset sync"])[0] == embed(["asset sync"])[0]


def test_simulated_resolution_is_distinct_from_a_successful_tool_call():
    simulator = SupportSimulator()
    simulator.begin(load_tickets()[2])
    assert simulator.reward(None) is None
    result = simulator.step("reconnect_vpn")
    assert not result["resolved"] and "AUTH17" in result["observed"]
    assert simulator.reward(None) == 0.0
    assert "error" in simulator.step("refresh_asset_token")
    simulator.begin(load_tickets()[3])
    assert simulator.step("refresh_asset_token")["resolved"]
    assert simulator.reward(None) == 1.0
    simulator.begin(load_tickets()[-1])
    assert not simulator.step("request_asset_access")["resolved"]
    assert simulator.reward(None) == 0.0


@pytest.mark.parametrize("rows,expected", [
    ([], {"requests": 0, "resolved": 0, "unresolved": 0,
          "unobserved": 0, "resolution_rate": None}),
    ([{"outcome": {"resolved": True}, "reward": 1.0},
      {"outcome": {"resolved": False}, "reward": 0.0},
      {"outcome": None, "reward": None}],
     {"requests": 3, "resolved": 1, "unresolved": 1,
      "unobserved": 1, "resolution_rate": 1 / 3}),
    ([{"outcome": {"resolved": False}, "reward": 1.0},
      {"outcome": {"resolved": True}, "reward": None},
      {"outcome": None, "reward": 1.0}],
     {"requests": 3, "resolved": 0, "unresolved": 1,
      "unobserved": 2, "resolution_rate": 0.0}),
])
def test_resolution_measurement_uses_outcomes_and_keeps_every_request(rows, expected):
    assert load_walkthrough().measure_resolution(rows) == expected


async def test_a_configured_encoder_is_used_for_writes_reads_and_reopening(tmp_path, monkeypatch):
    def forbidden_default(_):
        raise AssertionError("The configured encoder must avoid the default model download.")
    monkeypatch.setattr("helix.memory.episodic._embed", forbidden_default)
    path = tmp_path / "memory.sqlite"
    tier = EpisodicMemory(path, embedding_fn=embed, recency_weight=0)
    await tier.write(MemoryEntry(id="sync", scope_key=ScopeKey(Scope.USER, "maya"),
                                 content={"user_message": "CrateSync AUTH17", "final_output": "refresh session"}))
    reopened = EpisodicMemory(path, embedding_fn=embed, recency_weight=0)
    hits = await reopened.read(MemoryQuery(text="CrateSync AUTH17"), MemoryContext(user_id="maya"))
    assert [entry.id for entry in hits] == ["sync"]
    assert hits[0].embedding == embed(["CrateSync AUTH17\nrefresh session"])[0]


async def test_studio_walkthrough_credits_recalled_entries_and_reopens_them_for_the_same_user(tmp_path, monkeypatch, capsys):
    async def forbidden_provider(**_):
        raise AssertionError("A scripted walkthrough must not make provider calls.")
    monkeypatch.setattr("helix.llm_call.litellm.acompletion", forbidden_provider)
    module = load_walkthrough()
    report = await module.run_walkthrough(tmp_path / "run")
    rows = report["requests"]
    assert report["measurement"] == {
        "requests": 4, "resolved": 3, "unresolved": 1,
        "unobserved": 0, "resolution_rate": 0.75,
    }
    assert "Observed resolutions: 3/4 (75%); no outcome: 0" in capsys.readouterr().out
    assert [row["reward"] for row in rows] == [1.0, 1.0, 0.0, 1.0]
    assert rows[0]["recalled"] == []
    assert rows[1]["recalled"][0]["episode"] == "E1"
    assert rows[1]["recalled"][0]["q_before"] == 0.5
    assert rows[1]["recalled"][0]["q_after"] == 0.75
    assert rows[2]["recalled"][0]["episode"] == "E1"
    assert rows[2]["recalled"][0]["q_after"] == 0.375
    assert rows[3]["recalled"][0]["episode"] == "E3"
    assert rows[3]["recalled"][0]["q_after"] == 0.75
    assert "02_cratesync_assets.md" in rows[3]["answer"]
    for row in rows:
        assert [call["name"] for call in row["tool_calls"]] == ["retrieve", "support_step"]
        assert row["written"]["artifact_ref"][1] == 1
        for recalled in row["recalled"]:
            assert recalled["current_ref"][1] == recalled["recalled_ref"][1] + 1
            assert recalled["q_after"] == pytest.approx(recalled["q_before"] + .5 * (row["reward"] - recalled["q_before"]))
    assert len(report["handoff"]["maya"]["recalled"]) == 1
    assert "unresolved" in report["handoff"]["maya"]["answer"]
    assert "AUTH17" in report["handoff"]["maya"]["answer"]
    assert report["handoff"]["leo"]["recalled"] == []
    assert len(report["corpus_sha256"]) == 6
    saved = json.loads((tmp_path / "run/trace.json").read_text())
    assert saved["configuration"]["algorithm"] == "one_step_value_update"
    assert saved["measurement"] == report["measurement"]
    with pytest.raises(ValueError, match="existing evidence"):
        await module.run_walkthrough(tmp_path / "run")


async def test_live_mode_requires_an_explicit_model_before_creating_files(tmp_path):
    with pytest.raises(ValueError, match="explicit model"):
        await load_walkthrough().run_walkthrough(tmp_path / "live", live=True)
    assert not (tmp_path / "live").exists()
