"""Trajectory replay (JSONL load + pretty-print) tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from helix.replay import load_run, load_trajectories, pretty_print
from helix.trajectory import StepKind, Trajectory


def _make_run_file(tmp_path: Path) -> Path:
    t = Trajectory(task="What is GEPA?")
    t.append(StepKind.MODEL_CALL, {"response": {"content": "", "tool_calls": [{"function": {"name": "retrieve"}}]}})
    t.append(StepKind.TOOL_CALL, {"name": "retrieve", "arguments": {"query": "GEPA"}})
    t.append(StepKind.TOOL_RESULT, {"result": [{"source": "gepa.pdf", "page": 1, "text": "...", "score": 0.9}]})
    t.append(StepKind.MODEL_CALL, {"response": {"content": "GEPA stands for...", "tool_calls": None}})
    t.complete("GEPA stands for genetic-pareto reflective mutation.")

    record = {
        "question_id": "Q3",
        "band": 1,
        "question": "What is GEPA?",
        "reference_answer": "Genetic-Pareto reflective mutation.",
        "expected_failure_mode_v0": "",
        "tags": [],
        "agent_label": "v0",
        "answer": t.final_output,
        "trajectory": t.to_dict(),
        "metrics": {"elapsed_sec": 4.2, "num_tool_calls": 1, "num_model_calls": 2, "num_steps": 4, "outcome": "completed"},
        "judgment": None,
    }
    p = tmp_path / "test_run.jsonl"
    p.write_text(json.dumps(record) + "\n", encoding="utf-8")
    return p


def test_load_run_yields_records(tmp_path):
    p = _make_run_file(tmp_path)
    records = list(load_run(p))
    assert len(records) == 1
    assert records[0]["question_id"] == "Q3"


def test_load_trajectories_reconstructs_objects(tmp_path):
    p = _make_run_file(tmp_path)
    pairs = load_trajectories(p)
    assert len(pairs) == 1
    rec, traj = pairs[0]
    assert isinstance(traj, Trajectory)
    assert len(traj.steps) == 4
    assert traj.outcome.value == "completed"


def test_pretty_print_renders_all_step_kinds(tmp_path):
    p = _make_run_file(tmp_path)
    _, traj = load_trajectories(p)[0]
    rendered = pretty_print(traj)
    assert "model_call" in rendered
    assert "tool_call" in rendered
    assert "tool_result" in rendered
    assert "GEPA stands for" in rendered
    assert "gepa.pdf" in rendered
