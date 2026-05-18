"""Trajectory replay (Flavor A: debugging / time-travel).

Loads JSONL run files produced by the eval harness and renders trajectories
human-readably. Counterfactual replay against alternate prompts (Flavor B)
is deferred to Ch 3+ when memory makes the cost savings matter.

SPEC section 2.2 makes the architectural claim that trajectories are
replayable; this module is the most basic form of that promise.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from helix.trajectory import StepKind, Trajectory


def load_run(jsonl_path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield one record per line from an eval harness JSONL run.

    Each record has keys: question_id, band, question, reference_answer,
    answer, trajectory, metrics, judgment.
    """
    path = Path(jsonl_path)
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_trajectories(jsonl_path: str | Path) -> list[tuple[dict[str, Any], Trajectory]]:
    """Load every record from a run file and reconstruct its Trajectory object.

    Returns a list of (record_dict, trajectory) pairs. The record_dict gives
    you the question and metrics; the Trajectory gives you the replayable
    step list.
    """
    out: list[tuple[dict[str, Any], Trajectory]] = []
    for rec in load_run(jsonl_path):
        traj_data = rec.get("trajectory", {})
        if traj_data:
            out.append((rec, Trajectory.from_dict(traj_data)))
    return out


def pretty_print(trajectory: Trajectory, indent: int = 2) -> str:
    """Render a Trajectory as a numbered, indented step log.

    Tool calls are collapsed to one line with name + truncated args. Model
    calls show the assistant content. Tool results show source + score for
    retrieval-shaped results, raw payload otherwise.
    """
    pad = " " * indent
    lines: list[str] = []
    lines.append(f"Trajectory {trajectory.id}")
    lines.append(f"  task: {trajectory.task[:140]}{'...' if len(trajectory.task) > 140 else ''}")
    lines.append(f"  outcome: {trajectory.outcome.value}  steps: {len(trajectory.steps)}")
    if trajectory.artifacts_used:
        lines.append(f"  artifacts_used: {trajectory.artifacts_used}")
    lines.append("")

    for s in trajectory.steps:
        prefix = f"{pad}[{s.index:>2}] {s.kind.value:<14}"
        if s.kind == StepKind.MODEL_CALL:
            content = s.payload.get("response", {}).get("content", "")
            tool_calls = s.payload.get("response", {}).get("tool_calls", [])
            if tool_calls:
                tc_names = [tc.get("function", {}).get("name", "?") for tc in tool_calls]
                lines.append(f"{prefix} -> calls: {tc_names}")
                if content:
                    lines.append(f"{pad}     thought: {_truncate(content, 200)}")
            else:
                lines.append(f"{prefix} -> {_truncate(content, 200)}")
        elif s.kind == StepKind.TOOL_CALL:
            name = s.payload.get("name", "?")
            args = s.payload.get("arguments", {})
            lines.append(f"{prefix} {name}({_truncate(str(args), 150)})")
        elif s.kind == StepKind.TOOL_RESULT:
            result = s.payload.get("result", {})
            if isinstance(result, list) and result and isinstance(result[0], dict) and "source" in result[0]:
                # retrieval-shaped result
                summary = ", ".join(
                    f"{h.get('source', '?')}:p{h.get('page', '?')}@{h.get('score', 0):.2f}"
                    for h in result[:5]
                )
                lines.append(f"{prefix} {len(result)} hits — {summary}")
            else:
                lines.append(f"{prefix} {_truncate(str(result), 200)}")
        else:
            lines.append(f"{prefix} {_truncate(str(s.payload), 200)}")

    if trajectory.final_output:
        lines.append("")
        lines.append(f"  final_output: {_truncate(str(trajectory.final_output), 400)}")
    return "\n".join(lines)


def _truncate(s: str, n: int) -> str:
    if len(s) <= n:
        return s
    return s[: n - 3] + "..."


def print_run_summary(jsonl_path: str | Path) -> None:
    """Print a one-line-per-question summary of a run file."""
    print(f"\nRun: {jsonl_path}\n")
    for rec, traj in load_trajectories(jsonl_path):
        elapsed = rec.get("metrics", {}).get("elapsed_sec", 0)
        tools = rec.get("metrics", {}).get("num_tool_calls", 0)
        verdict = ""
        judgment = rec.get("judgment")
        if judgment and "preference" in judgment:
            verdict = f" judged: {judgment['preference']}"
        print(
            f"  [{rec['band']}] {rec['question_id']}: "
            f"{elapsed:>5.1f}s, {tools} tool calls, {len(traj.steps)} steps{verdict}"
        )
