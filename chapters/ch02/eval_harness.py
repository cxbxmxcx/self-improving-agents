"""Eval harness for HelixAgent versions.

Loads chapters/ch02/eval_questions.json, runs an Agent factory against every
question, and writes one JSONL line per question to a timestamped file under
chapters/ch02/runs/. v0 records answers and metrics with no judging. v1 plugs
in the pairwise judge by passing a non-None judge function to run_eval().

Each line of the JSONL file is one EvalRecord:
  - question id, band, the question text, the reference answer
  - the agent's answer (final_output)
  - the trajectory (full step list, replayable)
  - metrics: latency, num_tool_calls, num_model_calls
  - judgment: set by the optional judge callback (None for v0)

Designed so v0 vs v1 comparison later is just two JSONL files diffed.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from helix.agent import Agent
from helix.artifact import Artifact
from helix.env import load_env
from helix.trajectory import StepKind, Trajectory

load_env()

QUESTIONS_PATH = Path(__file__).parent / "eval_questions.json"
RUNS_DIR = Path(__file__).parent / "runs"

JudgeFn = Callable[["EvalRecord"], Awaitable[dict[str, Any]]]
AgentFactory = Callable[[], Awaitable[Agent] | Agent]


@dataclass
class EvalRecord:
    question_id: str
    band: int
    question: str
    reference_answer: str
    expected_failure_mode_v0: str
    tags: list[str]
    agent_label: str
    answer: str | None
    trajectory: dict[str, Any]
    metrics: dict[str, Any]
    judgment: dict[str, Any] | None = field(default=None)

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__


def load_questions(path: Path = QUESTIONS_PATH) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    return data["questions"]


def trajectory_metrics(trajectory: Trajectory, elapsed_sec: float) -> dict[str, Any]:
    num_model = sum(1 for s in trajectory.steps if s.kind == StepKind.MODEL_CALL)
    num_tool = sum(1 for s in trajectory.steps if s.kind == StepKind.TOOL_CALL)
    return {
        "elapsed_sec": round(elapsed_sec, 2),
        "num_model_calls": num_model,
        "num_tool_calls": num_tool,
        "num_steps": len(trajectory.steps),
        "outcome": trajectory.outcome.value,
    }


async def run_eval(
    agent_factory: AgentFactory,
    agent_label: str,
    questions: list[dict[str, Any]] | None = None,
    judge: JudgeFn | None = None,
    out_path: Path | None = None,
) -> Path:
    """Run an agent against the eval set; write one JSONL line per question.

    agent_factory: zero-arg callable that returns a fresh Agent. A new Agent is
        built per question so trajectories and working memory don't leak.
    agent_label: short tag for the run file ("v0", "v1_spo_round3", etc).
    questions: optional override; defaults to the full set from
        eval_questions.json.
    judge: optional async callback that scores a completed EvalRecord. v0
        passes None. v1 passes a pairwise judge.
    out_path: optional override for the JSONL output path.

    Returns the path of the JSONL file written.
    """
    questions = questions if questions is not None else load_questions()

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    if out_path is None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_path = RUNS_DIR / f"{agent_label}_{ts}.jsonl"

    print(f"Running {len(questions)} questions against agent '{agent_label}'")
    print(f"Writing results to: {out_path}\n")

    band_counts: dict[int, int] = {}

    with out_path.open("w", encoding="utf-8") as f:
        for i, q in enumerate(questions, 1):
            print(f"[{i:>2}/{len(questions)}] {q['id']} (band {q['band']}): {q['question'][:80]}...")

            agent_or_coro = agent_factory()
            agent = await agent_or_coro if asyncio.iscoroutine(agent_or_coro) else agent_or_coro
            t0 = time.perf_counter()
            try:
                answer, trajectory = await agent.run(q["question"])
            except Exception as exc:
                elapsed = time.perf_counter() - t0
                record = EvalRecord(
                    question_id=q["id"],
                    band=q["band"],
                    question=q["question"],
                    reference_answer=q["reference_answer"],
                    expected_failure_mode_v0=q.get("expected_failure_mode_v0", ""),
                    tags=q.get("tags", []),
                    agent_label=agent_label,
                    answer=None,
                    trajectory={},
                    metrics={
                        "elapsed_sec": round(elapsed, 2),
                        "error": f"{type(exc).__name__}: {exc}",
                    },
                )
                f.write(json.dumps(record.to_dict()) + "\n")
                print(f"    ERROR: {exc}")
                continue

            elapsed = time.perf_counter() - t0
            record = EvalRecord(
                question_id=q["id"],
                band=q["band"],
                question=q["question"],
                reference_answer=q["reference_answer"],
                expected_failure_mode_v0=q.get("expected_failure_mode_v0", ""),
                tags=q.get("tags", []),
                agent_label=agent_label,
                answer=answer,
                trajectory=trajectory.to_dict(),
                metrics=trajectory_metrics(trajectory, elapsed),
            )

            if judge is not None:
                try:
                    record.judgment = await judge(record)
                except Exception as exc:
                    record.judgment = {"error": f"{type(exc).__name__}: {exc}"}

            f.write(json.dumps(record.to_dict()) + "\n")
            f.flush()
            band_counts[q["band"]] = band_counts.get(q["band"], 0) + 1
            print(f"    answered in {elapsed:.1f}s, {record.metrics.get('num_tool_calls', 0)} tool calls")

    print(f"\nDone. {sum(band_counts.values())} questions completed across {len(band_counts)} bands.")
    print(f"Output: {out_path}")
    summarize(out_path)
    return out_path


def summarize(jsonl_path: Path) -> None:
    """Print per-band metric averages from a completed run file."""
    by_band: dict[int, list[dict[str, Any]]] = {}
    with jsonl_path.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            by_band.setdefault(rec["band"], []).append(rec)

    print("\nPer-band summary:")
    print(f"  {'band':<6}{'n':<5}{'mean_sec':<12}{'mean_tools':<14}{'errors':<8}")
    for band in sorted(by_band):
        rows = by_band[band]
        errors = sum(1 for r in rows if "error" in r["metrics"])
        ok = [r for r in rows if "error" not in r["metrics"]]
        mean_sec = sum(r["metrics"]["elapsed_sec"] for r in ok) / len(ok) if ok else 0
        mean_tools = sum(r["metrics"].get("num_tool_calls", 0) for r in ok) / len(ok) if ok else 0
        print(f"  {band:<6}{len(rows):<5}{mean_sec:<12.2f}{mean_tools:<14.2f}{errors:<8}")


def main() -> None:
    """Run the v0 baseline against all 20 eval questions, no judge."""
    from chapters.ch02.helixagent_v0 import build_agent

    asyncio.run(
        run_eval(
            agent_factory=lambda: build_agent(),
            agent_label="v0",
        )
    )


if __name__ == "__main__":
    main()
