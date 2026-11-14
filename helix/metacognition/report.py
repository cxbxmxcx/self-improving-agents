"""What a deployment's metacognition has done and what it has cost, read from its store.

The economics follow the study's E5 analysis: runs with a trusted procedure
offered against runs without, the learning spend, and how many requests at
the procedure's cost pay that spend back.
"""

from __future__ import annotations

from typing import Any

from helix.metacognition.store import MetaStore


def _cell() -> dict[str, float]:
    return {"runs": 0, "accepted": 0, "judged": 0, "reasoning_tokens": 0, "dollars": 0.0}


def _add(cell: dict[str, float], run: dict[str, Any]) -> None:
    cell["runs"] += 1
    cell["judged"] += run["verdict"] is not None
    cell["accepted"] += run["verdict"] == "accepted"
    cell["reasoning_tokens"] += run["reasoning_tokens"] or 0
    cell["dollars"] += run["dollars"] or 0.0


def _means(cell: dict[str, float]) -> dict[str, float]:
    runs = cell["runs"] or 1
    return {**cell, "mean_reasoning_tokens": cell["reasoning_tokens"] / runs, "mean_dollars": cell["dollars"] / runs}


def summarize(store: MetaStore) -> dict[str, Any]:
    by_effort: dict[str, dict[str, float]] = {}
    offered, without = _cell(), _cell()
    for run in store.runs():
        _add(by_effort.setdefault(run["effort"] or "default", _cell()), run)
        _add(offered if run["offered"] else without, run)
    logged = store.learning()
    attempts = [a for a in logged if not a["outcome"].startswith("skipped")]  # the budget's refusals aren't
    learning = {"attempts": len(attempts), "writer_calls": sum(a["writer_calls"] for a in attempts),
                "dollars": sum(a["dollars"] for a in attempts), "skipped": len(logged) - len(attempts)}
    with_memory, no_memory = _means(offered), _means(without)
    saving = no_memory["mean_dollars"] - with_memory["mean_dollars"]
    break_even = learning["dollars"] / saving if offered["runs"] and without["runs"] and saving > 0 else None
    names = sorted({v["name"] for v in store.versions()})
    return {
        "runs": store.count(),
        "by_effort": {effort: _means(cell) for effort, cell in by_effort.items()},
        "with_procedure": with_memory, "without_procedure": no_memory,
        "learning": learning, "break_even_requests": break_even,
        "procedures": {name: [(v["version"], v["state"], v["created_by"]) for v in store.versions(name)]
                       for name in names},
        # learners a tournament chose (A5, L2): procedure -> mechanism
        "learners": {name: chosen["mechanism"] for name, chosen in (store.setting("learners") or {}).items()},
    }


def render(summary: dict[str, Any]) -> str:
    lines = [f"{summary['runs']} runs"]
    for effort, cell in summary["by_effort"].items():
        lines.append(f"  {effort:<8} {cell['accepted']}/{cell['judged']} accepted of {cell['runs']},"
                     f" {cell['mean_reasoning_tokens']:.0f} reasoning tokens and ${cell['mean_dollars']:.4f} a run")
    learning = summary["learning"]
    skipped = f", {learning['skipped']} skipped by the budget" if learning.get("skipped") else ""
    lines.append(f"  learning: {learning['attempts']} attempts, {learning['writer_calls']} writer calls,"
                 f" ${learning['dollars']:.4f}{skipped}")
    for name, mechanism in summary.get("learners", {}).items():
        lines.append(f"  {name} learns by {mechanism}, chosen by a learner tournament")
    if summary["break_even_requests"] is not None:
        lines.append(f"  break-even: about {summary['break_even_requests']:.1f} requests with a procedure")
    for name, versions in summary["procedures"].items():
        lines.append(f"  {name}: " + ", ".join(f"v{v} {state} ({by})" for v, state, by in versions))
    return "\n".join(lines)
