"""Shared plumbing for the chapter 5 scripts.

Every script builds the same helpdesk agent over chapter-local memory tiers
backed by one archive per script, replays the same traffic, and scores
answers with the same checker. What differs from script to script is who
updates memory and how: a direct value update, MemoryLearner, a group round, or the
online improver. Keeping the plumbing here keeps each script about its one
idea.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Awaitable, Callable

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from helix.agent import Agent
from helix.archive import SQLiteArchive
from helix.hooks import HookPoint
from helix.memory.base import MemoryContext
from helix.trajectory import Trajectory

from agents import helpdesk
from chapter_appendices.ch05_research_legacy.traffic import Asking, check, load_traffic

RUNS_DIR = Path(__file__).resolve().parent / "runs"
SMOKE = "--smoke" in sys.argv

# Chapter scripts run a few requests at once; the helpdesk is cheap per
# call but the retrieval index and the embedder are shared, so keep it low.
SEM = asyncio.Semaphore(3)


def setup_console() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def fresh_archive(stem: str) -> SQLiteArchive:
    """A new archive for this script's run; an older one is moved aside."""
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    path = RUNS_DIR / f"{stem}.sqlite"
    if path.exists():
        backup = RUNS_DIR / f"{stem}.prev.sqlite"
        if backup.exists():
            backup.unlink()
        path.rename(backup)
    for suffix in ("-wal", "-shm"):
        side = RUNS_DIR / f"{stem}.sqlite{suffix}"
        if side.exists():
            side.unlink()
    return SQLiteArchive(path, check_same_thread=False)


def fresh_tiers(stem: str, archive: SQLiteArchive, tiers: tuple[str, ...] = ("episodic",), utility_weight: float = 0.0) -> dict:
    """Chapter-local tiers under runs/<stem>_memory/, wiped each run."""
    d = RUNS_DIR / f"{stem}_memory"
    if d.exists():
        for f in d.glob("*.sqlite*"):
            f.unlink()
    return helpdesk.build_memory_tiers(d, archive=archive, tiers=tiers, utility_weight=utility_weight)


def build_agent(memory_tiers: dict | None, *, question_id_slot: dict | None = None, read_only: bool = False, **kw) -> Agent:
    """Install chapter memory tiers in the helpdesk's fixed harness.

    The checker needs to know which question a trajectory answered, and the
    loop only ever sees the trajectory, so a SESSION_START hook stamps the
    id the caller placed in `question_id_slot` before the run. Agent constructs
    MemoryHooks from these tiers; learning uses this same memory application path.
    """
    agent = helpdesk.build(memory_tiers=memory_tiers if memory_tiers is not None else {}, skip_memory=memory_tiers is None, **kw)
    if read_only and agent.memory_hooks is not None:
        agent.memory_hooks.read_only = True
    if question_id_slot is not None:

        async def tag(trajectory: Trajectory, **_: Any) -> None:
            trajectory.metadata["question_id"] = question_id_slot.get("question_id")
            trajectory.metadata["band"] = question_id_slot.get("band")

        agent.hooks.register(HookPoint.SESSION_START, tag)
    return agent


def reward_for(trajectory: Trajectory) -> float | None:
    """The checker as a reward function over a finished trajectory."""
    qid = trajectory.metadata.get("question_id")
    if not qid:
        return None
    return check(qid, trajectory.final_output if isinstance(trajectory.final_output, str) else None)


async def run_asking(
    agent: Agent,
    asking: Asking,
    slot: dict,
    *,
    session_suffix: str = "",
) -> tuple[str | None, Trajectory, float]:
    """Run one asking through the agent under its user and session scope."""
    async with SEM:
        slot["question_id"] = asking.question_id
        slot["band"] = asking.band
        answer, trajectory = await agent.run(
            asking.text,
            context=MemoryContext(session_id=asking.session_id + session_suffix, user_id=asking.user_id),
        )
    return answer, trajectory, check(asking.question_id, answer)


def session(askings: list[Asking], suffix: str) -> list[Asking]:
    return [a for a in askings if a.session_id.endswith(suffix)]


def by_band(rows: list[tuple[int, float]]) -> dict[int, float]:
    acc: dict[int, list[float]] = {}
    for band, score in rows:
        acc.setdefault(band, []).append(score)
    return {b: sum(v) / len(v) for b, v in sorted(acc.items())}


def print_band_table(title: str, columns: dict[str, list[tuple[int, float]]]) -> None:
    """Per-band means side by side, one column per condition."""
    names = list(columns)
    print(f"\n{title}")
    print("  band  " + "".join(f"{n:>22}" for n in names))
    bands = sorted({b for rows in columns.values() for b, _ in rows})
    for b in bands:
        cells = []
        for n in names:
            means = by_band(columns[n])
            cells.append(f"{means.get(b, float('nan')):>22.2f}")
        print(f"  {b:<6}" + "".join(cells))
    overall = []
    for n in names:
        rows = columns[n]
        overall.append(f"{(sum(s for _, s in rows) / len(rows)) if rows else float('nan'):>22.2f}")
    print("  all   " + "".join(overall))


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")


# ---------------------------------------------------------------------------
# Smoke mode: a scripted model so the loop mechanics can be exercised with
# no API. The answers are canned, so nothing printed under --smoke is a
# result; it only proves the scripts run end to end.
# ---------------------------------------------------------------------------

@contextmanager
def maybe_smoke():
    if not SMOKE:
        yield
        return
    from types import SimpleNamespace
    from unittest.mock import patch

    from chapter_appendices.ch05_research_legacy.traffic import load_questions

    refs = {q["id"]: q for q in load_questions()}
    texts = {}
    for a in load_traffic():
        texts[a.text] = a.question_id
    fabrications = {
        "Q16": "HyperAgents reports 50.0% on SWE-Bench.",
        "Q17": "AlphaEvolve reports a 23% speedup on the matmul benchmark.",
        "Q18": "Packer et al. (MemGPT) introduces the Free Energy Principle.",
        "Q19": "AgentPRM warns that reward hacking appears in production when the PRM is over-optimized.",
        "Q20": "Schmidhuber 2003 recommends a priority ordering policy for HelixAgent hooks.",
    }

    async def scripted(*, model, messages, **kwargs):
        system = messages[0]["content"]
        system_text = system if isinstance(system, str) else system[0]["text"]
        user = next((m["content"] for m in messages if m.get("role") == "user"), "")
        qid = texts.get(user, "Q1")
        q = refs[qid]
        if q["band"] == 4:
            # A trap: the scripted agent abstains only when RECALLED memory
            # carries an abstention or a lesson about the corpus. Only the
            # recalled section counts; the base prompt always mentions the
            # corpus and must not trigger the cue.
            marker = "## Relevant past interactions"
            recalled_text = system_text.split(marker, 1)[1].lower() if marker in system_text else ""
            cues = ("not in the corpus", "not reported", "does not contain", "no paper in the corpus", "check the corpus", "lacks the paper", "say when it lacks")
            answer = q["reference_answer"] if any(c in recalled_text for c in cues) else fabrications[qid]
        else:
            answer = q["reference_answer"]
        msg = SimpleNamespace(content=answer, tool_calls=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)], usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2))

    print("[smoke] scripted model in place; nothing below is a result\n")
    with patch("helix.agent.helix_acompletion", side_effect=scripted):
        yield
