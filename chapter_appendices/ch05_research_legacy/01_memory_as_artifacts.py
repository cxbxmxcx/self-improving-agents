"""Chapter 5, section 5.1: memory entries are artifacts, memory is a hook pair.

The helpdesk runs three of alice's questions in one session and one reworded
question in the next. Nothing in the agent loop reads or writes memory: a
PRE_MODEL hook recalls, a SESSION_END hook writes back, and every entry the
write-back produces is a MEMORY_ENTRY artifact in the run's archive with a
version and a parent pointer. The script ends by upvoting one entry so the
reader can see a memory mutation leave the same lineage a prompt mutation
leaves in chapter 4.

Run:

    python chapter_appendices/ch05_research_legacy/01_memory_as_artifacts.py
    python chapter_appendices/ch05_research_legacy/01_memory_as_artifacts.py --smoke   # scripted model, no API
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from helix.artifact import ArtifactKind
from helix.env import load_env
from helix.hooks import HookPoint
from helix.memory import operators as ops
from helix.memory.archive_bridge import entry_payload, entry_utility
from helix.memory.hooks import RECALLED_KEY, WRITTEN_KEY

from chapter_appendices.ch05_research_legacy.common import (
    build_agent,
    fresh_archive,
    fresh_tiers,
    maybe_smoke,
    run_asking,
    session,
    setup_console,
)
from chapter_appendices.ch05_research_legacy.traffic import load_traffic

load_env()
setup_console()

STEM = "helpdesk_memory"


def describe_hooks(agent) -> None:
    """Show the reader that memory arrived as two hooks, not two loop branches."""
    print("Registered hooks on the helpdesk agent:")
    for point in (HookPoint.SESSION_START, HookPoint.PRE_MODEL, HookPoint.SESSION_END):
        names = [getattr(fn, "__qualname__", repr(fn)) for fn in agent.hooks._hooks[point]]
        print(f"  {point.value:<14} {names}")
    print()


async def main() -> None:
    archive = fresh_archive(STEM)
    tiers = fresh_tiers(STEM, archive)
    slot: dict = {}
    agent = build_agent(tiers, question_id_slot=slot)
    describe_hooks(agent)

    traffic = load_traffic()
    alice_s1 = [a for a in session(traffic, "-s1") if a.user_id == "alice"][:3]
    alice_s2 = [a for a in session(traffic, "-s2") if a.user_id == "alice" and a.question_id == alice_s1[0].question_id]

    print("Session 1: three questions, nothing to recall yet.\n")
    for asking in alice_s1:
        answer, traj, score = await run_asking(agent, asking, slot)
        recalled = traj.metadata.get(RECALLED_KEY, [])
        written = traj.metadata.get(WRITTEN_KEY, {})
        print(f"[{asking.question_id}] {asking.text[:70]}")
        print(f"    recalled {len(recalled)} entr{'y' if len(recalled) == 1 else 'ies'}; "
              f"wrote {written.get('artifact_ref')}; checker {score:.2f}")

    print("\nSession 2: the same question in other words.\n")
    for asking in alice_s2:
        answer, traj, score = await run_asking(agent, asking, slot)
        recalled = traj.metadata.get(RECALLED_KEY, [])
        print(f"[{asking.question_id}] {asking.text[:70]}")
        for r in recalled:
            print(f"    recalled {r['artifact_ref']} from {r['scope']}")
        print(f"    artifacts_used at step 0: {traj.artifacts_used.get(0)}")
        print(f"    checker {score:.2f}")

    print("\nThe archive after four runs (memory entries only):\n")
    rows = await archive.best(k=50, kind=ArtifactKind.MEMORY_ENTRY)
    everything = [v for v in await archive.all_artifacts_with_measurements() if v.artifact.kind is ArtifactKind.MEMORY_ENTRY]
    for v in everything:
        a = v.artifact
        payload = entry_payload(a)
        print(f"  {a.id} v{a.version}  parent={a.parent_id}  created_by={a.created_by}  "
              f"utility={entry_utility(a):.2f}  op={a.metadata.get('operator')}")
        print(f"      asked: {str(payload.get('user_message', ''))[:70]}")
    print(f"\n  {len(everything)} memory artifacts, {len(rows)} with measurements (none yet: nothing has scored them).")

    print("\nOne UPVOTE, and the entry has a lineage:\n")
    first = everything[0].artifact
    entry_id = first.metadata["entry_id"]
    updated = await ops.upvote(tiers["episodic"], entry_id, delta=0.2, reason="demo", created_by="human")
    chain = await archive.lineage((await archive.by_id(updated.artifact_ref[0])).artifact)
    for a in chain:
        print(f"  v{a.version}  parent={a.parent_id}  op={a.metadata.get('operator')}  utility={entry_utility(a):.2f}  by {a.created_by}")
    print(f"\nArchive: {archive.path}")
    print("Open the Observatory's Memory panel on this archive to see the same rows.")


if __name__ == "__main__":
    with maybe_smoke():
        asyncio.run(main())
