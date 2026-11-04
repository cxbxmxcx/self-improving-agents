"""Chapter 5, section 5.4: GRPO without the gradient.

One failed run is worth several lessons, and only some of them are worth
keeping. The sampler reads carol's trap-question trajectory and proposes N
candidate insights; each candidate is scored on its own by replaying the
trap questions with only that insight in memory; the checker's mean is the
candidate's score; the advantage is the score minus the group mean; the
positive-advantage insights become memory entries and the rest are culled.
That is the selection mechanic of Group Relative Policy Optimization with
the weight update removed: the reinforcement is an artifact that persists.

A second round seeds the sampler from the best survivor's replay, so the
group mean has a chance to rise.

Run:

    python chapter_appendices/ch05_research_legacy/05_group_round.py
    python chapter_appendices/ch05_research_legacy/05_group_round.py --smoke
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from helix.artifact import Artifact, ArtifactKind
from helix.env import load_env
from helix.eval.dataset import EvalQuestion
from helix.improvement.group_round import run_group_round
from helix.llm_call import acompletion
from helix.memory.archive_bridge import entry_payload
from helix.memory.base import MemoryEntry, Scope, ScopeKey
from helix.memory.episodic import EpisodicMemory
from helix.memory import operators as ops
from helix.search.group_relative import GroupRelativeSearch, InsightSampler, SplitSampler
from helix.signal import Cost, GapMeasurement, SignalKind

from chapter_appendices.ch05_research_legacy.common import (
    RUNS_DIR,
    SMOKE,
    build_agent,
    fresh_archive,
    fresh_tiers,
    maybe_smoke,
    run_asking,
    session,
    setup_console,
    write_jsonl,
)
from chapter_appendices.ch05_research_legacy.traffic import check, load_traffic

load_env()
setup_console()

STEM = "group_round"
GROUP_SIZE = 4
ROUNDS = 2
SAMPLER_MODEL = "claude-sonnet-4-5"


class CheckerSignal:
    """The chapter's checker as an absolute Signal over a trajectory.

    The group round runs candidates without a memory context, so the
    question is identified by its text rather than a tag on the trajectory.
    """

    kind = SignalKind.GROUND_TRUTH
    signal_id = "helpdesk-checker"
    signal_version = 1
    cost_estimate = Cost()

    def __init__(self) -> None:
        self._by_text = {a.text: a.question_id for a in load_traffic()}

    async def measure(self, candidate=None, trajectory=None, reference=None, ground_truth=None) -> GapMeasurement:
        qid = self._by_text.get(trajectory.task if trajectory else "", "")
        answer = trajectory.final_output if trajectory and isinstance(trajectory.final_output, str) else None
        return GapMeasurement(score=check(qid, answer), signal_id=self.signal_id, signal_version=1, metadata={"question_id": qid})


async def sonnet(prompt: str) -> str:
    resp = await acompletion(model=SAMPLER_MODEL, messages=[{"role": "user", "content": prompt}], temperature=0.8)
    return resp.choices[0].message.content or ""


def make_factory():
    """An agent whose only memory is the one candidate insight, at global scope.

    Global scope because the round runs questions without a user; read-only
    because an evaluation must not grow the memory it is measuring. The
    factory is async, which the round awaits.
    """

    async def factory(candidate: Artifact):
        tier = EpisodicMemory(":memory:", archive=None)
        insight = entry_payload(candidate)
        entry = MemoryEntry(
            id=f"cand-{candidate.version}",
            scope_key=ScopeKey(Scope.GLOBAL, "*"),
            content={"user_message": "a lesson from a past run", "final_output": str(insight.get("insight", ""))},
            score=0.5,
        )
        await tier.write(entry)
        return build_agent({"episodic": tier}, read_only=True)

    return factory


async def main() -> None:
    archive = fresh_archive(STEM)
    tiers = fresh_tiers(STEM, archive)
    epi = tiers["episodic"]
    traffic = load_traffic()
    traps_s1 = [a for a in session(traffic, "-s1") if a.user_id == "carol"]
    traps_s2 = [a for a in session(traffic, "-s2") if a.user_id == "carol"]
    questions = [EvalQuestion(id=a.question_id, band=a.band, question=a.text, reference_answer=a.reference_answer) for a in traps_s2]

    # The run the lessons come from: carol's first trap, no memory.
    slot: dict = {}
    bare = build_agent(None, question_id_slot=slot)
    seed_asking = traps_s1[0]
    answer, traj, score = await run_asking(bare, seed_asking, slot)
    print(f"Seed run [{seed_asking.question_id}] scored {score:.2f}: {' '.join((answer or '').split())[:140]}\n")

    sampler = SplitSampler(n=GROUP_SIZE) if SMOKE else InsightSampler(sonnet, n=GROUP_SIZE)
    signal = CheckerSignal()
    seed_artifact = Artifact(
        id="memory.episodic.seed-lessons",
        version=1,
        kind=ArtifactKind.MEMORY_ENTRY,
        content={"tier": "episodic", "scope": "global:*", "entry": {"insight": "Check the corpus before answering. Say when it lacks the paper. Never transpose a number from another paper. Name the paper the number came from.", "trajectory_id": traj.id}, "utility": 0.5, "revision": 0},
        created_by="human",
    )
    await archive.put_artifact(seed_artifact)

    log = []
    context = {"trajectory": traj}
    for round_index in range(ROUNDS):
        search = GroupRelativeSearch(sampler, group_size=GROUP_SIZE, context=context)
        result = await run_group_round(
            proposer=search,
            seed=seed_artifact,
            signal=signal,
            archive=archive,
            questions=questions,
            agent_factory=make_factory(),
            round_index=round_index,
            max_concurrent_questions=3,
        )
        print(f"Round {round_index}: group mean {result.group_mean:.2f}, std {result.group_std:.2f}, {len(result.selected)} of {result.n_candidates} kept\n")
        for variant, m in result.scores:
            adv = m.metadata.get("advantage", 0.0)
            mark = "+" if adv > 0 else " "
            text = str(entry_payload(variant.artifact).get("insight", ""))[:100]
            print(f"  {mark} score {m.score:.2f}  adv {adv:+.2f}  {text}")
            log.append({"round": round_index, "version": variant.artifact.version, "score": m.score, "advantage": adv, "insight": text})
        # The survivors become memory for carol: ExpeL's ADD, with the
        # group round's verdict as the entry's starting utility.
        for variant in result.selected:
            insight = str(entry_payload(variant.artifact).get("insight", ""))
            entry = MemoryEntry(
                id=f"insight-r{round_index}-v{variant.artifact.version}",
                scope_key=ScopeKey(Scope.USER, "carol"),
                content={"user_message": "lesson from a past trap question", "final_output": insight, "source_trajectory_id": traj.id},
                score=float(variant.measurement.score if variant.measurement else 0.5),
                metadata={"created_by": "group_round", "round": round_index},
            )
            await ops.add(epi, entry, created_by="group_round")
        print()
        # Next round: lessons from the best survivor's own replay.
        if result.best is not None and round_index + 1 < ROUNDS:
            best_agent = await make_factory()(result.best.artifact)
            _, best_traj, _ = await run_asking(best_agent, traps_s2[0], {}, session_suffix="-seed")
            context = {"trajectory": best_traj}

    print("Carol's memory after the rounds:")
    for e in sorted(await epi.list_all(), key=lambda e: e.score, reverse=True):
        print(f"  u={e.score:.2f}  {str(e.content.get('final_output', ''))[:110]}")
    write_jsonl(RUNS_DIR / f"{STEM}_log.jsonl", log)
    print(f"\nArchive: {archive.path}")


if __name__ == "__main__":
    with maybe_smoke():
        asyncio.run(main())
