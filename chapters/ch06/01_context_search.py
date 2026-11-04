"""Search the procedural-memory artifact the way chapters 2 to 4 searched prompts.

Chapter 4's GEPA searches the function the assistant recalls from
procedural memory. It starts from the function a model writes from the
contract alone, which the user accepts on none of the development
requests, and measures every candidate by the share of development
requests the user accepts when the assistant books its earliest start.
Every candidate is a version in the archive with its parent and score.

The best candidate is then checked on the validation and test requests.
When it is accepted on all of them, a person promotes it with --promote,
and it becomes the bootstrapped memory: the searched context that the
learning search must reproduce.

Replays the recording by default; --record sends missing calls to
gpt-5.4-mini and keeps them.
"""

from __future__ import annotations

import argparse
import asyncio
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chapters.ch06.assistant import context_search as cs
from chapters.ch06.assistant.harness import MEMORIES, PRICES
from chapters.ch06.assistant.recording import Recording
from helix.archive import SQLiteArchive
from helix.artifact import ArtifactKind, genesis
from helix.search.base import SearchBudget
from helix.search.gepa import GEPA

HERE = Path(__file__).parent
RECORDED = HERE / "assistant" / "recorded" / "context_search.jsonl"
SEED = MEMORIES / "find_shared_starts.contract_only.py"
SEARCHED = MEMORIES / "find_shared_starts.searched.py"
PROPOSER = "gpt-5.4-mini"
# The seed whose draws reproduce the recorded search's parent and operator
# choices; any seed works for a new recording.
RNG_SEED = 1165002


async def main(args: argparse.Namespace) -> None:
    recording = Recording(
        args.recording,
        mode="record" if args.record else "replay",
        distinct_repeats=True,
    )
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    for leftover in out.glob("context_search.sqlite*"):
        leftover.unlink()  # each run is its own search, not a continuation
    archive = SQLiteArchive(out / "context_search.sqlite")
    seed = genesis(
        "memory.find_shared_starts",
        ArtifactKind.CODE,
        SEED.read_text(encoding="utf-8"),
        created_by="no_experience",
    )
    search = GEPA(
        proposer_model=PROPOSER,
        eval_source=cs.NoQuestions(),
        population_size=args.population,
        generations=args.generations,
        mutation_prompt=cs.mutation_prompt(),
        crossover_prompt=cs.crossover_prompt(),
        evaluator=cs.evaluator,
        validate=cs.validate,
        proposer_params={"reasoning_effort": "medium"},
        # GEPA picks parents and operators at random; its own seeded
        # generator lets a replay choose as the recorded run did.
        rng=random.Random(RNG_SEED),
    )
    budget = SearchBudget(
        max_candidates=args.population * args.generations
    )
    with recording.active():
        seed_score = await cs.evaluator(seed)
        print(
            "seed v1 (contract only):"
            f" {seed_score.metadata['accepted']}/16 development"
            " requests accepted"
        )
        winners = [
            v async for v in search.propose(seed, None, archive, budget)
        ]
    lineage = await archive.all_artifacts_with_measurements()
    for variant in sorted(lineage, key=lambda v: v.artifact.version):
        if variant.measurement is None:
            continue
        art, score = variant.artifact, variant.measurement.score or 0.0
        parent = art.parent_id[1] if art.parent_id else "-"
        accepted = variant.measurement.metadata.get("accepted", "?")
        print(
            f"  v{art.version:<3} parent v{parent:<3}"
            f" {art.created_by:<16} {accepted:>2}/16 accepted,"
            f" score {score:.3f}"
        )
    archive.close()
    if not winners:
        print("No candidate was produced.")
        return
    winner = winners[0].artifact
    source = winner.content
    print(f"\nKept: v{winner.version}")
    for split in ("validation", "test"):
        accepted, total, _ = await cs.acceptance(source, split)
        print(f"  {split}: {accepted}/{total} accepted")
    (out / "searched_find_shared_starts.py").write_text(
        source, encoding="utf-8"
    )
    if args.record:
        print(
            f"\n{recording.live_calls} live proposer calls,"
            f" ${recording.spent(PRICES):.2f}"
        )
    if args.promote:
        SEARCHED.write_text(
            f"# Found by the context search (v{winner.version}),"
            " promoted by a person.\n" + source,
            encoding="utf-8",
        )
        print(f"Promoted: {SEARCHED.relative_to(REPO_ROOT)}")
    else:
        print(
            "Not promoted; a person promotes the searched memory with"
            " --promote."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--record", action="store_true", help="Call gpt-5.4-mini for calls the recording lacks.")
    parser.add_argument("--recording", type=Path, default=RECORDED)
    parser.add_argument("--population", type=int, default=4)
    parser.add_argument("--generations", type=int, default=4)
    parser.add_argument("--promote", action="store_true", help="Save the kept function as the searched memory.")
    parser.add_argument("--output", type=Path, default=HERE / "runs" / "context_search")
    args = parser.parse_args()
    if args.record:
        from helix.env import load_env

        load_env()
    try:
        asyncio.run(main(args))
    except KeyError as exc:
        sys.exit(f"{exc.args[0]}. The recording has no runs for this search yet; "
                 "record them with --record (paid calls to gpt-5.4-mini).")
