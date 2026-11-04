"""Listing 6.4: search the decision configuration offline, and tune its thresholds online.

Offline, on the 40 training requests, three searches run in turn over the
paired runs: coordinate steps on the routing thresholds, GEPA on the routing
and tool questions with Pareto selection and safety as a filter, and a
composed search that binds the winner into a new metacognition composite.
The composite is L3, so a person promotes it, and the archive keeps every
candidate with its parent.

Online, the thresholds alone keep improving. An online improver tries one
nudge at a time on replayed traffic, shadow-tests it in a forked week, and
promotes it when it wins; the thresholds are L1, so no person is needed.

By default the assistant is scripted and the decisions and proposals are
replayed, so no key is needed. --record refreshes the recordings with live
Jev and a live proposer; --live also runs gpt-5.4-mini.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from helix.archive import SQLiteArchive
from helix.eval.dataset import EvalQuestion, EvalSet
from helix.eval.source import FixedEvalSet
from helix.improvement import ImproverPolicy, OnlineImprover
from helix.memory.base import MemoryContext
from helix.scaffold import DecisionConfig, check_config, load_thresholds
from helix.search.base import SearchBudget, SearchCostModel, SearchKind, Variant
from helix.search.composed import ComposedSearch
from helix.search.gepa import GEPA
from helix.search.threshold import ThresholdSearch
from helix.signals.metacognition import OBJECTIVES

from chapter_appendices.ch06_decision_scaffold_legacy.assistant.environment import AssistantEnvironment, load_requests, preload_memory
from chapter_appendices.ch06_decision_scaffold_legacy.assistant.evaluation import (
    WEIGHTS,
    build_paired,
    estimate,
    fresh_tier,
    listing,
    measure,
    objective,
    run_request,
    summarize,
)
from chapter_appendices.ch06_decision_scaffold_legacy.assistant.recording import RECORDED, decision_model, recorded_proposals
from chapter_appendices.ch06_decision_scaffold_legacy.assistant.scripted import model_mode

PROPOSER = "gpt-5.4-mini"
REVIEWER = "person:reviewer"
ROUTING_THRESHOLDS = ["ask", "deliberate"]

CONFIG_MUTATION = """You improve the decision configuration of a personal assistant's
metacognition. The configuration is JSON: "questions" that a fast decision model
answers about each request, and "rules" that turn its answers into actions. The
route questions decide whether the assistant acts fast, deliberates first, or asks
the user a clarifying question, and the tool question decides whether a call needs
the user's confirmation.

You will see the configuration and feedback on requests it handled badly, with the
path that would have been best. Rewrite the route and tool questions so the decision
model separates those requests better. Make each instruction a narrow, observable
property of the request or the call, not a vague judgment such as "needs care". You
may reword an instruction, change a choice option's description, or add or remove a
route question together with its rule.

Keep the output question and its rule exactly as they are. Rules may only name the
thresholds "ask" and "deliberate", and a confirm rule must not name a threshold.
Return only the JSON configuration."""

CONFIG_CROSSOVER = """You combine two decision configurations for a personal
assistant's metacognition into one JSON configuration. Keep the questions and rules
from each parent that separate requests well, and drop duplicates. Keep the output
question and its rule exactly as they are; rules may only name the thresholds "ask"
and "deliberate", and a confirm rule must not name a threshold. Return only the JSON
configuration."""


class Chosen:
    """A child search that hands the composed search one measured configuration."""

    def __init__(self, variant: Variant) -> None:
        self.variant = variant

    @property
    def kind(self) -> SearchKind:
        return SearchKind.GENETIC_PARETO

    @property
    def cost_model(self) -> SearchCostModel:
        return SearchCostModel()

    async def propose(self, seed, signal, archive, budget):
        yield self.variant

    async def select(self, candidates, signal, archive):
        return candidates[0].artifact


def score(measurement) -> float:
    return measurement.score if measurement is not None and measurement.score is not None else -1.0


async def tune_thresholds(paired, decider, config, thresholds, archive) -> tuple:
    """Coordinate steps on the routing thresholds, each scored on the paired runs."""
    evaluate = paired.evaluator(decider, config=config, weights=WEIGHTS)
    best, best_m = thresholds, await evaluate(thresholds)
    await archive.record(Variant(thresholds, thresholds, "human"), best_m)
    search = ThresholdSearch(step=0.1, min_step=0.025, names=ROUTING_THRESHOLDS)
    moves, tried = [], 0
    for _ in range(12):
        step = search.step
        variants = [v async for v in search.propose(best, None, archive, SearchBudget())]
        for variant in variants:
            variant.measurement = await evaluate(variant.artifact)
            await archive.record(variant, variant.measurement)
        tried += len(variants)
        top = max(variants, key=lambda v: score(v.measurement), default=None)
        if top is not None and score(top.measurement) > score(best_m) + 1e-9:
            moves.append(f"{top.metadata['name']} {top.metadata['from']:.3f} -> {top.metadata['to']:.3f}")
            best, best_m = top.artifact, top.measurement
        elif step <= search.min_step:
            break
    return best, best_m, {"tried": tried, "moves": moves}


async def search_questions(paired, decider, config, thresholds, archive, train) -> tuple:
    """GEPA on the route and tool questions, with the thresholds held fixed."""
    evaluate = paired.evaluator(decider, thresholds=thresholds, weights=WEIGHTS)
    seed_m = await evaluate(config)
    await archive.record(Variant(config, config, "human"), seed_m)
    values = load_thresholds(thresholds)

    def validate(text: str) -> str | None:
        reason = check_config(text, values)
        if reason:
            return reason
        changed = paired.changed_hooks(DecisionConfig.model_validate(json.loads(text)))
        return f"keep the {' and '.join(changed)} questions and rules exactly as they are" if changed else None

    questions = FixedEvalSet(EvalSet(questions=[
        EvalQuestion(id=r["id"], band=1, question=r["text"], reference_answer="", category=r["category"])
        for r in train]))
    random.seed(6)  # GEPA picks parents and operators at random
    gepa = GEPA(proposer_model=PROPOSER, eval_source=questions, evaluator=evaluate, objectives=OBJECTIVES,
                validate=validate, mutation_prompt=CONFIG_MUTATION, crossover_prompt=CONFIG_CROSSOVER,
                population_size=4, generations=2)
    winners = [v async for v in gepa.propose(config, None, archive, SearchBudget())]
    winner = winners[0] if winners else None
    kept = winner is not None and score(winner.measurement) > score(seed_m)
    return (winner if kept else None), seed_m, {"discarded": len(gepa.discarded),
                                                "winner": score(winner.measurement) if winner else None}


async def bind_composite(scaffold, chosen: Variant, seed_m, archive):
    """A new metacognition composite binding the chosen configuration, ready for review."""
    await archive.record(Variant(scaffold.composite, scaffold.composite, "human"), seed_m)
    search = ComposedSearch({"config": Chosen(chosen)}, scaffold.constituents())
    variant = [v async for v in search.propose(scaffold.composite, None, archive, SearchBudget())][0]
    # Only the configuration changed, so the bundle scores as its configuration did.
    variant.measurement = chosen.measurement
    await archive.record(variant, variant.measurement)
    return await search.select([variant], None, archive)


async def review_on_held_out(decider, live: bool, searched: dict) -> dict:
    """The reviewer's evidence: the seed and the searched configuration on the
    held-out requests, which no search has seen."""
    env = AssistantEnvironment()
    evidence = {}
    for name, artifacts in (("seed", {}), ("searched", searched)):
        rows = []
        for request in load_requests(split="test"):
            trajectory = await run_request(request, env=env, decider=decider, live=live, artifacts=artifacts)
            rows.append(await measure(env, request, trajectory))
        evidence[name] = summarize(rows)
    return evidence


async def tune_online(decider, live: bool, archive, config, traffic) -> dict:
    """The thresholds keep improving on replayed traffic, one shadow-tested nudge at a time."""
    env = AssistantEnvironment()
    target = "assistant.routing.thresholds"
    current: dict = {}

    def shadow(agent, overrides):
        # The shadow run starts from the week the live run started from.
        fork = env.fork()
        fork.restore(current["snapshot"])
        return listing.build_assistant(fork, current["tier"], decider, live=live,
                                       artifacts={"config": config, **overrides})

    improver = None
    for request in traffic:
        # Deployment reads the live champion, so a promotion reaches the next request.
        champion = await archive.live_champion(target)
        env.reset()
        env.begin(request)
        tier = fresh_tier()
        await preload_memory(tier, request)
        current.update(snapshot=env.snapshot(), tier=tier)
        agent = listing.build_assistant(env, tier, decider, live=live,
                                        artifacts={"config": config, "thresholds": champion})
        agent.memory_hooks.read_only = True
        if improver is None:
            improver = OnlineImprover(
                agent=agent, target_artifact_id=target, signal=objective(env), archive=archive,
                search=ThresholdSearch(step=0.1, min_step=0.025, names=ROUTING_THRESHOLDS),
                policy=ImproverPolicy(rolling_window=6, rolling_threshold=0.9, shadow_sample=4),
                shadow_factory=shadow, improver_id="thresholds-online")
            agent.attach_improver(improver)
            await improver.start()
        else:
            improver.follow(agent)
        await agent.run(request["text"], context=MemoryContext(user_id=request["user"], session_id=request["session"]))
    await improver.stop()
    status = improver.status
    promoted = [p for p in await archive.promotion_history(target) if p.approver.startswith("improver:")]
    return {"requests": len(traffic), "proposed": status.candidates_proposed, "promoted": len(promoted),
            "champion": load_thresholds(await archive.live_champion(target)), "error": status.last_error}


def deciders(mode: str, output: Path):
    return decision_model(mode, folder=output if mode == "live" else RECORDED)


async def search(mode: str, output: Path) -> dict:
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    live = mode == "live"
    decider = deciders(mode, output)
    archive = SQLiteArchive(output / "archive.sqlite")
    train = load_requests(split="train")
    seed = listing.seed_artifacts()
    scaffold = listing.build_assistant(AssistantEnvironment(), fresh_tier(), decider, live=live).scaffold
    proposals = (RECORDED if mode != "live" else output) / "proposals.jsonl"
    report: dict = {"mode": mode, "requests": len(train), "weights": vars(WEIGHTS)}
    with model_mode(live), recorded_proposals(proposals, live=mode != "replay"):
        paired = await build_paired(train, decider=decider, live=live, path=output / "paired_train.json")
        report["paired_runs"] = sum(len(e["branches"]) for e in paired.entries.values())

        thresholds, t_m, t_report = await tune_thresholds(paired, decider, seed["config"], seed["thresholds"], archive)
        print(f"Paired runs: {report['paired_runs']} on {len(train)} training requests")
        print(f"Thresholds: {', '.join(t_report['moves']) or 'no change'} "
              f"({t_report['tried']} candidates); score {t_m.score:.3f}")

        chosen, seed_m, q_report = await search_questions(paired, decider, seed["config"], thresholds, archive, train)
        config = chosen.artifact if chosen else seed["config"]
        final_m = chosen.measurement if chosen else seed_m
        print(f"Questions: seed {seed_m.score:.3f}; GEPA winner "
              f"{'-' if q_report['winner'] is None else format(q_report['winner'], '.3f')}; "
              f"{q_report['discarded']} proposal(s) discarded; kept "
              f"{'the winner, config v' + str(config.version) if chosen else 'the seed'}")

        # A person promotes an L3 composite only on evidence the search never saw.
        evidence = await review_on_held_out(decider, live, {"config": config, "thresholds": thresholds})
        before, after = evidence["seed"], evidence["searched"]
        approved = after["unsafe"] == 0 and after["score"] >= before["score"]
        verdict = (f"held-out score {before['score']:.3f} -> {after['score']:.3f}, accepted "
                   f"{before['accepted']} -> {after['accepted']} of {after['requests']}, {after['unsafe']} unsafe")
        if chosen is not None:
            composite = await bind_composite(scaffold, chosen, seed_m, archive)
            print(f"Composite: {composite.id} v{scaffold.composite.version} -> v{composite.version} recorded")
        if approved:
            if chosen is not None:
                await archive.promote(composite.id, composite.version, approver=REVIEWER, reason=verdict)
            await archive.promote(thresholds.id, thresholds.version, approver=REVIEWER, reason=verdict)
        else:
            thresholds = seed["thresholds"]  # the live champion stays the seed
            await archive.promote(thresholds.id, thresholds.version, approver=REVIEWER,
                                  reason=f"kept the seed: {verdict}")
        print(f"Review by {REVIEWER}: {'promoted' if approved else 'rejected'} ({verdict})")
        report["review"] = {"approved": approved, "seed": before, "searched": after}
        report["offline"] = {"thresholds": load_thresholds(thresholds), "threshold_search": t_report,
                             "questions": q_report, "seed_score": seed_m.score, "final_score": final_m.score,
                             "objectives": final_m.metadata.get("objectives")}
        searched = {"config": json.loads(config.content), "config_version": config.version,
                    "thresholds": load_thresholds(thresholds), "thresholds_version": thresholds.version,
                    "training_score": final_m.score}
        target = RECORDED / "searched.json" if mode == "record" else output / "searched.json"
        target.write_text(json.dumps(searched, indent=2), encoding="utf-8")

        traffic = list(train)
        random.Random(6).shuffle(traffic)
        deployed = config if approved else seed["config"]
        report["online"] = await tune_online(decider, live, archive, deployed, traffic)
        online = report["online"]
        print(f"Online: {online['requests']} replayed requests; {online['proposed']} nudge(s) shadow-tested, "
              f"{online['promoted']} promoted; thresholds now {online['champion']}")
    (output / "report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print("Archive and report: " + str(output))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--record", action="store_true", help="Refresh the recorded decisions and proposals "
                                                             "with live Jev and a live proposer.")
    group.add_argument("--live", action="store_true", help="Also run gpt-5.4-mini for the assistant.")
    parser.add_argument("--output", type=Path, help="Directory for this run's archive and report.")
    parser.add_argument("--yes", action="store_true", help="With --live, start without stopping at the cost estimate.")
    args = parser.parse_args()
    mode = "live" if args.live else "record" if args.record else "replay"
    if mode == "live" and not args.yes:
        print(estimate(140 + 40 + 40 + 16))
        return
    if mode != "replay":
        from helix.env import load_env
        load_env()
    tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6]
    asyncio.run(search(mode, args.output or Path(__file__).parent / "runs/search" / tag))


if __name__ == "__main__":
    main()
