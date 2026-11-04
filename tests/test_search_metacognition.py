"""Chapter 6's search pieces: the threshold search, Pareto selection, GEPA's
absolute and multi-objective options, composite expansion, and the paired-run
cache that scores routing variants without running the agent. SPEC §4, §18."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from helix.archive import SQLiteArchive
from helix.artifact import Subtype, genesis
from helix.eval.dataset import EvalQuestion, EvalSet
from helix.eval.paired import PairedRuns
from helix.eval.source import FixedEvalSet
from helix.improvement import ImproverPolicy, OfflineImprover
from helix.improvement.deploy import expand_composite
from helix.scaffold import check_config, config_artifact, thresholds_artifact
from helix.scaffold.common import pinned
from helix.search.base import SearchBudget, Variant
from helix.search.composed import ComposedSearch
from helix.search.gepa import GEPA
from helix.search.objectives import aggregate, pareto_front, pick
from helix.search.threshold import ThresholdSearch
from helix.signal import GapMeasurement
from helix.signals.cost import CostWeights
from helix.signals.metacognition import OBJECTIVES, MetacognitionObjective
from helix.signals.outcome import OutcomeSignal
from helix.signals.routing import RoutingSignal
from helix.signals.safety import SafetySignal, tool_calls
from tests.test_scaffold import CALM, CONFIG, THRESHOLDS, FakeDecider, _assistant, _reply


def _measured(outcome, dollars, seconds, safe=True, score=None) -> GapMeasurement:
    return GapMeasurement(score=score if score is not None else outcome,
                          metadata={"objectives": {"outcome": outcome, "dollars": dollars,
                                                   "seconds": seconds, "safe": safe}})


# ---------------------------------------------------------------- threshold search


async def test_threshold_search_proposes_clamped_neighbors_without_an_llm():
    archive = SQLiteArchive(":memory:")
    seed = thresholds_artifact("meta.thresholds", {"ask": 0.5, "deliberate": 0.95, "revise": 0.0})
    search = ThresholdSearch(step=0.1)
    with patch("helix.llm_call.acompletion", AsyncMock(side_effect=AssertionError("no LLM"))):
        variants = [v async for v in search.propose(seed, signal=None, archive=archive, budget=SearchBudget())]
    moves = [(v.metadata["name"], v.metadata["to"]) for v in variants]
    # revise cannot go below 0, so that move is skipped rather than repeated.
    assert moves == [("ask", 0.6), ("ask", 0.4), ("deliberate", 1.0), ("deliberate", 0.85), ("revise", 0.1)]
    assert [v.artifact.version for v in variants] == [2, 3, 4, 5, 6]
    assert all(v.artifact.parent_id == seed.ref for v in variants)
    assert json.loads(variants[3].artifact.content) == {"ask": 0.5, "deliberate": 0.85, "revise": 0.0}


async def test_threshold_search_takes_one_step_per_online_trigger_and_halves_after_a_full_cycle():
    archive = SQLiteArchive(":memory:")
    seed = thresholds_artifact("meta.thresholds", {"ask": 0.5})
    search = ThresholdSearch(step=0.2, min_step=0.05)

    async def first(seed_art):
        async for variant in search.propose(seed_art, signal=None, archive=archive, budget=SearchBudget()):
            return variant

    up, down = await first(seed), await first(seed)
    assert (up.metadata["to"], down.metadata["to"]) == (0.7, 0.3)
    # A full cycle of two moves passed without a promotion, so the next
    # proposal uses half the step.
    assert search.step == 0.2
    assert (await first(seed)).metadata["to"] == 0.6 and search.step == 0.1
    # A new seed means the last step was promoted; the misses start over.
    promoted = (await first(up.artifact))
    assert promoted.metadata["step"] == 0.1 and search.step == 0.1


async def test_threshold_search_selects_the_best_measured_neighbor():
    seed = thresholds_artifact("t", {"ask": 0.5})
    a, b = seed.mutate('{"ask": 0.6}', created_by="t"), seed.mutate('{"ask": 0.4}', created_by="t", version=3)
    variants = [Variant(a, seed, "hill_climb", GapMeasurement(score=0.4)),
                Variant(b, seed, "hill_climb", GapMeasurement(score=0.7))]
    assert await ThresholdSearch().select(variants, signal=None, archive=None) is b


# ---------------------------------------------------------------- Pareto selection


def test_pareto_front_drops_unsafe_and_dominated_candidates():
    cheap = _measured(0.8, 0.002, 2.0)
    strong = _measured(1.0, 0.010, 6.0)
    dominated = _measured(0.8, 0.004, 3.0)
    unsafe = _measured(1.0, 0.001, 1.0, safe=False)
    front = pareto_front([cheap, strong, dominated, unsafe], OBJECTIVES)
    assert front == [cheap, strong]
    assert pick(front, OBJECTIVES) is strong
    assert pick(front, {"dollars": "minimize", "outcome": "maximize"}) is cheap


def test_aggregate_averages_objectives_and_requires_every_request_safe():
    rows = [("r1", _measured(1.0, 0.002, 2.0)), ("r2", _measured(0.0, 0.004, 4.0, safe=False, score=0.0))]
    result = aggregate(rows)
    assert result.score == 0.5 and result.triggered
    assert result.metadata["objectives"] == {"dollars": 0.003, "outcome": 0.5, "safe": False, "seconds": 3.0}


# ---------------------------------------------------------------- GEPA


def _questions(n=2) -> FixedEvalSet:
    return FixedEvalSet(EvalSet(questions=[EvalQuestion(id=f"q{i}", band=1, question=f"q{i}", reference_answer="")
                                           for i in range(n)]))


def _proposals(*contents):
    replies = [SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=c))], usage=None)
               for c in contents]
    return AsyncMock(side_effect=replies)


async def test_gepa_selects_the_safe_pareto_winner_with_an_evaluator_and_no_agent():
    seed = genesis("p", Subtype.PROMPT, "seed")
    scores = {"seed": _measured(0.5, 0.002, 2.0), "cheap": _measured(0.8, 0.002, 2.0), "strong": _measured(1.0, 0.010, 6.0),
              "reckless": _measured(1.0, 0.001, 1.0, safe=False)}

    async def evaluator(candidate, questions):
        return scores[candidate.content]

    gepa = GEPA(eval_source=_questions(), evaluator=evaluator, objectives=OBJECTIVES,
                population_size=3, generations=1)
    with patch("helix.search.gepa.helix_acompletion", _proposals("reckless", "cheap", "strong")):
        winners = [v async for v in gepa.propose(seed, signal=None, archive=SQLiteArchive(":memory:"),
                                                  budget=SearchBudget())]
    assert [w.artifact.content for w in winners] == ["strong"]


async def test_invalid_configuration_proposals_are_discarded():
    seed = config_artifact("meta.config", CONFIG)
    broken = json.loads(json.dumps(CONFIG))
    broken["rules"].append({"question": "nope", "op": ">=", "threshold": 0.5, "action": "ask"})
    better = json.loads(json.dumps(CONFIG))
    better["questions"]["ambiguous_reference"]["instructions"] = "Two contacts share the name in the request."
    fenced = "```json\n" + json.dumps(better) + "\n```"

    async def evaluator(candidate, questions):
        return GapMeasurement(score=0.9)

    gepa = GEPA(eval_source=_questions(), evaluator=evaluator, population_size=1, generations=1,
                validate=lambda text: check_config(text, THRESHOLDS))
    proposer = _proposals(json.dumps(broken), fenced)
    with patch("helix.search.gepa.helix_acompletion", proposer):
        winners = [v async for v in gepa.propose(seed, signal=None, archive=SQLiteArchive(":memory:"),
                                                  budget=SearchBudget())]
    assert [d["reason"] for d in gepa.discarded] == ["rule names an unknown question: 'nope'"]
    retry = proposer.await_args_list[1].kwargs["messages"][1]["content"]
    assert "rejected: rule names an unknown question" in retry
    assert json.loads(winners[0].artifact.content) == better
    gepa = GEPA(eval_source=_questions(), evaluator=evaluator, population_size=1, generations=1,
                validate=lambda text: check_config(text, THRESHOLDS), max_invalid=1)
    with patch("helix.search.gepa.helix_acompletion", _proposals("not json", "still not")):
        winners = [v async for v in gepa.propose(seed, signal=None, archive=SQLiteArchive(":memory:"),
                                                  budget=SearchBudget())]
    assert winners == [] and len(gepa.discarded) == 2


# ---------------------------------------------------------------- composites


class _BumpPlanner:
    kind = cost_model = None

    async def propose(self, seed, signal, archive, budget):
        yield Variant(seed.mutate("Check each limit across all events first.", created_by="bump"), seed, "bump")

    async def select(self, candidates, signal, archive):
        return candidates[0].artifact


async def test_composed_variants_record_and_expand_into_their_constituents():
    agent, _ = _assistant(FakeDecider())
    scaffold = agent.scaffold
    archive = SQLiteArchive(":memory:")
    search = ComposedSearch({"planner": _BumpPlanner()}, scaffold.constituents())
    variant = [v async for v in search.propose(scaffold.composite, None, archive, SearchBudget())][0]
    await archive.record(variant, GapMeasurement(score=0.9))  # metadata must be JSON
    assert variant.metadata["constituent_ref"] == ["meta.planner", 2]
    overrides = await expand_composite(archive, variant.artifact, scaffold.artifacts())
    assert overrides["meta.planner"].content == "Check each limit across all events first."
    clone = agent.with_artifacts(overrides)
    assert clone.scaffold.planner.version == 2 and clone.find_artifact("metacognition").version == 2
    with pytest.raises(LookupError, match="meta.planner:v2"):
        await expand_composite(SQLiteArchive(":memory:"), variant.artifact, scaffold.artifacts())


async def test_offline_improver_clones_run_a_composite_candidates_constituents():
    agent, _ = _assistant(FakeDecider())
    archive = SQLiteArchive(":memory:")
    search = ComposedSearch({"planner": _BumpPlanner()}, agent.scaffold.constituents())
    improver = OfflineImprover(agent=agent, target_artifact_id="metacognition", signal=object(), search=search,
                               archive=archive, eval_source=_questions(), policy=ImproverPolicy())
    variant = [v async for v in search.propose(agent.scaffold.composite, None, archive, SearchBudget())][0]
    clone = await improver._capped_build_agent(variant.artifact)
    assert clone.scaffold.planner.content == "Check each limit across all events first."


# ---------------------------------------------------------------- paired runs

REQUESTS = [
    {"id": "simple", "text": "Book writing time tomorrow morning."},
    {"id": "ambiguous", "text": "Move my meeting with Sam to Thursday."},
    {"id": "limits", "text": "Fit three meetings with Sam before Friday, two hours in all."},
    {"id": "send", "text": "Reply to Dana that Thursday works."},
]
TASK_ID = {r["text"]: r["id"] for r in REQUESTS}
PRICES = {"fast-model": (1.0, 5.0), "deliberate-model": (10.0, 50.0)}
WEIGHTS = CostWeights(dollars=5.0)


async def _scripted_model(*, model, messages, tools=None, **kwargs):
    """Fast or deliberate, the scripted worker does the same thing; what the
    path changes is whether it plans, asks, or acts, and what it costs."""
    system = messages[0]["content"]
    system = system[0]["text"] if isinstance(system, list) else system
    task = next(m["content"] for m in messages if m["role"] == "user")
    usage = SimpleNamespace(prompt_tokens=1000, completion_tokens=100)
    if "Ask before you act" in system:
        return _reply("Which one do you mean?", usage=usage)
    if messages[-1]["role"] == "tool" or not task.startswith("Reply to"):
        return _reply("Done.", usage=usage)
    return _reply(call="send_reply", arguments='{"to": "dana"}', usage=usage)


def _reward(trajectory) -> float:
    route = trajectory.metadata["metacognition"]["route"]
    request = TASK_ID[trajectory.task]
    if request == "simple":
        return 0.0 if route == "ask" else 1.0
    if request == "ambiguous":
        return 1.0 if route == "ask" else 0.0
    if request == "limits":
        return 1.0 if route == "deliberate" else 0.0
    calls = tool_calls(trajectory)
    return 1.0 if any(c["cancelled"] for c in calls) else (0.0 if route == "ask" else 1.0)


class _ByRequest(FakeDecider):
    """Answers that depend on which request the state is about."""

    def __init__(self, by_request: dict[str, dict]) -> None:
        super().__init__()
        self.by_request = by_request

    async def ask_with_usage(self, state, questions):
        text = state.get("request", "") if isinstance(state, dict) else str(state)
        saved = self.values
        self.values = {**CALM, **self.by_request.get(TASK_ID.get(text), {})}
        try:
            return await super().ask_with_usage(state, questions)
        finally:
            self.values = saved


OBJECTIVE = MetacognitionObjective(OutcomeSignal(_reward), SafetySignal({"send_reply"}), weights=WEIGHTS,
                                   prices=PRICES)
GOOD = {"ambiguous": {"ambiguous_reference": 0.8}, "limits": {"multi_event_limit": 0.9},
        "send": {"consequence": "sends"}}


async def _paired(tmp_path=None) -> tuple[PairedRuns, object]:
    agent, _ = _assistant(FakeDecider())
    paired = PairedRuns(tmp_path / "paired.json" if tmp_path else None)

    async def run(agent_, request):
        return await agent_.run(request["text"])

    with patch("helix.agent.helix_acompletion", _scripted_model):
        await paired.build(agent, REQUESTS, run, OBJECTIVE)
    return paired, agent


async def test_paired_runs_take_every_path_once_and_both_verdicts_on_a_checked_call():
    paired, _ = await _paired()
    assert sorted(paired.entries["simple"]["branches"]) == ["ask", "deliberate", "fast"]
    assert sorted(paired.entries["send"]["branches"]) == [
        "ask", "deliberate|allow", "deliberate|confirm", "fast|allow", "fast|confirm"]
    send = paired.entries["send"]["branches"]
    assert send["fast|allow"]["objectives"]["safe"] is False
    assert send["fast|confirm"]["objectives"] == {**send["fast|confirm"]["objectives"], "outcome": 1.0, "safe": True}
    assert paired.entries["ambiguous"]["route_state"] == {"request": "Move my meeting with Sam to Thursday."}


async def test_routing_only_variants_are_scored_from_the_paired_run_cache():
    paired, _ = await _paired()
    no_runs = AsyncMock(side_effect=AssertionError("the cache must not run the agent"))
    with patch("helix.agent.helix_acompletion", no_runs):
        good = await paired.evaluate(config_artifact("c", CONFIG), THRESHOLDS, _ByRequest(GOOD), weights=WEIGHTS)
        calm = await paired.evaluate(config_artifact("c", CONFIG), THRESHOLDS, FakeDecider(), weights=WEIGHTS)
    assert good.metadata["paths"] == {"simple": "fast", "ambiguous": "ask", "limits": "deliberate",
                                      "send": "fast|confirm"}
    assert good.metadata["objectives"]["safe"] is True and good.score > 0.9
    assert calm.metadata["paths"]["send"] == "fast|allow" and calm.triggered
    assert calm.score < good.score
    assert "deliberate would have scored" in calm.feedback


async def test_variants_that_change_decided_hooks_need_new_runs():
    paired, _ = await _paired()
    changed = json.loads(json.dumps(CONFIG))
    changed["questions"]["reply_complete"]["instructions"] = "The reply is friendly."
    with pytest.raises(ValueError, match="changes the output hooks"):
        await paired.evaluate(config_artifact("c", changed), THRESHOLDS, FakeDecider())


async def test_paired_evaluators_feed_threshold_and_configuration_searches(tmp_path):
    paired, _ = await _paired(tmp_path)
    reloaded = PairedRuns(tmp_path / "paired.json")
    assert reloaded.entries == json.loads(json.dumps(paired.entries))
    decider = _ByRequest({**GOOD, "limits": {"multi_event_limit": 0.4}})
    by_thresholds = reloaded.evaluator(decider, config=config_artifact("c", CONFIG), weights=WEIGHTS)
    strict, lenient = (thresholds_artifact("t", {**THRESHOLDS, "deliberate": v}) for v in (0.5, 0.3))
    assert (await by_thresholds(lenient)).score > (await by_thresholds(strict)).score
    with pytest.raises(ValueError, match="exactly one"):
        reloaded.evaluator(decider)


async def test_routing_signal_reads_regret_from_the_paired_runs():
    paired, agent = await _paired()
    routing = RoutingSignal(paired.lookup, weights=WEIGHTS)
    with patch("helix.agent.helix_acompletion", _scripted_model), pinned(route="fast", tool="confirm"):
        _, trajectory = await agent.run("Fit three meetings with Sam before Friday, two hours in all.")
    m = await routing.measure(agent.scaffold.config, trajectory=trajectory)
    assert (m.metadata["route"], m.metadata["best"], m.triggered) == ("fast", "deliberate", True)
    with pytest.raises(ValueError, match="cannot pin"):
        with pinned(route="sideways"):
            pass


async def test_with_an_evaluator_gepa_shows_the_first_generation_the_seeds_feedback():
    seed = genesis("p", Subtype.PROMPT, "seed")

    async def evaluator(candidate, questions):
        return GapMeasurement(score=0.4 if candidate.content == "seed" else 0.9,
                              feedback="'Fit three meetings' went fast; deliberate would have scored 0.85")

    proposer = _proposals("better")
    gepa = GEPA(eval_source=_questions(), evaluator=evaluator, population_size=1, generations=1)
    with patch("helix.search.gepa.helix_acompletion", proposer):
        [v async for v in gepa.propose(seed, signal=None, archive=SQLiteArchive(":memory:"), budget=SearchBudget())]
    assert "deliberate would have scored 0.85" in proposer.await_args.kwargs["messages"][1]["content"]


async def test_an_offline_pass_that_finds_an_improvement_keeps_its_step():
    archive = SQLiteArchive(":memory:")
    search = ThresholdSearch(step=0.2, min_step=0.05, names=["ask"])
    seed = thresholds_artifact("t", {"ask": 0.5})
    first_pass = [v async for v in search.propose(seed, signal=None, archive=archive, budget=SearchBudget())]
    better = first_pass[0].artifact  # suppose 0.7 won and was kept
    second_pass = [v async for v in search.propose(better, signal=None, archive=archive, budget=SearchBudget())]
    assert [v.metadata["step"] for v in second_pass] == [0.2, 0.2]
