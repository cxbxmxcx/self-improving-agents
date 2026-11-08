"""A5, L1: chapter 7's learning mechanisms as source, run as the loop's learners (ARCHITECTURE.md, A5)."""

from __future__ import annotations

import ast
import json
import random
from pathlib import Path

import pytest
from basket import (
    CASES,
    PRICES,
    V1,
    V2,
    WRITTEN,
    CaseAPI,
    agent,
    basket_total,
    make_meta,
    practice,
    serve,
    uses,
    world,
)
from helix.memory.code_skills import skill_tool
from helix.memory.learning_mechanisms import Episode, Writer
from helix.memory.mechanism_runtime import MechanismBudget, UnsafeMechanism, check_mechanism
from helix.metacognition import (
    CALENDAR,
    GENERIC,
    SEEDS,
    DesignPolicy,
    LearnerSelection,
    LearningProblem,
    SourceLearner,
    seed_source,
)
from helix.metacognition.design import (
    approve_mechanism,
    design_learners,
    install_mechanism,
    ranked_mechanisms,
    register_mechanism,
)
from helix.metacognition.mechanisms import resolve_mechanism
from helix.metacognition.sources import CONTROL, PROMPT_FIELDS, SEEDS_DIR, prompt_constants

from examples.expenses import study as expense_study
from metacog import Recording
from metacog.testing import RespondingModel, ScriptedModel, call, text

CHAPTER_7 = Path(__file__).parents[2] / "chapters" / "ch07" / "mechanisms"
EVERY = (*SEEDS, CONTROL)


def code_without_prompts(source: str) -> str:
    """The source's syntax tree, with its prompt constants left out."""
    tree = ast.parse(source)
    tree.body = [node for node in tree.body if not (
        isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id in PROMPT_FIELDS)]
    return ast.dump(tree)


def test_the_seeds_are_chapter_7s_files():
    if not CHAPTER_7.exists():
        pytest.skip("chapter 7 isn't beside this checkout")
    for name in EVERY:
        assert (SEEDS_DIR / f"{name}.py").read_text("utf-8") == (CHAPTER_7 / f"{name}.py").read_text("utf-8"), name


@pytest.mark.parametrize("pack", [GENERIC, CALENDAR], ids=["generic", "calendar"])
def test_a_seed_is_chapter_7s_code_with_a_packs_prompts(pack):
    for name in EVERY:
        original = (SEEDS_DIR / f"{name}.py").read_text("utf-8")
        seeded = seed_source(name, pack)
        check_mechanism(seeded)
        assert code_without_prompts(seeded) == code_without_prompts(original), name
        constants = prompt_constants(seeded)
        assert constants and set(constants) == set(prompt_constants(original)), name
        for constant, value in constants.items():
            assert value == getattr(pack, PROMPT_FIELDS[constant]).replace("{observations}", "{calendars}")


def test_with_the_books_prompts_a_seed_is_chapter_7s_seed():
    for name in EVERY:
        original = (SEEDS_DIR / f"{name}.py").read_text("utf-8")
        assert prompt_constants(seed_source(name, CALENDAR)) == prompt_constants(original), name


def episode(case_id: str, success: bool, cases: dict = CASES) -> Episode:
    case = cases[case_id]
    steps = [{"tool": "price", "arguments": {"item": item}, "result": case["prices"][item]} for item in case["items"]]
    if case["tax"]:
        steps.append({"tool": "tax", "arguments": {}, "result": case["tax"]})
    return Episode(case_id, f"What do {', '.join(case['items'])} cost?", steps, success,
                   "accepted" if success else f"the total is {case['expected']}")


EPISODES = [episode("A", True), episode("D1", False), episode("B", True)]


def scripted_writer() -> RespondingModel:
    """The writer's first reply sums the prices (wrong once tax appears); every later reply adds the tax."""
    replies = iter([V1, *[V2] * 40])
    return RespondingModel(lambda request: text(f"```python\n{next(replies)}```"))


@pytest.mark.parametrize("name", EVERY)
async def test_each_seed_run_as_source_learns_what_its_class_learns(name):
    runs = []
    for learner in (resolve_mechanism(name, GENERIC), SourceLearner.seed(name, GENERIC)):
        model = scripted_writer()
        with model.active():
            learned = await learner.learn(EPISODES, practice, Writer(model="gpt-5.4-mini"), basket_total.contract)
        runs.append(([r["messages"] for r in model.requests], learned.source, learned.writer_calls, learned.values,
                     learned.notes))
    assert runs[0] == runs[1]
    assert runs[0][2] >= 1


async def test_the_value_learning_seed_replays_d1s_recorded_learning(tmp_path):
    """Run as source, chapter 7's value-learning seed makes exactly the writer calls D1 recorded (DOMAINS.md, D1)."""
    recorded = Path(__file__).parents[1] / "examples" / "expenses" / "recorded" / "study.jsonl"
    if not recorded.exists():
        pytest.skip("no recorded D1 study")
    args = expense_study.parse(["--phases", "learn", "--recording", str(recorded), "--output", str(tmp_path)])
    recording = Recording(recorded)
    with recording.active():
        by_class = await expense_study.phase_learn(args, recording)
        by_source = await expense_study.phase_learn(args, recording, learner=SourceLearner.seed("value_learning"))
    assert by_source == by_class


async def test_a_procedure_can_have_its_own_learner():
    meta = make_meta(learners={"basket_total": SourceLearner.seed("skill_synthesis")})
    meta.install("basket_total", V1)
    shop = agent(meta)
    model = ScriptedModel([*uses(["apple", "pear"], 5), *uses(["apple", "pear"], 5), *uses(["pear", "pear"], 6),
                           text("7"), text("8"), WRITTEN])
    with model.active():
        for case_id in ("S1", "D1", "D2", "A", "B"):
            last = await serve(shop, case_id)
    attempt = meta.store.learning()[-1]
    assert (attempt["mechanism"], attempt["writer_calls"], attempt["outcome"]) == ("skill_synthesis", 1,
                                                                                    "passed, installed")
    assert last.metacognition.events[-1].startswith("learner: skill_synthesis wrote basket_total from 2 runs")
    assert meta.store.trusted("basket_total")["created_by"] == "skill_synthesis"
    assert model.replies == []


async def test_a_mechanism_over_its_budget_is_logged_with_what_it_spent():
    learner = SourceLearner.seed("value_learning", budget=MechanismBudget(writer_calls=1))
    meta = make_meta(learners={"basket_total": learner})
    meta.install("basket_total", V1)
    shop = agent(meta)
    model = ScriptedModel([*uses(["apple", "pear"], 5), *uses(["apple", "pear"], 5), *uses(["pear", "pear"], 6),
                           text("7"), text("8"), WRITTEN])
    with model.active():
        for case_id in ("S1", "D1", "D2", "A", "B"):
            await serve(shop, case_id)
    attempt = meta.store.learning()[-1]
    assert attempt["outcome"].startswith("failed: MechanismStopped") and attempt["writer_calls"] == 1
    assert meta.store.trusted("basket_total") is None  # still quarantined; the next attempt is scheduled as usual


def test_unsafe_mechanism_source_is_refused_when_it_is_declared():
    with pytest.raises(UnsafeMechanism):
        SourceLearner("import os\n\n\nasync def learn(lab):\n    return os.getcwd()\n")
    with pytest.raises(ValueError, match="learners name no procedure"):
        make_meta(learners={"no_such_procedure": "skill_synthesis"})


# ------------------------------------------------------------------ L2: the loop chooses a learner

def shop_and_writer(request: dict):
    """The scripted shop and writer. The shop calls the procedure when it is offered and otherwise answers the
    total with tax (the drift has come). The writer, with no runs to read, writes the function that ignores
    tax; shown runs, it writes the one that adds it."""
    if request["messages"][0]["content"].startswith("You are writing a Python function"):
        shown = "User's verdict:" in json.dumps(request["messages"])
        return text(f"```python\n{V2 if shown else V1}```")
    items = request["messages"][1]["content"].removeprefix("What do ").removesuffix(" cost?").split(", ")
    results = [m for m in request["messages"] if m["role"] == "tool"]
    offered = {t["function"]["name"] for t in request.get("tools") or []}
    if "basket_total" in offered and not results:
        return call("basket_total", {"items": items})
    if results:
        return text(str(json.loads(results[-1]["content"])))
    return text(str(sum(PRICES[item] + 1 for item in items)))


async def test_when_learning_keeps_failing_a_tournament_chooses_a_learner_that_works(tmp_path):
    store = str(tmp_path / "store")
    meta = make_meta(learner=SourceLearner.seed("no_experience"), store=store,
                     selection=LearnerSelection(candidates=["no_experience", "skill_synthesis"]))
    meta.install("basket_total", V1)
    shop = agent(meta)
    reports = []
    with RespondingModel(shop_and_writer).active():
        for case_id in ("S1", "D1", "D2", "A", "B", "D3", "A", "B", "D3", "A", "B"):
            reports.append((await serve(shop, case_id)).metacognition)
    # Learning without runs to read keeps failing the gate; after two refusals the loop holds a tournament.
    assert [(a["mechanism"], a["outcome"]) for a in meta.store.learning()] == [
        ("no_experience", "refused"), ("no_experience", "refused"), ("tournament:skill_synthesis", "passed, installed")]
    tournament = next(e for r in reports for e in r.events if e.startswith("learner: tournament"))
    assert tournament == ("learner: tournament for basket_total (2 learned versions in a row failed the gate): "
                          "no_experience 0/2 $0.0000, skill_synthesis 2/2 $0.0000; chose skill_synthesis")
    assert all(r.offered and r.verdict.accepted for r in reports[-2:])  # the new version serves
    # The choice is kept in the store, so another process on it learns basket_total the same way.
    again = make_meta(store=store)
    assert again.learner_for("basket_total").name == "skill_synthesis"
    assert again.summary()["learners"] == {"basket_total": "skill_synthesis"}
    # A learned version quarantined soon after it was trusted starts a tournament too.
    for report in reports[-2:]:
        await meta.feedback(report.run_id, False, "a person disagrees")
    assert meta.store.trusted("basket_total") is None
    assert meta._tournament_reason("basket_total") == "the last learned version was quarantined after 2 runs"
    assert make_meta()._tournament_reason("basket_total") is None  # no tournaments without a selection


# ------------------------------------------------------------------ L3: designing a learner

# A design the scripted designer proposes: skill synthesis whose writer prompt asks the writer to consider every
# run. The scripted writer adds the tax only when asked that way, so only this design learns the tax classes.
MARK = "Consider every run."
DESIGNED = seed_source("skill_synthesis").replace("{contract}", "{contract}\n\n" + MARK, 1)
# The same writer prompt in a mechanism that writes once and never checks itself on the runs it learns from.
ONE_SHOT = seed_source("success_distillation").replace("{contract}", "{contract}\n\n" + MARK, 1)
# Exempt baskets: the tax tool reports a tax, but these baskets are paid without it.
EXEMPT = {"X1": {**world(["apple"], 1), "expected": 2}, "X2": {**world(["pear", "pear"], 1), "expected": 6},
          "X3": {**world(["apple", "pear"], 1), "expected": 5}}
MORE_TAX = {"T1": world(["apple"], 1), "T2": world(["pear", "pear"], 1), "T3": world(["apple", "apple"], 1)}


def practice_on(cases: dict):
    """Function-level practice on a class of baskets of its own."""
    async def tried(source: str, case_id: str):
        case = cases[case_id]
        got = await skill_tool("basket_total", "", source, bind={"api": CaseAPI(case)}).call(items=case["items"])
        ok = got == case["expected"]
        return (1.0 if ok else 0.0), "accepted" if ok else f"returned {got}, expected {case['expected']}", ""
    return tried


def problem(cases: dict, learn: list[str], held_out: list[str]) -> LearningProblem:
    tried = practice_on(cases)
    return LearningProblem("basket_total", basket_total.contract, [episode(c, True, cases) for c in learn], tried,
                           held_out, tried)


TAX = problem(CASES, ["A", "B"], ["G1", "G2"])


def designer_and_writer(design: str):
    """The scripted designer proposes `design`; the scripted writer adds the tax only when asked to consider
    every run."""
    def respond(request: dict):
        first = request["messages"][0]["content"]
        if "learning mechanism" in first:  # GEPA's mutation or crossover
            return text(f"```python\n{design}```")
        if first.startswith("You are writing a Python function"):
            return text(f"```python\n{V2 if MARK in json.dumps(request['messages']) else V1}```")
        raise AssertionError(f"an unexpected model call: {first[:80]!r}")
    return respond


async def test_a_design_that_learns_better_and_generalizes_is_accepted_and_competes(tmp_path):
    with RespondingModel(designer_and_writer(DESIGNED)).active():
        design = await design_learners([TAX], [problem(MORE_TAX, ["T1"], ["T2", "T3"])], population=2,
                                       generations=1, rng=random.Random(7))
    assert design.accepted and design.source.strip() == DESIGNED.strip(), design.reason
    assert (design.train, design.test) == (1.0, 1.0)
    assert all(scores["train"] == 0.0 for scores in design.seeds.values())  # every seed writes the tax-free total
    assert design.reason.startswith(f"beats {design.best_seed} on training (1.00 vs 0.00)")
    # Registered, it waits for a person; approved, it competes in the loop's learner tournaments.
    meta = make_meta(store=str(tmp_path / "store"), selection=LearnerSelection(candidates=["skill_synthesis"]))
    name = register_mechanism(meta, design)
    assert [c.name for c in meta._tournament_candidates()] == ["skill_synthesis"]
    approve_mechanism(meta, name)
    assert [c.name for c in meta._tournament_candidates()] == ["skill_synthesis", name]


async def test_a_design_that_only_wins_on_training_is_refused():
    with RespondingModel(designer_and_writer(ONE_SHOT)).active():
        design = await design_learners([TAX], [problem(EXEMPT, ["X1"], ["X2", "X3"])], population=2,
                                       generations=1, rng=random.Random(7))
    # It learns the training class but never checks itself, and on exempt baskets the best seed was right.
    # (Skill synthesis with the same prompt repairs itself on an exempt basket it learns from, and holds.)
    assert not design.accepted and (design.train, design.test) == (1.0, 0.0)
    assert "loses on the test problems (0.00 vs 1.00)" in design.reason
    with pytest.raises(ValueError, match="was not accepted"):
        register_mechanism(make_meta(), design)


async def test_the_cli_lists_and_approves_designed_mechanisms(tmp_path, capsys):
    from metacog.cli import main

    with RespondingModel(designer_and_writer(DESIGNED)).active():
        design = await design_learners([TAX], [problem(MORE_TAX, ["T1"], ["T2", "T3"])], population=2,
                                       generations=1, rng=random.Random(7))
    store = str(tmp_path / "store")
    name = register_mechanism(make_meta(store=store), design)
    main(["mechanisms", store])
    assert f"{name}  proposed  transfers  from value_learning  train 1.00  test 1.00" in capsys.readouterr().out
    main(["approve-mechanism", store, name])
    assert capsys.readouterr().out.strip() == f"{name} may compete in learner tournaments"


async def test_when_no_mechanism_learns_a_procedure_the_worker_designs_one(tmp_path):
    def respond(request: dict):
        first = request["messages"][0]["content"]
        if "learning mechanism" in first:  # the designer
            return text(f"```python\n{DESIGNED}```")
        if first.startswith("You are writing a Python function"):  # only asked to consider every run, it adds tax
            return text(f"```python\n{V2 if MARK in json.dumps(request['messages']) else V1}```")
        return shop_and_writer(request)  # the shop

    meta = make_meta(learning="worker", store=str(tmp_path / "store"),
                     selection=LearnerSelection(candidates=["value_learning", "skill_synthesis"],
                                                design=DesignPolicy(population=2, generations=1)))
    meta.install("basket_total", V1)
    shop = agent(meta)
    events = []
    with RespondingModel(respond).active():
        for batch in (("S1", "D1", "D2", "A", "B"), ("D3", "A"), ("B", "D3")):
            for case_id in batch:
                await serve(shop, case_id)
            events += await meta.learn_due()  # the worker's pass
    [name] = meta.store.setting("mechanisms")
    # Learning failed twice, the tournament's winner failed too, so the worker designed a mechanism.
    assert [(a["mechanism"], a["outcome"]) for a in meta.store.learning()] == [
        ("value_learning", "refused"), ("value_learning", "refused"), ("tournament:value_learning", "refused"),
        (f"design:{name}", "design proposed")]
    assert events[-1] == (f"learner: designed {name} for basket_total (beats value_learning on training (1.00 vs "
                          "0.00); no test problems from other task classes, so a person must judge whether it "
                          "generalizes); it competes once a person approves it")
    assert meta.store.setting("mechanisms")[name]["state"] == "proposed"
    approve_mechanism(meta, name)
    assert [c.name for c in meta._tournament_candidates()] == ["value_learning", "skill_synthesis", name]


NO_TAX = {"N1": world(["apple"]), "N2": world(["pear", "pear"]), "N3": world(["apple", "pear"])}


async def test_a_tie_on_the_test_problems_is_refused_unless_ties_are_allowed_and_ranks_below_transfer(tmp_path):
    # On baskets without tax, adding a tax of zero is right too, so the design and the seed tie there.
    verdicts = {}
    for ties in (False, True):
        with RespondingModel(designer_and_writer(DESIGNED)).active():
            verdicts[ties] = await design_learners([TAX], [problem(NO_TAX, ["N1"], ["N2", "N3"])], population=2,
                                                   generations=1, rng=random.Random(7), ties=ties)
    assert (verdicts[False].accepted, verdicts[False].standing) == (False, "refused")
    assert verdicts[False].reason.endswith("ties it on the test problems (1.00 vs 1.00), so there is no sign it "
                                           "transfers")
    assert (verdicts[True].accepted, verdicts[True].standing) == (True, "ties")
    with RespondingModel(designer_and_writer(DESIGNED)).active():
        transfers = await design_learners([TAX], [problem(MORE_TAX, ["T1"], ["T2", "T3"])], population=2,
                                          generations=1, rng=random.Random(7))
    # Registered side by side, the design that transfers ranks first, in listings and in tournaments.
    meta = make_meta(store=str(tmp_path / "store"), selection=LearnerSelection(candidates=["skill_synthesis"]))
    tie = install_mechanism(meta, verdicts[True].source, "tied")
    mechanisms = meta.store.setting("mechanisms")
    mechanisms["tied"].update(standing="ties", test=1.0, train=1.0)
    meta.store.put_setting("mechanisms", mechanisms)
    winner = register_mechanism(meta, transfers)
    approve_mechanism(meta, winner)
    assert [name for name, _ in ranked_mechanisms(meta)] == [winner, tie]
    assert [c.name for c in meta._tournament_candidates()] == ["skill_synthesis", winner, tie]


async def test_validation_inside_the_search_picks_the_design_that_generalizes():
    # The designer first proposes the one-shot design, then the self-checking one. Both learn the tax class
    # perfectly; only the self-checking one also learns the exempt class, which validates every candidate.
    proposals = iter([ONE_SHOT, *[DESIGNED] * 20])

    def respond(request: dict):
        first = request["messages"][0]["content"]
        if "learning mechanism" in first:
            return text(f"```python\n{next(proposals)}```")
        return designer_and_writer(DESIGNED)(request)

    with RespondingModel(respond).active():
        design = await design_learners([TAX], [problem(MORE_TAX, ["T1"], ["T2", "T3"])],
                                       validation=[problem(EXEMPT, ["X1"], ["X2", "X3"])], population=2,
                                       generations=2, rng=random.Random(7))
    assert design.source.strip() == DESIGNED.strip() and design.validation == 1.0
    assert (design.accepted, design.standing, design.train, design.test) == (True, "transfers", 1.0, 1.0)
    # The one-shot design was measured too: as good on training, and nothing on validation.
    assert {(row["score"], row["validation"]) for row in design.lineage if row["created_by"] != "value_learning"} >= {
        (1.0, 0.0), (1.0, 1.0)}
