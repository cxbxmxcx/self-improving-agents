"""A5, L1: chapter 7's learning mechanisms as source, run as the loop's learners (ARCHITECTURE.md, A5)."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest
from basket import CASES, PRICES, V1, V2, WRITTEN, agent, basket_total, make_meta, practice, serve, uses
from helix.memory.learning_mechanisms import Episode, Writer
from helix.memory.mechanism_runtime import MechanismBudget, UnsafeMechanism, check_mechanism

from examples.expenses import study as expense_study
from metacog import Recording
from metacog.meta import CALENDAR, GENERIC, SEEDS, LearnerSelection, SourceLearner, seed_source
from metacog.meta.mechanisms import resolve_mechanism
from metacog.meta.sources import CONTROL, PROMPT_FIELDS, SEEDS_DIR, prompt_constants
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


def episode(case_id: str, success: bool) -> Episode:
    case = CASES[case_id]
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
