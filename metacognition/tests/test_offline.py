"""The offline toolkit and the CLI on the scripted basket domain."""

from __future__ import annotations

from basket import CASES, V1, V2, agent, make_meta, practice, serve, uses, verifier
from helix.eval.effort import EffortCell, EffortSweep, RunRecord

from metacog import cli
from metacog.meta import SOLID, Controller
from metacog.offline import Case, Floors, calibrate, practice_evaluator
from metacog.testing import ScriptedModel, text


def sweep_of(arm: str, solved: dict[str, int], total: int = 8) -> EffortSweep:
    cells = {e: EffortCell(e, [RunRecord(f"t{i}", e, i < n) for i in range(total)]) for e, n in solved.items()}
    return EffortSweep(arm, cells)


def test_floor_tables_choose_the_cheapest_setting_that_clears_the_bar(tmp_path):
    floors = Floors({"memory": sweep_of("memory", {"none": 8, "low": 8, "medium": 8, "high": 8}),
                     "no_memory": sweep_of("no_memory", {"none": 2, "low": 5, "medium": 7, "high": 8})})
    assert floors.table() == {"memory": "none", "no_memory": "medium"}
    assert floors.table(SOLID) == {"memory": "none", "no_memory": "high"}
    weak = Floors({"no_memory": sweep_of("no_memory", {"none": 1, "low": 4, "medium": 4, "high": 3})})
    assert weak.table() == {"no_memory": "medium"}  # no floor: the best record, ties to more reasoning
    path = floors.save(tmp_path)
    assert Controller.from_floors(path).table == {"memory": "none", "no_memory": "medium"}
    assert Floors.load(path).table() == floors.table()


async def test_practice_evaluator_counts_a_case_only_when_every_input_variant_passes():
    async def variant(source: str, case_id: str):
        return (0.0, "rejected when items are upper case", "called with APPLE") if case_id == "G2" else (1.0, "", "")

    evaluate = practice_evaluator(practice, ["G1", "G2", "D1"], variants=[variant])
    measured = await evaluate(V2)
    assert (measured.metadata["accepted"], measured.score) == (2, 2 / 3)
    assert "G2: rejected when items are upper case (input variant 1)" in measured.feedback
    assert "called with APPLE" in measured.feedback
    stale = await evaluate(V1)
    assert stale.metadata["accepted"] == 0 and "G1: returned 2, expected 3" in stale.feedback


async def test_calibrate_measures_each_arm_without_recording():
    meta = make_meta()
    meta.install("basket_total", V2)
    cases = [Case("A", "What do apple, pear cost?", CASES["A"]), Case("B", "What do pear, pear cost?", CASES["B"])]
    # One case at a time, efforts in order; without memory the agent guesses wrong at none and right at high.
    model = ScriptedModel([text("5"), text("6"), text("7"), text("8"),
                           *uses(["apple", "pear"], 7), *uses(["pear", "pear"], 8),
                           *uses(["apple", "pear"], 7), *uses(["pear", "pear"], 8)])
    with model.active():
        floors = await calibrate(agent(meta), cases, verifier=verifier, efforts=("none", "high"), concurrency=1)
    assert floors.table(SOLID) == {"memory": "none", "no_memory": "high"}
    assert [r["reasoning_effort"] for r in model.requests[::1]][:4] == ["none", "none", "high", "high"]
    assert meta.store.count() == 0


async def test_cli_status_runs_show_quarantine_and_report(tmp_path, capsys):
    store = tmp_path / "shop.meta"
    meta = make_meta(store=str(store))
    meta.install("basket_total", V1)
    with ScriptedModel(uses(["apple", "pear"], 5) * 2).active():
        await serve(agent(meta), "S1")
        await serve(agent(meta), "S1")
    assert cli.main(["status", str(store)]) == 0
    out = capsys.readouterr().out
    assert "2 runs" in out and "basket_total: v1 trusted (human)" in out
    assert cli.main(["runs", str(store)]) == 0
    assert "basket_total@1" in capsys.readouterr().out
    assert cli.main(["show", str(store), "basket_total"]) == 0
    assert "return sum(api.price(item) for item in items)" in capsys.readouterr().out
    assert cli.main(["approve", str(store), "basket_total", "1"]) == 1  # not awaiting approval
    assert cli.main(["quarantine", str(store), "basket_total", "--reason", "prices changed"]) == 0
    assert meta.store.trusted("basket_total") is None
    page = tmp_path / "report.html"
    assert cli.main(["report", str(store), "--html", str(page)]) == 0
    assert "Metacognition report" in page.read_text(encoding="utf-8")
