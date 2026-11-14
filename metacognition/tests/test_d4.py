"""D4 (DOMAINS.md): the D1 and D3 studies on Claude Haiku 5.5 and Sonnet 5.5, replayed from their recordings.

Live once on 2026-11-07, through helix.metacognition.claude (Anthropic's SDK).
gpt-5.4-mini's results are D1's and D3's own.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from helix.metacognition.offline import Floors

from examples.expenses import study as d1
from examples.invoices import study as d3

ROOT = Path(__file__).parents[1] / "examples"
EFFORTS = ("none", "low", "medium", "high")
HAIKU, SONNET = "claude-haiku-5-5", "claude-sonnet-5-5"
D1_RECORDING = {HAIKU: ROOT / "expenses" / "recorded" / "d4_haiku.jsonl",
                SONNET: ROOT / "expenses" / "recorded" / "d4_sonnet.jsonl"}
D3_RECORDING = {HAIKU: ROOT / "invoices" / "recorded" / "d4_haiku.jsonl",
                SONNET: ROOT / "invoices" / "recorded" / "d4_sonnet.jsonl",
                "gpt-5.4-mini": ROOT / "invoices" / "recorded" / "d4_mini.jsonl"}
MEMORIES = ROOT / "expenses" / "memories"


def solved(path: Path, arm: str) -> list[int]:
    cells = Floors.load(path).sweeps[arm].cells
    return [cells[e].solved for e in EFFORTS if e in cells]


def needs(path: Path) -> None:
    if not path.exists():
        pytest.skip(f"no recording {path.name}")


@pytest.mark.parametrize(("model", "without", "phases"), [
    (HAIKU, [12, 15, 16, 15], ["floors", "learn", "memory", "online"]),
    (SONNET, [15, 16, 16, 16], ["floors", "learn", "memory"]),
])
async def test_d1_on_claude_learns_a_procedure_that_holds_at_every_setting(model, without, phases, tmp_path, capsys):
    needs(D1_RECORDING[model])
    await d1.main(d1.parse(["--model", model, "--phases", *phases, "--recording", str(D1_RECORDING[model]),
                            "--output", str(tmp_path)]))
    out = capsys.readouterr().out
    assert solved(tmp_path / "floors_no_memory.json", "no_memory") == without
    assert solved(tmp_path / "floors_memory_learned.json", "memory") == [16, 16, 16, 16]
    assert "validation 16/16 accepted, 16/16 match the hand-written reference" in out
    # A procedure another model wrote gives the same answers, so its runs are the same calls.
    await d1.main(d1.parse(["--model", model, "--phases", "memory", "--memory", "file", "--procedure-file",
                            str(MEMORIES / "learned_by_gpt_5_4_mini.py"), "--recording",
                            str(D1_RECORDING[model]), "--output", str(tmp_path)]))
    assert solved(tmp_path / "floors_memory_file.json", "memory") == [16, 16, 16, 16]


@pytest.mark.parametrize("writer", ["claude_haiku_5_5", "claude_sonnet_5_5"])
async def test_gpt_5_4_mini_serves_a_procedure_a_claude_writer_learned(writer, tmp_path):
    study = ROOT / "expenses" / "recorded" / "study.jsonl"
    needs(study)
    await d1.main(d1.parse(["--phases", "memory", "--memory", "file", "--procedure-file",
                            str(MEMORIES / f"learned_by_{writer}.py"), "--recording", str(study),
                            "--output", str(tmp_path)]))
    assert solved(tmp_path / "floors_memory_file.json", "memory") == [16, 16, 16, 16]


@pytest.mark.parametrize(("model", "without", "tool"), [
    (HAIKU, [10, 10, 10, 10], [16, 15, 13, 12]),
    (SONNET, [10, 10, 10, 10], [10, 11, 10, 9]),
])
async def test_d3_on_claude_the_more_it_reasons_the_more_it_overrules_the_payable(model, without, tool, tmp_path):
    needs(D3_RECORDING[model])
    await d3.main(d3.parse(["--model", model, "--phases", "floors", "memory", "--recording",
                            str(D3_RECORDING[model]), "--output", str(tmp_path)]))
    assert solved(tmp_path / "floors_no_memory.json", "no_memory") == without
    assert solved(tmp_path / "floors_memory.json", "memory") == tool


@pytest.mark.parametrize(("model", "efforts"), [
    (HAIKU, list(EFFORTS)), (SONNET, ["none", "low"]), ("gpt-5.4-mini", list(EFFORTS))])
async def test_d3_told_to_act_on_the_payables_answer_every_model_gets_every_payment(model, efforts, tmp_path):
    needs(D3_RECORDING[model])
    await d3.main(d3.parse(["--model", model, "--contract", "act", "--phases", "memory", "--efforts", *efforts,
                            "--recording", str(D3_RECORDING[model]), "--output", str(tmp_path)]))
    assert solved(tmp_path / "floors_memory.json", "memory") == [16] * len(efforts)


async def test_d3_on_haiku_with_the_payable_gpt_5_4_minis_designed_learner_wrote(tmp_path):
    needs(D3_RECORDING[HAIKU])
    payable = ROOT / "invoices" / "memories" / "amount_payable_designed_mini.py"
    await d3.main(d3.parse(["--model", HAIKU, "--phases", "memory", "--payable-file", str(payable),
                            "--recording", str(D3_RECORDING[HAIKU]), "--output", str(tmp_path)]))
    assert solved(tmp_path / "floors_memory.json", "memory") == [16, 11, 11, 10]
