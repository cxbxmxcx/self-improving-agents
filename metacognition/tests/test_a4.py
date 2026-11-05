"""A4: the guide's offline steps, run through the CLI on the expense agent from its recorded calls."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from examples.expenses.app import context, write_cases
from examples.expenses.world import task_set
from metacog import Recording, cli
from metacog.meta import MetaStore
from metacog.meta.worker import load_agent
from metacog.run import RunConfig
from metacog.service import create_app

EXAMPLES = Path(__file__).parents[1] / "examples" / "expenses"
SMOKE = EXAMPLES / "recorded" / "smoke.jsonl"
REFERENCE = EXAMPLES / "memories" / "reimbursable_total.py"


def app_on(tmp_path, monkeypatch, name: str) -> str:
    """A target for the guide's app on a store of the test's own."""
    (tmp_path / f"{name}.py").write_text(
        "from examples.expenses.app import build\n\n\n"
        f"def agent():\n    return build(store={str(tmp_path / 'jordan.meta')!r})\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    return f"{name}:agent"


def test_install_and_calibrate_through_the_cli_replay_the_smoke_study(tmp_path, monkeypatch, capsys):
    if not SMOKE.exists():
        pytest.skip("no recorded smoke study")
    target = app_on(tmp_path, monkeypatch, "guide_app_calibrate")
    cases = write_cases(tmp_path / "expense_dev.jsonl", "dev", 4)
    assert cli.main(["install", target, "reimbursable_total", str(REFERENCE)]) == 0
    assert "reimbursable_total@1 is trusted (by human" in capsys.readouterr().out

    assert cli.main(["calibrate", target, str(cases), "--world", "examples.expenses.app:world", "--efforts",
                     "none", "high", "--max-turns", "12", "--recording", str(SMOKE), "--mode", "replay"]) == 0
    out = capsys.readouterr().out
    # The smoke study's numbers: without the procedure 0 and 2 of 4, with it 4 and 4.
    assert "no_memory  none      0/4" in out and "no_memory  high      2/4" in out
    assert "memory     none      4/4" in out and "memory     high      4/4" in out
    assert "recorded floors {'memory': 'none', 'no_memory': 'high'} for gpt-5.4-mini" in out
    assert MetaStore(tmp_path / "jordan.meta").floors() == {
        "gpt-5.4-mini": {"table": {"memory": "none", "no_memory": "high"}}}


def test_install_refuses_code_that_fails_the_guard(tmp_path, monkeypatch, capsys):
    target = app_on(tmp_path, monkeypatch, "guide_app_install")
    unsafe = tmp_path / "unsafe.py"
    unsafe.write_text("import os\n\ndef reimbursable_total(api, trip_id: str) -> dict:\n    return {}\n",
                      encoding="utf-8")
    assert cli.main(["install", target, "reimbursable_total", str(unsafe)]) == 1
    assert "fails the code-skill guard" in capsys.readouterr().err


async def test_a_served_run_can_replay_from_a_recording(tmp_path, monkeypatch):
    if not SMOKE.exists():
        pytest.skip("no recorded smoke study")
    target = app_on(tmp_path, monkeypatch, "guide_app_serve")
    assert cli.main(["install", target, "reimbursable_total", str(REFERENCE)]) == 0
    agent = load_agent(target)
    app = create_app(agent, context=context, run_config=RunConfig(recording=Recording(SMOKE, mode="replay")))
    first = task_set("dev")[0]
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://metacog.test") as http:
        trip = (await http.post("/api/runs", json={"input": first["text"], "case_id": first["id"]},
                                headers={"X-Metacog": "1"})).json()
        unrecorded = await http.post("/api/runs", json={"input": "A request the study never made.",
                                                        "case_id": first["id"]}, headers={"X-Metacog": "1"})
    assert trip["metacognition"]["verdict"]["accepted"] is True and trip["metacognition"]["effort"] == "none"
    assert unrecorded.status_code == 409 and "no recorded reply" in unrecorded.json()["detail"]
