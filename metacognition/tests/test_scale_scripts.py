"""The two scale scripts, end to end on small streams (scripted models, no key).

`examples/scale/long_run.py` serves one request at a time for many days, and `examples/scale/load.py` serves many
at once. Each is run here on a stream small enough for a test, with a few weeks of each kind generated into a folder
of the test's own.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from examples.scale import load, long_run, stream
from examples.scale.scripted import NOW


@pytest.fixture
def small(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.setattr(stream, "SERVED_WEEKS", 6)
    monkeypatch.setattr(stream, "HELD_BACK_WEEKS", 6)
    monkeypatch.setitem(NOW, "conditions", "ordinary")
    return tmp_path


def test_the_gate_weeks_of_a_relearned_function_are_named_as_their_generators_name_them():
    for name, generator in stream.GENERATORS.items():
        assert generator.generate(30_001)["id"] == stream.FUNCTION_WEEKS[name][0]


async def test_a_short_stream_recovers_from_each_change_and_asks_the_builder_only_on_a_change(small: Path):
    args = long_run.parse(["--requests", "192", "--per-day", "24", "--quiet-days", "0", "--quiet",
                           "--weeks", str(small / "weeks.json"), "--output", str(small / "out.json")])
    out = await long_run.main(args)
    summary, days = out["summary"], {row["day"]: row for row in out["days"]}
    assert [day for day, row in days.items() if row["changed"]] == [4, 6, 8]
    assert summary["stalled_days"] == [] and summary["false_alarm_days"] == []
    assert summary["changes_checked"] == summary["changes_recovered"] == 1  # day 5 is the calm day after a change
    assert summary["builder_calls_on_calm_days"] == 0
    # The first night's search wrote the component. No change of note had anyone rewrite it: each time the function
    # under the program was found at fault, quarantined and relearned, and that night the program passed again.
    assert out["calls"]["builder: rewrite"] == 1
    for day in (4, 6, 8):
        assert (days[day]["quarantined"], days[day]["relearned"], days[day]["trusted"]) == (1, 1, True)
        # Two wrong actions switch the program off, and that night switches it back on.
        assert days[day]["wrong"] <= 2 and not days[day]["off"]
    # Day 7's requests fail while the calendar is down, which is nobody's fault: nothing is quarantined or relearned.
    assert days[7]["outage"] and days[7]["right"] < days[7]["requests"]
    assert (days[7]["quarantined"], days[7]["relearned"], days[7]["builder"]) == (0, 0, 0)
    assert days[5]["program"] >= 0.9 * days[3]["program"] > 0  # the day after a change, programs serve as before


async def test_a_short_load_run_gives_the_same_outcomes_at_every_level_and_relearns_once(small: Path):
    args = load.parse(["--requests", "40", "--levels", "1", "8", "--agent-seconds", "0.004", "--jev-seconds", "0.002",
                       "--verdict-seconds", "0.004", "--weeks", str(small / "weeks.json"),
                       "--output", str(small / "load.json")])
    out = await load.main(args)
    assert all(row["right"] == row["requests"] for row in out["throughput"])
    assert out["same_outcomes_at_every_level"]
    watched = {row["at_once"]: row for row in out["change"] if row["watched"]}
    alone = next(row for row in out["change"] if not row["watched"])
    for row in watched.values():  # however many were in flight, one quarantine and one relearning repaired it
        assert (row["quarantined"], row["relearnings_started"], row["trusted_at_the_end"]) == (1, 1, True)
    assert watched[1]["right"] >= 36  # one at a time, the change costs the few answers that showed it
    assert (alone["quarantined"], alone["relearnings_started"]) == (0, 0)  # with no loop, the stale function serves on
    assert alone["right"] < watched[alone["at_once"]]["right"]
