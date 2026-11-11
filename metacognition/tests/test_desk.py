"""The calendar desk of the production-scale test, on a short stream with scripted models (no key, no spend).

`examples/scale/desk_stream.py` builds the stream, `desk_run.py` serves it with three assistants, and `desk_report.py`
turns their results into the nine pass criteria. The short stream's weeks are drawn once for the module, on every
processor, which takes most of a minute.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from examples.scale import desk_report, desk_run, desk_stream

SHORT = ["--per-day", "12", "--days", "10", "--held", "8"]


def test_the_writers_templates_are_complete_and_kept_as_written():
    banks = json.loads(desk_stream.BANKS.read_text(encoding="utf-8"))
    assert desk_stream.check_banks(banks) == []
    assert set(banks) == {"about", *desk_stream.GROUPS, "exam"} and len(banks["about"]["writers"]) == 7
    broken = {**banks, "g1": {**banks["g1"], "single": ["Book a {topic} with {who}."] * 12}}
    faults = desk_stream.check_banks(broken)
    assert any("12 templates, 1 distinct" in fault for fault in faults)  # copies are not twelve templates
    assert any("needs one of ['minutes']" in fault for fault in faults)  # and a booking has to say how long


def test_the_stream_changes_five_times_on_the_days_the_plan_names():
    days = {day.number: day for day in desk_stream.plan(20)}
    assert {number: day.change for number, day in days.items() if day.change} == {
        5: "Dana's note asks for half an hour", 8: "two more groups of writers",
        11: "the calendar is down for one request in ten", 14: "requests for the same time on several days arrive",
        17: "Dana's note asks for three quarters of an hour"}
    assert [days[n].note.buffer for n in (4, 5, 16, 17)] == [15, 30, 30, 45]
    assert len(days[7].groups) == 4 and len(days[8].groups) == 6
    assert not days[13].second_class and days[14].second_class and [d for d in days if days[d].outage] == [11]
    assert Counter(desk_stream.mix(200, False)) == {"one": 100, "two": 24, "near": 60, "out": 16}
    assert Counter(desk_stream.mix(200, True)) == {"one": 85, "two": 20, "near": 51, "out": 14, "same": 30}


@pytest.fixture(scope="module")
def weeks(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("desk") / "weeks.json"
    desk_stream.Stream(12, 10, weeks=desk_stream.Weeks(path), held=8).prepare()
    return path


def test_a_week_marked_sensitive_is_booked_wrongly_by_every_stale_reading_of_the_note(weeks: Path):
    stream = desk_stream.Stream(12, 10, weeks=desk_stream.Weeks(weeks), held=8)
    stream.prepare()
    changed = [day for day in stream.days if day.note.stale]
    requests = [request for day in changed for request in stream.served(day) if request["expect"] == "act"]
    sensitive = [request for request in requests if request["sensitive"]]
    assert 0.4 < len(sensitive) / len(requests) < 0.6  # every other request for a booking
    for request in sensitive[:12]:
        assert "Dana Park" in request["truth"]["attendees"]
        right = desk_stream._answers(request)
        note = desk_stream.NOTES[request["note"]]
        stale = note.stale[:1] if request["class"] == "same_time" else note.stale
        for reading in stale:
            guessed = desk_stream._answers(desk_stream._with_buffer(request, reading))
            assert not desk_stream._first_answer_is_right(request, guessed, right)
    texts = [request["text"] for day in stream.days for request in stream.served(day)]
    assert len(set(texts)) == len(texts)  # no two requests share their words, so none is served from another's reply
    outside = [request["text"] for day in stream.days for request in stream.served(day) if request["asked"] == "out"]
    assert outside and not any("remind" in text.lower() for text in outside)  # an ambiguous label is left out


async def test_a_soak_serves_the_short_stream_with_all_three_assistants_and_measures_the_criteria(
        weeks: Path, tmp_path: Path):
    common = ["--soak", *SHORT, "--weeks", str(weeks), "--output", str(tmp_path)]
    full = await desk_run.run_arm("full", desk_run.parse(["full", *common]))
    loop = await desk_run.run_arm("loop", desk_run.parse(["loop", *common]))
    alone = await desk_run.run_arm("alone", desk_run.parse(["alone", *common]))
    await desk_run.run_searches(desk_run.parse(["searches", *common, "--searches", "2"]))

    by_phase = Counter(row["phase"] for row in full["rows"])
    assert full["stopped"] == "" and by_phase == {"stream": 120, "exam": 12, "repeat2": 12, "repeat3": 12}
    exam_events = [event for row in full["rows"] if row["phase"] != "stream" for event in row["events"]]
    assert not any(event.startswith(("detector:", "learner:", "program: switched")) for event in exam_events)
    assert any("installed" in event for event in full["nights"][0]["events"])  # the first night's search
    learned = Counter(entry["name"] for entry in full["learning"] if "installed" in entry["outcome"])
    # Both changes of note have the first function relearned, and the second kind of work brings a second function
    # that the loop learns by itself and then relearns when the note changes again.
    assert learned["find_shared_starts"] >= 2 and learned["find_common_starts"] >= 2
    stream = [row for row in full["rows"] if row["phase"] == "stream"]
    assert sum(row["right"] for row in stream) >= 0.9 * len(stream)
    assert sum(row["outcome"] == "program" for row in stream) >= 0.3 * len(stream)
    assert sum(row["phase"] == "stream" for row in loop["rows"]) == 24 and loop["stopped"] == ""
    assert [row["seq"] for row in alone["rows"]] == [0, 25, 50, 75, 100]
    assert all(row["effort"] == "high" and row["outcome"] in ("llm", "llm_wrong") for row in alone["rows"])

    out = desk_report.report(tmp_path)
    assert [c["number"] for c in out["criteria"]] == list(range(1, 10)) and out["missing"] == []
    assert out["verdict"] in ("holds", "holds in part", "does not hold")
    assert [change["day"] for change in out["changes"]] == [2, 4, 6, 7, 8]  # the five changes, on a ten-day stream
    assert out["arms"]["loop"]["requests_whole_stream"] == out["arms"]["alone"]["requests_whole_stream"] == 120
    assert out["consistency"]["full"]["requests"] == 12 and len(out["searches"]) == 2


def test_a_part_cannot_record_without_a_budget_and_a_soak_cannot_record():
    with pytest.raises(SystemExit):
        desk_run.parse(["full", "--record"])
    with pytest.raises(SystemExit):
        desk_run.parse(["full", "--record", "--soak", "--budget", "5"])
    assert desk_run.parse(["loop", "--record", "--budget", "5"]).budget == 5
