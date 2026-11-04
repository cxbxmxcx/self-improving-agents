"""A personal agent creates calendar events and learns from explicit feedback."""

from importlib import import_module
import json
from pathlib import Path
import subprocess
import sys

import pytest


def calendar_module():
    return import_module("chapters.ch05.02_calendar_agent")


def walkthrough():
    return import_module("chapters.ch05.03_memory_learning")


def test_booking_persists_and_refuses_conflicts_without_overwriting(tmp_path):
    module = calendar_module()
    environment = module.CalendarEnvironment(tmp_path / "calendar.json")
    request = module.load_requests()[0]
    environment.begin(request)
    before = environment.read()
    assert before["available_starts"] == ["09:00", "14:00"]
    booking = environment.book("14:00")
    assert booking["valid"] is True
    assert booking["accepted"] is None
    saved = json.loads(environment.path.read_text())
    assert len(saved["events"]) == 1
    assert "error" in environment.book("09:00")
    environment.begin(dict(request, id="another-request"))
    assert "error" in environment.book("14:00")
    assert json.loads(environment.path.read_text()) == saved


def test_feedback_is_hidden_until_review_and_missing_feedback_is_not_failure(tmp_path):
    module = calendar_module()
    environment = module.CalendarEnvironment(tmp_path / "calendar.json")
    request = module.load_requests()[2]
    environment.begin(request)
    assert "09:00" not in request["question"]
    assert "preferred" not in json.dumps(environment.read()).lower()
    assert "Mornings work better" not in json.dumps(environment.read())
    assert "error" in environment.review()
    assert environment.reward(None) is None
    booking = environment.book("14:00")
    assert booking["accepted"] is None and environment.reward(None) is None
    review = environment.review()
    assert review["valid"] and review["accepted"] is False
    assert "Mornings work better" in review["feedback"]
    assert environment.reward(None) == 0.0
    assert json.loads(environment.path.read_text())["events"][0]["start"] == "14:00"
    environment.begin(dict(module.load_requests()[3], feedback=None))
    environment.book("09:00")
    assert environment.review()["accepted"] is None
    assert environment.reward(None) is None


@pytest.mark.parametrize("change", [{"user": "stranger"}, {"date": "not-a-date"}])
def test_unknown_users_and_invalid_dates_are_rejected(tmp_path, change):
    module = calendar_module()
    environment = module.CalendarEnvironment(tmp_path / "calendar.json")
    with pytest.raises(ValueError):
        environment.begin(dict(module.load_requests()[0], **change))
    assert not environment.path.exists()


def test_booking_validates_duration_and_start(tmp_path):
    module = calendar_module()
    environment = module.CalendarEnvironment(tmp_path / "calendar.json")
    environment.begin(module.load_requests()[0])
    assert "error" in environment.book("09:00", duration_minutes=30)
    assert environment.result["valid"] is False
    assert environment.result["accepted"] is None
    assert environment.reward(None) == 0.0
    assert "error" in environment.book("03:00")
    assert not environment.path.exists()


def test_calendar_reopens_events_and_filters_them_by_user(tmp_path):
    module = calendar_module()
    path = tmp_path / "calendar.json"
    request = module.load_requests()[0]
    environment = module.CalendarEnvironment(path)
    environment.begin(request)
    environment.book("14:00")
    environment.review()
    reopened = module.CalendarEnvironment(path)
    reopened.begin(request)
    assert reopened.read()["available_starts"] == ["09:00"]
    assert reopened.read()["events"][0]["accepted"] is True
    reopened.begin(dict(request, user="leo"))
    assert reopened.read()["events"] == []
    assert reopened.read()["available_starts"] == ["09:00", "14:00"]


def test_numbered_agent_listing_runs_outside_repository(tmp_path):
    script = Path(__file__).resolve().parents[1] / "chapters/ch05/02_calendar_agent.py"
    result = subprocess.run([sys.executable, str(script)], cwd=tmp_path,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert "Created: 14:00 Writing (45 minutes)" in result.stdout
    assert "Reward: 1.0" in result.stdout


async def test_current_explicit_instruction_overrides_past_preference():
    from chapters.ch05.personal.scripted import scripted_response
    result = await scripted_response(messages=[
        {"role": "system", "content": "Rules\n## Relevant past interactions\nMornings work better now; please use 09:00 for writing."},
        {"role": "user", "content": "Schedule 45 minutes for writing on 2026-10-09 at 14:00."},
        {"role": "tool", "content": json.dumps({"available_starts": ["09:00", "14:00"], "date": "2026-10-09"})},
    ])
    call = result.choices[0].message.tool_calls[0]
    assert call.function.name == "book_time"
    assert json.loads(call.function.arguments)["start"] == "14:00"


async def test_accepted_past_booking_guides_the_next_open_time_request():
    from chapters.ch05.personal.scripted import scripted_response
    result = await scripted_response(messages=[
        {"role": "system", "content": "Rules\n## Relevant past interactions\nCreated a 45-minute writing block on 2026-10-07 at 09:00. Booking feedback: accepted."},
        {"role": "user", "content": "Schedule 45 minutes for writing on 2026-10-09."},
        {"role": "tool", "content": json.dumps({"available_starts": ["09:00", "14:00"], "date": "2026-10-09"})},
    ])
    call = result.choices[0].message.tool_calls[0]
    assert json.loads(call.function.arguments)["start"] == "09:00"


async def test_walkthrough_books_then_reviews_and_credits_recalled_experiences(tmp_path, monkeypatch):
    async def forbidden_provider(**_):
        raise AssertionError("The default walkthrough must not call a provider.")
    monkeypatch.setattr("helix.llm_call.litellm.acompletion", forbidden_provider)
    report = await walkthrough().run_walkthrough(tmp_path / "run")
    rows = report["requests"]
    assert [row["reward"] for row in rows] == [1.0, 1.0, 0.0, 1.0]
    assert [row["outcome"]["start"] for row in rows] == ["14:00", "14:00", "14:00", "09:00"]
    assert rows[0]["recalled"] == []
    assert rows[1]["recalled"][0]["episode"] == "E1"
    assert rows[1]["recalled"][0]["q_after"] == 0.75
    assert rows[2]["recalled"][0]["episode"] == "E1"
    assert rows[2]["recalled"][0]["q_after"] == 0.375
    assert rows[3]["recalled"][0]["episode"] == "E3"
    assert rows[3]["recalled"][0]["q_after"] == 0.75
    assert "Mornings work better" in rows[2]["answer"]
    for row in rows:
        assert [call["name"] for call in row["tool_calls"]] == ["read_calendar", "book_time", "review_booking"]
        assert row["available_starts"] == ["09:00", "14:00"]
        for recalled in row["recalled"]:
            assert recalled["current_ref"][1] == recalled["recalled_ref"][1] + 1
    assert report["measurement"] == {
        "requests": 4, "valid_bookings": 4, "invalid_bookings": 0, "accepted": 3, "rejected": 1,
        "unobserved": 0, "acceptance_rate": 0.75,
    }
    assert len(report["handoff"]["maya"]["recalled"]) == 1
    assert report["handoff"]["leo"]["recalled"] == []
    calendar = json.loads((tmp_path / "run/calendar.json").read_text())
    assert len(calendar["events"]) == 4
    assert calendar["events"][2]["start"] == "14:00"
    assert calendar["events"][2]["accepted"] is False
    assert "Mornings work better" in calendar["events"][2]["feedback"]
    from helix.archive import SQLiteArchive
    tier = walkthrough().open_memory(tmp_path / "run", SQLiteArchive(tmp_path / "run/archive.sqlite"))
    episode_id = next(entry_id for entry_id, alias in report["episodes"].items() if alias == "E3")
    retained = await tier.get(episode_id)
    assert "Mornings work better" in retained.content["final_output"]
    assert retained.score == .75
    saved = json.loads((tmp_path / "run/trace.json").read_text())
    assert saved["measurement"] == report["measurement"]
    with pytest.raises(ValueError, match="existing evidence"):
        await walkthrough().run_walkthrough(tmp_path / "run")


def test_measurement_keeps_unobserved_requests_in_the_denominator():
    rows = [
        {"outcome": {"valid": True, "accepted": True}, "reward": 1.0},
        {"outcome": {"valid": True, "accepted": False}, "reward": 0.0},
        {"outcome": {"valid": True, "accepted": None}, "reward": None},
        {"outcome": None, "reward": None},
        {"outcome": {"valid": False, "accepted": None}, "reward": 0.0},
    ]
    assert walkthrough().measure_outcomes(rows) == {
        "requests": 5, "valid_bookings": 3, "invalid_bookings": 1, "accepted": 1, "rejected": 1,
        "unobserved": 2, "acceptance_rate": 0.2,
    }


async def test_live_mode_requires_model_before_creating_output(tmp_path):
    with pytest.raises(ValueError, match="explicit model"):
        await walkthrough().run_walkthrough(tmp_path / "live", live=True)
    assert not (tmp_path / "live").exists()
