"""A personal agent that creates writing blocks in a persistent local calendar."""

from __future__ import annotations

from datetime import date
import json
from pathlib import Path
import sys
import tempfile
import uuid

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from helix.agent import Agent
from helix.artifact import Subtype, genesis
from helix.tools import tool

PROMPT = """You are a personal calendar agent. Complete the user's request by
creating one 45-minute Writing event in the local demonstration calendar.
Read the calendar, choose an available start, book it, then request feedback
using review_booking. The tools modify a local JSON file, not a live account.
Earlier interactions are dated experiences, not permanent user preferences.
Current explicit instructions take priority over recalled history. When the
request leaves the time open, use relevant prior outcomes to choose a slot.
If no experience helps, use 14:00 when available. Never double-book or change
another event. Report what you created and quote the user's feedback exactly,
including a rejection or a changed preference. A rejection does not move the
event: leave it as created and report that fact. Pending feedback is unknown.
For a history-only request, summarize supplied past interactions without tools.
"""


def load_requests() -> list[dict]:
    path = Path(__file__).parent / "personal/requests.json"
    return json.loads(path.read_text(encoding="utf-8"))


class CalendarEnvironment:
    """Persist calendar actions; reveal scripted user feedback only after booking.

    Feedback fixtures model a user's review, not model predictions. Neither
    the prompt nor read_calendar receives the fixture's preferred start.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self.request: dict | None = None
        self.result: dict | None = None

    def begin(self, request: dict) -> None:
        if request["user"] not in {"maya", "leo"}:
            raise ValueError("Unknown calendar user.")
        date.fromisoformat(request["date"])
        self.request, self.result = dict(request), None

    def _load(self) -> dict:
        return json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {"events": []}

    def _save(self, data: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
        temporary.replace(self.path)

    def read(self) -> dict:
        if self.request is None:
            return {"error": "No active calendar request."}
        events = [event for event in self._load()["events"]
                  if event["user"] == self.request["user"] and event["date"] == self.request["date"]]
        starts = []
        for start in ("09:00", "14:00"):
            minute = int(start[:2]) * 60
            overlaps = any(
                minute < int(event["start"][:2]) * 60 + int(event["start"][3:]) + event["duration_minutes"]
                and minute + 45 > int(event["start"][:2]) * 60 + int(event["start"][3:])
                for event in events
            )
            if not overlaps:
                starts.append(start)
        return {"user": self.request["user"], "date": self.request["date"],
                "duration_minutes": 45, "available_starts": starts, "events": events}

    def book(self, start: str, duration_minutes: int = 45) -> dict:
        if self.request is None:
            return {"error": "No active calendar request."}
        if self.result is not None:
            return {"error": "One booking attempt per request; report its result."}
        if duration_minutes != 45 or start not in self.read()["available_starts"]:
            self.result = {"valid": False, "accepted": None, "feedback": None,
                           "error": "Choose an available start and a 45-minute duration."}
            return dict(self.result)
        self.result = {"event_id": str(uuid.uuid4()), "user": self.request["user"],
                       "date": self.request["date"], "title": "Writing", "start": start,
                       "duration_minutes": duration_minutes, "valid": True,
                       "accepted": None, "feedback": None}
        data = self._load()
        data["events"].append(dict(self.result))
        self._save(data)
        return dict(self.result)

    def review(self) -> dict:
        if self.result is None:
            return {"error": "Create a booking before requesting feedback."}
        if not self.result["valid"] or self.result["accepted"] is not None:
            return dict(self.result)
        feedback = self.request.get("feedback")
        if feedback is None:
            return dict(self.result)
        accepted = self.result["start"] == feedback["accepted_start"]
        self.result.update(accepted=accepted,
                           feedback=feedback["if_accepted" if accepted else "if_rejected"])
        data = self._load()
        for event in data["events"]:
            if event["event_id"] == self.result["event_id"]:
                event.update(self.result)
        self._save(data)
        return dict(self.result)

    def reward(self, trajectory) -> float | None:
        """A valid first booking earns one only when explicitly accepted.

        Invalid attempts and explicit rejection earn zero. Pending feedback
        on a valid booking gives no learning target.
        """
        if self.result is None:
            return None
        if not self.result["valid"]:
            return 0.0
        if self.result["accepted"] is None:
            return None
        return float(self.result["accepted"])


def build_agent(tier, calendar: CalendarEnvironment, *, model: str) -> Agent:
    @tool(description="Read the current user's requested date in the local calendar; return available starts.")
    async def read_calendar() -> dict:
        return calendar.read()

    @tool(description="Create one Writing event in the local calendar at an available HH:MM start.")
    async def book_time(start: str, duration_minutes: int = 45) -> dict:
        return calendar.book(start, duration_minutes)

    @tool(description="Request the user's scripted review of the created booking; may return pending feedback.")
    async def review_booking() -> dict:
        return calendar.review()

    agent = Agent(
        system_prompt=genesis("prompt.personal.calendar", Subtype.PROMPT, PROMPT),
        tools=[read_calendar, book_time, review_booking], model=model,
        max_iterations=5, max_tool_calls=3, memory_tiers={"episodic": tier},
    )
    agent.memory_hooks.episodic_k = 1
    return agent


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="calendar-listing-") as directory:
        calendar = CalendarEnvironment(Path(directory) / "calendar.json")
        calendar.begin(load_requests()[0])
        print("Available:", ", ".join(calendar.read()["available_starts"]))
        print("Created:", calendar.book("14:00")["start"], "Writing (45 minutes)")
        print("Feedback:", calendar.review()["feedback"])
        print("Reward:", calendar.reward(None))
