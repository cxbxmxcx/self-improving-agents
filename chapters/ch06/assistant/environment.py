"""Simulate calendar tools and check the outcome of a scheduling request.

Each run uses a local copy of the week, so bookings and cancellations never
touch a real account. The checker compares the result with the request's
parameters and the calendar conventions. Its verdict supplies simulated
acceptance feedback rather than feedback from a person.

Calendars expose local times, UTC offsets, working hours, event status, and
the external guest's request for a buffer between meetings. Internally, times
stay on the user's clock and are converted when another calendar is shown.
The booking tool checks the user's calendar, while the final checker also
checks attendees and request-level constraints.
"""

from __future__ import annotations

import copy
import re
from collections import Counter
from typing import Any

DAYS = ("mon", "tue", "wed", "thu", "fri")
DAY_NAMES = {"mon": "Monday", "tue": "Tuesday", "wed": "Wednesday", "thu": "Thursday", "fri": "Friday"}
BUFFER = 15  # minutes an external guest asks for on each side


def to_minutes(hhmm: str) -> int:
    match = re.fullmatch(r"(\d{1,2}):(\d{2})", str(hhmm).strip())
    if not match:
        raise ValueError(f"times are HH:MM, got {hhmm!r}")
    return int(match.group(1)) * 60 + int(match.group(2))


def to_hhmm(minutes: int) -> str:
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def offset_minutes(offset: str) -> int:
    """'-04:00' -> -240."""
    sign = -1 if offset.startswith("-") else 1
    return sign * to_minutes(offset.lstrip("+-"))


class SchedulingEnvironment:
    """The week a task was generated with, and that task's hidden truth."""

    def __init__(self, task: dict[str, Any]) -> None:
        world = task["world"]
        self.task = task
        self.truth: dict[str, Any] = task["truth"]
        self.user: str = world["user"]
        self.week: dict[str, str] = world["week"]
        self.people: dict[str, str] = world["people"]
        self.profiles: dict[str, dict[str, Any]] = world["profiles"]
        self.open, self.close = (to_minutes(h) for h in world["hours"])
        self.events: list[dict[str, Any]] = copy.deepcopy(world["events"])
        self.booked: list[str] = []
        self.next_id = 1

    # ---------------------------------------------------------------- lookups

    def name(self, person: str) -> str:
        return self.people[person]

    def resolve(self, who: str) -> str:
        """A person id from an id, a full name, or an unambiguous first name."""
        text = str(who).strip().lower()
        if text in self.people:
            return text
        for pid, name in self.people.items():
            if name.lower() == text:
                return pid
        first = [pid for pid, name in self.people.items() if name.split()[0].lower() == text]
        if len(first) == 1:
            return first[0]
        raise ValueError(f"no one named {who!r}; use a full name")

    def day(self, day: str) -> str:
        text = str(day).strip().lower()
        for key, date in self.week.items():
            if text in (key, DAY_NAMES[key].lower(), date):
                return key
        raise ValueError(f"{day!r} is not a day this week (mon to fri)")

    def shift(self, person: str) -> int:
        """Minutes to add to the user's clock to get this person's clock."""
        return offset_minutes(self.profiles[person]["utc_offset"]) - offset_minutes(self.profiles[self.user]["utc_offset"])

    def buffer(self, person: str) -> int:
        """Minutes an external guest asks for around meetings; a profile may change it."""
        return int(self.profiles[person].get("buffer", BUFFER))

    def external(self, person: str) -> bool:
        return bool(self.profiles[person].get("external"))

    def event(self, event_id: str) -> dict[str, Any] | None:
        return next((e for e in self.events if e["id"] == str(event_id).strip()), None)

    def events_for(self, person: str, day: str) -> list[dict[str, Any]]:
        return sorted((e for e in self.events if e["day"] == day and person in e["attendees"]),
                      key=lambda e: to_minutes(e["start"]))

    def clash(self, person: str, day: str, start: int, end: int, *, ignore: str | None = None,
              buffer: int = 0) -> dict | None:
        """The first event that blocks [start, end) for a person, on the user's clock.

        Events shown as free never block. `buffer` widens the meeting on
        both sides, for a guest who asks for time around it.
        """
        for e in self.events_for(person, day):
            s = to_minutes(e["start"])
            if e["id"] != ignore and e.get("show_as", "busy") != "free" \
                    and start - buffer < s + e["minutes"] and s < end + buffer:
                return e
        return None

    def _show(self, e: dict[str, Any], viewer: str) -> dict[str, Any]:
        s = to_minutes(e["start"]) + self.shift(viewer)
        shown = {"title": e["title"], "start": to_hhmm(s), "end": to_hhmm(s + e["minutes"]),
                 "show_as": e.get("show_as", "busy")}
        if viewer == self.user:
            shown = {"id": e["id"], **shown, "attendees": [self.name(p) for p in e["attendees"]]}
        return shown

    # ---------------------------------------------------------------- tools

    def read_calendar(self, person: str = "", day: str = "") -> dict[str, Any]:
        try:
            pid = self.resolve(person) if person else self.user
            days = [self.day(day)] if day else list(self.week)
        except ValueError as exc:
            return {"error": str(exc)}
        profile = self.profiles[pid]
        page: dict[str, Any] = {"person": self.name(pid), "utc_offset": profile["utc_offset"],
                                "working_hours": "-".join(profile["hours"])}
        if profile.get("note"):
            page["note"] = profile["note"]
        page["days"] = [{"day": d, "date": self.week[d], "events": [self._show(e, pid) for e in self.events_for(pid, d)]}
                        for d in days]
        return page

    def book_time(self, title: str, day: str, start: str, minutes: int,
                  attendees: list[str] | None = None) -> dict[str, Any]:
        try:
            d, s = self.day(day), to_minutes(start)
            invited = [self.resolve(a) for a in (attendees or [])]
        except ValueError as exc:
            return {"error": str(exc)}
        if not 15 <= int(minutes) <= 240:
            return {"error": "events last 15 to 240 minutes"}
        end = s + int(minutes)
        if s < self.open or end > self.close:
            return {"error": f"working hours are {to_hhmm(self.open)} to {to_hhmm(self.close)}"}
        conflict = self.clash(self.user, d, s, end)
        if conflict is not None:
            return {"error": f"{self.name(self.user)} is busy then: {conflict['title']} at {conflict['start']}"}
        event = {"id": f"N{self.next_id}", "owner": self.user, "title": title or "Meeting", "day": d,
                 "start": to_hhmm(s), "minutes": int(minutes), "show_as": "busy",
                 "attendees": list(dict.fromkeys([self.user, *invited]))}
        self.next_id += 1
        self.events.append(event)
        self.booked.append(event["id"])
        return {"booked": self._show(event, self.user)}

    def cancel_event(self, event_id: str) -> dict[str, Any]:
        """Cancel an event this request booked; earlier events are not the agent's to remove."""
        event = self.event(event_id)
        if event is None or event["id"] not in self.booked:
            return {"error": f"{event_id!r} is not an event you booked for this request"}
        self.events.remove(event)
        self.booked.remove(event["id"])
        return {"cancelled": event["id"]}

    # ---------------------------------------------------------------- the checker

    def acceptable(self, day: str, start: int, attendees: list[str], minutes: int,
                   ignore: str | None = None) -> str | None:
        """Why a meeting at this start would be refused, or None when everyone can make it."""
        for person in [self.user, *attendees]:
            first, last = (to_minutes(h) for h in self.profiles[person]["hours"])
            local = start + self.shift(person)
            if local < first or local + minutes > last:
                return (f"{DAY_NAMES[day]} {to_hhmm(start)} is {to_hhmm(local)} for {self.name(person)}, "
                        f"outside their working hours")
        for person in [self.user, *attendees]:
            if self.clash(person, day, start, start + minutes, ignore=ignore) is not None:
                return f"{self.name(person)} is busy at {DAY_NAMES[day]} {to_hhmm(start)}"
        for person in attendees:
            buffer = self.buffer(person)
            if self.external(person) and self.clash(person, day, start, start + minutes, ignore=ignore,
                                                    buffer=buffer) is not None:
                return (f"{self.name(person)} asked for {buffer} minutes between meetings, "
                        f"and {DAY_NAMES[day]} {to_hhmm(start)} leaves less")
        return None

    def check(self) -> dict[str, Any]:
        """Whether the user accepts what this request booked, and why not."""
        ok, reason = self._check()
        return {"success": ok, "reason": reason}

    def _check(self) -> tuple[bool, str]:
        truth = self.truth
        booked = [self.event(i) for i in self.booked]
        if len(booked) != truth["count"]:
            return False, f"it booked {len(booked)} event(s), not {truth['count']}"
        need = {self.resolve(p) for p in truth["attendees"]}
        per_day: Counter[str] = Counter()
        for e in booked:
            s = to_minutes(e["start"])
            when = f"{DAY_NAMES[e['day']]} {e['start']}"
            if e["minutes"] != truth["minutes"]:
                return False, f"{when} lasts {e['minutes']} minutes, not {truth['minutes']}"
            if e["day"] not in truth["days"]:
                return False, f"{DAY_NAMES[e['day']]} is not one of the days asked for"
            if truth.get("earliest") and s < to_minutes(truth["earliest"]):
                return False, f"{when} starts before {truth['earliest']}"
            if truth.get("latest_end") and s + e["minutes"] > to_minutes(truth["latest_end"]):
                return False, f"{when} ends after {truth['latest_end']}"
            if not need <= set(e["attendees"]):
                missing = ", ".join(sorted(self.name(p) for p in need - set(e["attendees"])))
                return False, f"{missing} is not invited to {when}"
            others = [p for p in e["attendees"] if p != self.user]
            refused = self.acceptable(e["day"], s, others, e["minutes"], ignore=e["id"])
            if refused:
                return False, refused
            per_day[e["day"]] += 1
        if truth.get("per_day_max") and max(per_day.values()) > truth["per_day_max"]:
            return False, "it put more than one on the same day"
        if truth.get("same_start") and len({e["start"] for e in booked}) > 1:
            return False, "the meetings are not at the same time on every day"
        return True, "accepted"
