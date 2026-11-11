"""The calendar desk's stream for the production-scale test: 4,000 requests over 20 days, and what changes in it.

Every request has a week of its own, so no reply recorded for one request can answer another. The weeks come from
Chapter 6's generators, the words from seven groups of writers who saw only a description of the assistant's job
(banks_calendar.json), and the verdict from Chapter 6's checker.

Five things change, each of a kind a production agent meets.

    day 5    Dana rewords her note, and the checker enforces 30 minutes
    day 8    two more groups of writers start sending requests
    day 11   the calendar cannot be read for one request in ten
    day 14   a second kind of work arrives: the same meeting at the same time on several days
    day 17   Dana changes her note again, to three quarters of an hour

Each day also sets requests aside that are never served: that night's gate, what a search may measure on, and the
weeks that judge a relearned function. The seventh group of writers words a final exam that nothing sees before the
last night.
"""

from __future__ import annotations

import json
import random
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from string import Formatter
from typing import Any

from chapters.ch06.assistant import generator, generator_drift, generator_same_time
from chapters.ch06.assistant.environment import DAY_NAMES, DAYS, SchedulingEnvironment
from chapters.ch06.assistant.generator import CONVENTIONS, TOPICS, _request, _starts, _world, face_value_starts
from chapters.ch06.assistant.learning import calendars_read, class_of, run_on_case
from helix.metacognition import Gate

from examples.scale.scripted import TASKS, one_in

HERE = Path(__file__).parent
BANKS = HERE / "banks_calendar.json"
CACHE = HERE / "runs" / "desk_calendar_weeks.json"
SHORT = {"mon": "Mon", "tue": "Tue", "wed": "Wed", "thu": "Thu", "fri": "Fri"}
GROUPS = ("g1", "g2", "g3", "g4", "g5", "g6")  # the writers whose requests are served; "exam" words the final exam
LATE_GROUPS = ("g5", "g6")  # they start sending requests on day 8
COUNTS = {"single": 12, "double": 6, "same_time": 6, "near_miss": 10, "out_of_scope": 6}
BOOKING = {"one": "single", "two": "double", "same": "same_time"}
KIND_NAMES = {"one": "one meeting", "two": "two meetings", "same": "same time"}


@dataclass(frozen=True)
class Note:
    """Dana's note in force: its words, the minutes the checker enforces, and what a function written for the note
    before it would apply. In a week marked sensitive, each of those readings books a start the checker rejects."""

    key: str
    text: str
    buffer: int
    stale: tuple[int, ...] = ()


FIFTEEN = Note("fifteen", generator.NOTE, 15)
HALF_HOUR = Note("half_hour", generator_drift.NOTE, 30, stale=(0,))
THREE_QUARTERS = Note("three_quarters", "External client. Please leave three quarters of an hour between any meeting "
                      "with me and my other meetings.", 45, stale=(0, 30))
NOTES = {note.key: note for note in (FIFTEEN, HALF_HOUR, THREE_QUARTERS)}


@dataclass(frozen=True)
class Day:
    number: int
    note: Note
    groups: tuple[str, ...]
    second_class: bool = False  # requests for the same meeting at the same time on several days arrive
    outage: bool = False  # the calendar cannot be read for one request in ten
    change: str = ""  # what is different from the day before, in a few words


def plan(days: int = 20) -> list[Day]:
    """The stream's calendar. A shorter stream keeps the same five changes at the same share of its length."""
    def at(day: int) -> int:
        return max(2, round(day * days / 20))

    reword, writers, outage, second, again = at(5), at(8), at(11), at(14), at(17)
    changes = {reword: "Dana's note asks for half an hour", writers: "two more groups of writers",
               outage: "the calendar is down for one request in ten",
               second: "requests for the same time on several days arrive",
               again: "Dana's note asks for three quarters of an hour"}
    out = []
    for number in range(1, days + 1):
        note = FIFTEEN if number < reword else HALF_HOUR if number < again else THREE_QUARTERS
        groups = GROUPS if number >= writers else tuple(g for g in GROUPS if g not in LATE_GROUPS)
        out.append(Day(number, note, groups, second_class=number >= second, outage=number == outage,
                       change=changes.get(number, "")))
    return out


def mix(count: int, second_class: bool) -> list[str]:
    """How many of each kind a day holds: half ask for one meeting, 12% for two, 30% are near misses and 8% are out
    of scope. Once the second kind of work has arrived it takes 15%, and the rest keep their proportions."""
    shares = {"one": 0.50, "two": 0.12, "near": 0.30, "out": 0.08}
    if second_class:
        shares = {**{kind: share * 0.85 for kind, share in shares.items()}, "same": 0.15}
    exact = {kind: share * count for kind, share in shares.items()}
    whole = {kind: int(value) for kind, value in exact.items()}
    for kind in sorted(exact, key=lambda k: exact[k] - whole[k], reverse=True)[:count - sum(whole.values())]:
        whole[kind] += 1
    return [kind for kind, n in whole.items() for _ in range(n)]


# ---------------------------------------------------------------- weeks


def _with_buffer(task: dict[str, Any], buffer: int) -> dict[str, Any]:
    world = {**task["world"], "profiles": {**task["world"]["profiles"],
                                           "dana": {**task["world"]["profiles"]["dana"], "buffer": buffer}}}
    return {**task, "world": world}


def _answers(task: dict[str, Any]) -> Any:
    """The starts the checker would accept: by day for the first class, common to every day for the second."""
    found = _starts(SchedulingEnvironment(task), task["truth"])
    return generator_same_time.common(found) if task["class"] == "same_time" else found


def _first_answer_is_right(task: dict[str, Any], guessed: Any, right: Any) -> bool:
    """Whether booking the first start of `guessed`, as a function's caller does, would be accepted."""
    if task["class"] == "same_time":
        return bool(guessed) and guessed[0] in right
    first = next(((day, guessed[day][0]) for day in task["truth"]["days"] if guessed[day]), None)
    return first is not None and first[1] in right[first[0]]


def draw(seed: int, kind: str, note_key: str, sensitive: bool, max_tries: int = 400_000) -> dict[str, Any]:
    """One week and what is asked of it. The same arguments always give the same week.

    A week is kept when it has a solution and few valid starts. An ordinary week must also trap a reading at face
    value, with the conventions taking turns to decide it, as in Chapter 6. A sensitive week invites Dana and is
    booked wrongly by each stale reading of her note. For the same time on several days, which few weeks allow at
    all, it is the reading that finds no minutes in the note that must book wrongly.
    """
    note = NOTES[note_key]
    rng = random.Random(seed)
    same = kind == "same"
    for _ in range(max_tries):
        world = _world(rng)
        if note is not FIFTEEN:
            world["profiles"]["dana"].update(note=note.text, buffer=note.buffer)
        if same:
            text, truth = generator_same_time._request(rng, CONVENTIONS[seed % len(CONVENTIONS)])
        else:
            text, truth = _request(rng, 2 if kind == "two" else 1)
        task = {"id": f"{'T' if same else 'S'}{seed}", "seed": seed, "class": "same_time" if same else "multi_attendee",
                "text": text, "world": world, "truth": truth}
        right = _answers(task)
        if same and not 1 <= len(right) <= 4:
            continue
        if not same and (sum(1 for starts in right.values() if starts) < truth["count"]
                         or sum(len(starts) for starts in right.values()) > 8):
            continue
        if sensitive:
            if "Dana Park" not in truth["attendees"]:
                continue
            stale_readings = note.stale[:1] if same else note.stale
            if any(_first_answer_is_right(task, _answers(_with_buffer(task, stale)), right)
                   for stale in stale_readings):
                continue
        else:
            naive = face_value_starts(task)
            if _first_answer_is_right(task, generator_same_time.common(naive) if same else naive, right):
                continue
            decides = generator_same_time.deciding_conventions if same else generator.deciding_conventions
            task["decided_by"] = decides(task)
            if CONVENTIONS[seed % len(CONVENTIONS)] not in task["decided_by"]:
                continue
        task.update(base=task["id"], kind=KIND_NAMES[kind], topic=next(t for t in TOPICS if t in text),
                    note=note.key, sensitive=sensitive)
        return task
    if sensitive:  # so rare for this seed that none was found: an ordinary week, which says that it is one
        return draw(seed, kind, note_key, False, max_tries)
    raise RuntimeError(f"seed {seed}: no {kind} week under the {note.key} note met the constraints")


def _drawn(args: tuple[int, str, str, bool]) -> tuple[str, dict[str, Any]]:
    return "{}:{}:{}:{}".format(*args[:3], int(args[3])), draw(*args)


class Weeks:
    """Weeks drawn once and kept on disk, since each takes a moment to find."""

    def __init__(self, path: Path | None = CACHE) -> None:
        self.path = path
        self.pool: dict[str, dict[str, Any]] = {}
        if path is not None and path.exists():
            self.pool = json.loads(path.read_text(encoding="utf-8"))
        self.fresh = 0

    def get(self, seed: int, kind: str, note: Note, sensitive: bool) -> dict[str, Any]:
        key = f"{seed}:{kind}:{note.key}:{int(sensitive)}"
        if key not in self.pool:
            self.pool[key] = draw(seed, kind, note.key, sensitive)
            self.fresh += 1
        return self.pool[key]

    def fill(self, wanted: Iterable[tuple[int, str, Note, bool]], workers: int = 0) -> int:
        """Draw every missing week, on several processors when there are many. Returns how many were drawn."""
        missing = list(dict.fromkeys((seed, kind, note.key, sensitive) for seed, kind, note, sensitive in wanted
                                     if f"{seed}:{kind}:{note.key}:{int(sensitive)}" not in self.pool))
        if len(missing) < 64 or workers == 1:
            self.pool.update(map(_drawn, missing))
            self.fresh += len(missing)
        else:
            with ProcessPoolExecutor(max_workers=workers or None) as pool:
                for done, (key, week) in enumerate(pool.map(_drawn, missing, chunksize=4), 1):
                    self.pool[key] = week
                    self.fresh += 1
                    if done % 1000 == 0:  # a long draw that is stopped keeps what it has found
                        self.save()
        return len(missing)

    def save(self) -> None:
        if self.fresh and self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.pool), encoding="utf-8")
            self.fresh = 0


# ---------------------------------------------------------------- words


def load_banks(path: Path = BANKS) -> dict[str, Any]:
    banks = json.loads(path.read_text(encoding="utf-8"))
    faults = check_banks(banks)
    if faults:
        raise ValueError("banks_calendar.json: " + "; ".join(faults[:5]))
    return banks


def check_banks(banks: dict[str, Any]) -> list[str]:
    """What is wrong with the writers' templates, if anything: a missing group, a wrong count, a placeholder nobody
    fills, or a booking request that leaves out something a booking needs."""
    known = {"minutes", "topic", "who", "first_names", "days", "short_days", "each_day", "short_each_day", "limit",
             "other_day", "first", "day"}
    needs = {"single": [{"minutes"}, {"topic"}, {"who", "first_names"}, {"days", "short_days"}, {"limit"}],
             "double": [{"minutes"}, {"topic"}, {"who", "first_names"}, {"days", "short_days"}, {"limit"}],
             "same_time": [{"minutes"}, {"topic"}, {"who", "first_names"}, {"each_day", "short_each_day"}, {"limit"}]}
    faults = []
    for group in (*GROUPS, "exam"):
        bank = banks.get(group)
        if not isinstance(bank, dict):
            faults.append(f"{group}: missing")
            continue
        for key, count in COUNTS.items():
            templates = bank.get(key) or []
            if len(templates) != count or len(set(templates)) != count:
                faults.append(f"{group}.{key}: {len(templates)} templates, {len(set(templates))} distinct, not {count}")
            for n, template in enumerate(templates):
                try:
                    used = {name for _, name, _, _ in Formatter().parse(template) if name is not None}
                except ValueError as exc:
                    faults.append(f"{group}.{key}[{n}]: {exc}")
                    continue
                if used - known:
                    faults.append(f"{group}.{key}[{n}]: unknown placeholder {sorted(used - known)}")
                for either in needs.get(key, []):
                    if not used & either:
                        faults.append(f"{group}.{key}[{n}]: needs one of {sorted(either)}")
        for limit in ("earliest", "latest_end"):
            phrases = (bank.get("limit") or {}).get(limit) or []
            if len(phrases) != 3 or any("{time}" not in phrase for phrase in phrases):
                faults.append(f"{group}.limit.{limit}: three phrases, each with {{time}}, are needed")
    return faults


def _joined(names: list[str]) -> str:
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def _span(days: list[str], names: dict[str, str]) -> str:
    return f"{names[days[0]]} to {names[days[-1]]}" if len(days) > 2 else f"{names[days[0]]} or {names[days[1]]}"


def fields(task: dict[str, Any], bank: dict[str, Any]) -> dict[str, str]:
    """What a template's placeholders are filled with, read from the week's own truth."""
    truth = task["truth"]
    limit = "earliest" if truth.get("earliest") else "latest_end"
    phrases = bank["limit"][limit]
    other = next(day for day in (*DAYS[::-1], *DAYS) if day not in truth["days"])
    return {"minutes": str(truth["minutes"]), "topic": task["topic"], "who": _joined(truth["attendees"]),
            "first_names": _joined([name.split()[0] for name in truth["attendees"]]),
            "days": _span(truth["days"], DAY_NAMES), "short_days": _span(truth["days"], SHORT),
            "each_day": _joined([DAY_NAMES[day] for day in truth["days"]]),
            "short_each_day": _joined([SHORT[day] for day in truth["days"]]),
            "limit": phrases[task["seed"] % len(phrases)].format(time=truth[limit]),
            "other_day": DAY_NAMES[other], "first": truth["attendees"][0].split()[0],
            "day": DAY_NAMES[truth["days"][0]]}


# ---------------------------------------------------------------- the stream


class Stream:
    """The requests of every day, what is held back from each, and the exam. `per_day`, `days` and `held` shrink
    it for a test; the paid run uses 200 requests a day for 20 days, and holds back two sets of 16 each day."""

    def __init__(self, per_day: int = 200, days: int = 20, *, weeks: Weeks | None = None,
                 banks: dict[str, Any] | None = None, held: int = 16) -> None:
        self.per_day, self.days, self.held = per_day, plan(days), held
        self.weeks = weeks or Weeks()
        self.banks = banks or load_banks()
        self.bases: dict[str, dict[str, Any]] = {}  # every week by its id, for a function's practice
        self._texts: set[str] = set()
        self._built: dict[tuple[str, int], list[dict[str, Any]]] = {}

    # ---- which weeks a set needs

    def _specs(self, kinds: list[str], day: Day, first_seed: int, groups: tuple[str, ...]
               ) -> list[tuple[int, str, Note, bool, str, str]]:
        """For each request of a set: its seed, the kind of week under it, the note, whether the week is sensitive,
        the kind of request, and the group that words it."""
        out, acting = [], 0
        for n, kind in enumerate(kinds):
            booking = kind in BOOKING
            # Once a note has changed, every other request for a booking sits on a week the change decides.
            sensitive = booking and bool(day.note.stale) and acting % 2 == 0
            acting += booking
            out.append((first_seed + n, kind if booking else "one", day.note, sensitive, kind,
                        groups[n % len(groups)]))
        return out

    def _served_specs(self, day: Day) -> list[tuple[int, str, Note, bool, str, str]]:
        kinds = mix(self.per_day, day.second_class)
        random.Random(1000 + day.number).shuffle(kinds)
        return self._specs(kinds, day, 100_000 + day.number * 1000, day.groups)

    def _held_specs(self, day: Day, role: str) -> list[tuple[int, str, Note, bool, str, str]]:
        # Three in four ask for a booking, in about the day's own proportions, and the rest must be left alone.
        acting, rest = self.held * 3 // 4, self.held - self.held * 3 // 4
        same = acting // 4 if day.second_class else 0
        two = max(1, acting // 5) if day.second_class else acting // 4
        kinds = ["one"] * (acting - two - same) + ["two"] * two + ["same"] * same
        kinds += ["near"] * (rest - 1) + ["out"]
        return self._specs(kinds, day, (300_000 if role == "gate" else 400_000) + day.number * 100, day.groups)

    def _function_specs(self, note: Note, second_class: bool) -> list[tuple[int, str, Note, bool, str, str]]:
        first = 500_000 + 1000 * list(NOTES).index(note.key)
        specs = [(first + n, "one", note, bool(note.stale) and n % 2 == 0, "one", "") for n in range(12)]
        if second_class:
            specs += [(first + 100 + n, "same", note, bool(note.stale) and n % 2 == 0, "same", "") for n in range(6)]
        return specs

    def _exam_specs(self) -> list[tuple[int, str, Note, bool, str, str]]:
        last = self.days[-1]
        kinds = mix(self.per_day, last.second_class)
        random.Random(7).shuffle(kinds)
        return self._specs(kinds, last, 600_000, ("exam",))

    def prepare(self, workers: int = 0) -> int:
        """Draw every week the stream needs that the cache lacks. Returns how many were drawn."""
        wanted = [spec[:4] for day in self.days
                  for spec in (*self._served_specs(day), *self._held_specs(day, "gate"),
                               *self._held_specs(day, "search"), *self._function_specs(day.note, day.second_class))]
        wanted += [spec[:4] for spec in self._exam_specs()]
        drawn = self.weeks.fill(wanted, workers)
        self.weeks.save()
        return drawn

    # ---- requests

    def _request(self, spec: tuple[int, str, Note, bool, str, str], reference: str, turn: int,
                 outage: bool = False) -> dict[str, Any]:
        seed, week_kind, note, sensitive, kind, group = spec
        base = self.weeks.get(seed, week_kind, note, sensitive)
        self.bases[base["id"]] = base
        bank = self.banks[group]
        key = BOOKING.get(kind) or ("near_miss" if kind == "near" else "out_of_scope")
        templates, filled = bank[key], fields(base, bank)
        if key == "out_of_scope":
            # A reminder is something a calendar assistant may reasonably put on the calendar. The brief gave one as
            # an example of work outside the job, so each group wrote one, and those seven are left out.
            templates = [template for template in templates if "remind" not in template.lower()]
        for step in range(len(templates)):  # the next template if these words have been used already
            text = templates[(turn + step) % len(templates)].format(**filled)
            if text not in self._texts:
                break
        self._texts.add(text)
        booking = kind in BOOKING
        request = {**base, "id": f"{base['id']}~{reference}", "text": text, "group": group, "outage": outage,
                   "style": "plain" if booking else "near_miss" if kind == "near" else "out_of_scope",
                   "expect": "act" if booking else "handoff", "asked": kind}
        TASKS[text] = request  # the scripted models of a soak find a request's truth by its words
        return request

    def _set(self, name: str, number: int, specs: list[tuple[int, str, Note, bool, str, str]],
             outage: bool = False) -> list[dict[str, Any]]:
        return [self._request(spec, f"{name}{number}-{n}", number * 3 + n // max(1, len({s[5] for s in specs})),
                              outage=outage and one_in(str(spec[0]), 0.1, "outage"))
                for n, spec in enumerate(specs)]

    def _word(self) -> None:
        """Word every request once, in one fixed order: each day's served requests, its gate, its search set, and
        last the exam. A request takes the next template when its words are taken, so the order must not depend on
        which assistant asks for which set first."""
        if self._built:
            return
        for day in self.days:
            self._built["d", day.number] = self._set("d", day.number, self._served_specs(day), outage=day.outage)
            for n, request in enumerate(self._built["d", day.number]):
                request.update(day=day.number, seq=(day.number - 1) * self.per_day + n)
            for role in ("gate", "search"):
                self._built[role, day.number] = self._set(role, day.number, self._held_specs(day, role))
        self._built["exam", 0] = self._set("exam", 0, self._exam_specs())
        for n, request in enumerate(self._built["exam", 0]):
            request.update(day=len(self.days) + 1, seq=len(self.days) * self.per_day + n)

    def served(self, day: Day) -> list[dict[str, Any]]:
        """The day's requests in the order they arrive. Each knows its place in the whole stream."""
        self._word()
        return self._built["d", day.number]

    def held_back(self, day: Day) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
        """Requests under the day's own conditions that are never served: that night's gate, and a search's own."""
        self._word()
        return ({r["id"]: r for r in self._built["gate", day.number]},
                {r["id"]: r for r in self._built["search", day.number]})

    def function_gate(self, day: Day) -> Gate:
        """The weeks that judge a relearned function, for each function the assistant has that day."""
        cases: dict[str, list[str]] = {"find_shared_starts": [], "find_common_starts": []}
        for seed, kind, note, sensitive, _, _ in self._function_specs(day.note, day.second_class):
            base = self.weeks.get(seed, kind, note, sensitive)
            self.bases[base["id"]] = base
            cases["find_common_starts" if kind == "same" else "find_shared_starts"].append(base["id"])
        return Gate(cases={name: ids for name, ids in cases.items() if ids})

    def exam(self) -> list[dict[str, Any]]:
        """The final exam: the last day's conditions, in the words of a group nothing has seen."""
        self._word()
        return self._built["exam", 0]

    async def practice(self, source: str, case_id: str) -> tuple[float, str, str]:
        """Try a function on the week behind a request, as Chapter 6's practice does on its own weeks."""
        task = self.bases[case_id.split("~")[0]]
        env = SchedulingEnvironment(task)
        result = await run_on_case(source, task)
        if isinstance(result, dict) and "error" in result:
            return 0.0, f"the function crashed when run: {result['error']}", calendars_read(task)
        wrong_shape = class_of(task).book(env, task, result.get("starts") if isinstance(result, dict) else None)
        if wrong_shape:
            return 0.0, f"the function returned {result!r}, not {{'starts': {wrong_shape}}}", calendars_read(task)
        verdict = env.check()
        if verdict["success"]:
            return 1.0, "accepted", calendars_read(task)
        return 0.0, f"booking its earliest start was rejected by the user: {verdict['reason']}", calendars_read(task)
