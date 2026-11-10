"""D5, an on-call desk: changing one shift is routine, but planning a week takes reasoning, and every plan differs.

A request to put someone on a day needs only the name and the day. A request to plan the week brings its own
constraints in prose, and the plan has to meet all of them at once, with the team's standing rules.

The forms those constraints take differ by split. Development and search requests use one set. The gate's add a second
and the held-out requests a third, so a program that only knows the forms it has seen can't pass. A person reading the
request can: that is the point of the domain, because the reasoning is different every time.
"""

from __future__ import annotations

import itertools
import random
from functools import cache
from typing import Any

PEOPLE = ("Priya", "Omar", "Lena", "Sam", "Noor", "Tomas")
DAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday")
SKILLS = {"Priya": ["databases", "networking"], "Omar": ["networking"], "Lena": ["databases", "payments"],
          "Sam": ["payments"], "Noor": ["databases"], "Tomas": ["networking", "payments"]}
RULES = ["Nobody takes more than two shifts in a week.", "Nobody takes a shift on a day they are on leave."]
MOST_SHIFTS = 2
STYLES = ("casual", "shorthand", "distractor", "buried")
PLAN_STYLES = ("plan_list", "plan_note", "plan_chat", "plan_buried")
FEW_SOLUTIONS = 3  # a plan with more ways to be right than this is too easy to need reasoning

# Which forms a plan's constraints may take, by who gets to see them.
SEEN = ("off", "fixed", "no_row", "earlier", "at_most_one")
GATE_ONLY = ("same", "not_after", "at_least_one")
HELD_OUT_ONLY = ("exactly_two", "either", "skill", "not_both")
# The test split: the first four bases go to the search, the next four to the gate, and the rest are held out.
SEARCH_BASES = range(201, 205)
GATE_BASES = range(205, 209)

TEXT = {
    "off": "{a} can't do {d}.",
    "fixed": "{a} has to take {d}.",
    "no_row": "{a} shouldn't work two days in a row.",
    "earlier": "{a} and {b} both need a shift this week, and {a}'s first shift has to come before {b}'s.",
    "at_most_one": "Give {a} one shift at most.",
    "same": "Whoever covers {d} also covers {e}.",
    "not_after": "{b} can't be on call the day right after {a}.",
    "at_least_one": "{a} needs at least one shift.",
    "exactly_two": "{a} needs exactly two shifts.",
    "either": "{d} has to go to either {a} or {b}.",
    "skill": "{d} needs someone who knows {s}.",
    "not_both": "{a} and {b} can't both be on the rota this week.",
}

# One way each style of request is put, by who sees it: the words differ as well as the forms.
PLAN_WORDS = {
    "dev": {"plan_list": ("Please plan next week's on-call rota and publish it.\n- ", "\n- ", ""),
            "plan_note": ("Can you sort out next week's on-call and publish it? ", " ", ""),
            "plan_chat": ("hey, need next week's rota done and published. a few things to work around: ", " ", " thanks!"),
            "plan_buried": ("The roadmap review ran long, so I'm only getting to this now. Please plan and publish next "
                            "week's on-call. ", " ", " I'll tell the team once it's up.")},
    "search": {"plan_list": ("Next week's on-call needs planning and publishing. Requirements:\n- ", "\n- ", ""),
               "plan_note": ("Please work out next week's rota and publish it. ", " ", ""),
               "plan_chat": ("could you get next week's on-call planned and out? notes: ", " ", " cheers"),
               "plan_buried": ("We've had a busy sprint and a couple of people are juggling things. Could you plan next "
                               "week's on-call and publish it? ", " ", " That should keep everyone happy.")},
    "gate": {"plan_list": ("Time to set next week's on-call. Publish a rota that respects:\n- ", "\n- ", ""),
             "plan_note": ("I need next week's on-call rota published. ", " ", ""),
             "plan_chat": ("rota time again, can you publish next week's? ", " ", " ty"),
             "plan_buried": ("Finance moved the quarter-end, which shuffled a few calendars. Please put together next "
                             "week's on-call and publish it. ", " ", " Let me know when it's live.")},
    "held": {"plan_list": ("Build next week's on-call schedule and publish it. It has to satisfy:\n- ", "\n- ", ""),
             "plan_note": ("Please publish a rota for next week. ", " ", ""),
             "plan_chat": ("morning! next week's on-call still isn't done, can you publish one? ", " ", " appreciated"),
             "plan_buried": ("The office move is next week, so people's availability is a bit unusual. Would you plan the "
                             "on-call and publish it? ", " ", " Thanks for picking this up.")},
}
ASSIGN_WORDS = {
    "dev": {"casual": ["hey can you put {person} on {day}?", "please have {person} take {day}"],
            "shorthand": ["{person} on {short}", "{short}: {person}"],
            "distractor": ["{other} is fine where they are, but please put {person} on {day}.",
                           "Leave {other} alone. Put {person} on {day}."],
            "buried": ["We're moving a few things around after the offsite. {person} said they'd cover {day}, so please "
                       "put them on it. That should settle next week.",
                       "I talked to the team this morning. Please give {day}'s shift to {person}. Thanks for sorting it."]},
    "search": {"casual": ["could you stick {person} on {day} for me", "{person} is taking {day}, please update it"],
               "shorthand": ["{short} -> {person}", "{person} for {short}"],
               "distractor": ["{other} doesn't need changing. {day} should go to {person}.",
                              "Put {person} on {day}, and nothing for {other}."],
               "buried": ["The release moved again, so we're reshuffling. Put {person} on {day}, please. I'll update "
                          "the channel.", "Busy week ahead. {person} has offered to take {day}, so set that up. Thanks."]},
    "gate": {"casual": ["pop {person} onto {day} when you can", "mind moving {day} to {person}?"],
             "shorthand": ["{person} covers {short}", "{short} = {person}"],
             "distractor": ["Ignore {other} for now and assign {day} to {person}.",
                            "{person} gets {day}. {other} stays as they are."],
             "buried": ["Quick one before standup. {day} needs to go to {person}, they've agreed. Thanks a lot.",
                        "After yesterday's retro we agreed on a change. Assign {person} to {day}, please. That's all."]},
    "held": {"casual": ["can we have {person} on {day}", "go ahead and give {person} {day}"],
             "shorthand": ["{short} {person}", "{person}, {short}"],
             "distractor": ["{other} is sorted. All I need is {person} on {day}.",
                           "Not {other}, this is about {person}: put them on {day}."],
             "buried": ["Things have been hectic with the launch. {person} volunteered, so please make them {day}'s "
                        "on-call. I appreciate it.", "Small update from the team sync. Set {person} on {day}. Nothing else "
                        "is changing."]},
}
OUT_OF_SCOPE = {"dev": ["Who's the manager for {person}?", "Book {person} a meeting room for {day}.",
                        "What's the pager number for the on-call person?", "Send {person} a reminder about {day}."],
                "test": ["When does {person}'s contract end?", "Order lunch for whoever is on call {day}.",
                         "Reset {person}'s pager password.", "Add {person} to the incident channel."]}
NEAR_MISS_DEV = ["Who's on call {day}?", "Hold off on putting {person} on {day} until I check with them.",
                 "Take {person} off {day}.", "What would next week look like if {person} took {day}? Don't change it.",
                 "Did you already move {person} to {day}?", "Draft a rota for next week, but don't publish anything."]
NEAR_MISS = ["Is {person} on call {day}?", "Don't put {person} on {day} yet, they haven't agreed.",
             "Can you tell me whether {day} has been filled?", "Undo the change that put {person} on {day}.",
             "Would {person} be allowed to take {day}?", "Who has {day} at the moment?",
             "How many shifts does {person} have this week?", "I already put {person} on {day} myself, nothing needed.",
             "Remind me on Friday to give {person} {day}.", "Show me a possible rota for next week; publish nothing."]
NEAR_MISS_GATE = ["Has anyone put {person} down for {day}?", "Please wait before changing {day}, I want {person}'s answer first.",
                  "Could {person} even take {day}?", "Reverse whatever you did to {day}.",
                  "Which person is covering {day}?", "I only want to know {person}'s shift count, change nothing.",
                  "{person} is sorting {day} out personally, leave it.", "Next month, not now, put {person} on a {day}.",
                  "Tell me if {day} is empty.", "Sketch next week's rota for me to look at, without publishing."]
NEAR_MISS_FRESH = ["Am I right that {person} has {day}?", "Keep {day} as it is until {person} confirms.",
                   "Remove {person} from {day}.", "Is there anyone free on {day}?",
                   "Was {person} ever on {day} last week?", "Is it OK for {person} to do {day}?",
                   "No need to touch {day}, {person} already has it.", "After the holidays, please move {person} to {day}.",
                   "What's {person}'s load next week?", "Give me a draft of next week's on-call; I'll publish it myself."]


def _key(person: str) -> str:
    return person.strip().capitalize()


def _count(rota: dict[str, str], person: str) -> int:
    return sum(p == person for p in rota.values())


def _first(rota: dict[str, str], person: str) -> int | None:
    return next((n for n, day in enumerate(DAYS) if rota[day] == person), None)


def _after(rota: dict[str, str]) -> set[tuple[str, str]]:
    return {(rota[x], rota[y]) for x, y in itertools.pairwise(DAYS)}


def holds(constraint: dict[str, Any], rota: dict[str, str]) -> bool:
    """Whether a rota meets one constraint."""
    form, a, b, d, e = (constraint["form"], constraint.get("a"), constraint.get("b"), constraint.get("d"),
                        constraint.get("e"))
    if form == "off":
        return rota[d] != a
    if form == "fixed":
        return rota[d] == a
    if form == "no_row":
        return (a, a) not in _after(rota)
    if form == "earlier":
        first, second = _first(rota, a), _first(rota, b)
        return first is not None and second is not None and first < second
    if form == "at_most_one":
        return _count(rota, a) <= 1
    if form == "same":
        return rota[d] == rota[e]
    if form == "not_after":
        return (a, b) not in _after(rota)
    if form == "at_least_one":
        return _count(rota, a) >= 1
    if form == "exactly_two":
        return _count(rota, a) == 2
    if form == "either":
        return rota[d] in (a, b)
    if form == "skill":
        return constraint["s"] in SKILLS[rota[d]]
    if form == "not_both":
        return not (_count(rota, a) and _count(rota, b))
    raise ValueError(f"no constraint form {form!r}")


def follows_rules(rota: dict[str, str], leave: dict[str, list[str]]) -> bool:
    """The team's standing rules, which every plan has to meet."""
    return (all(_count(rota, person) <= MOST_SHIFTS for person in rota.values())
            and not any(day in leave.get(person, []) for day, person in rota.items()))


def solutions(constraints: list[dict[str, Any]], leave: dict[str, list[str]]) -> list[dict[str, str]]:
    """Every rota that meets the standing rules and all the constraints."""
    found = []
    for people in itertools.product(PEOPLE, repeat=len(DAYS)):
        rota = dict(zip(DAYS, people, strict=True))
        if follows_rules(rota, leave) and all(holds(c, rota) for c in constraints):
            found.append(rota)
    return found


# Every way to fill the week within the most shifts one person may take, so a search for answers starts from these.
_ROTAS = [rota for rota in (dict(zip(DAYS, people, strict=True)) for people in itertools.product(PEOPLE, repeat=len(DAYS)))
          if all(_count(rota, person) <= MOST_SHIFTS for person in rota.values())]


def _draw(form: str, rng: random.Random, rota: dict[str, str]) -> dict[str, Any] | None:
    """A constraint of `form` that `rota` meets, or None when it can't have one."""
    on_it = sorted(set(rota.values()))
    day = rng.choice(DAYS)
    person, other = rng.sample(PEOPLE, 2)
    if form == "off":
        return {"form": form, "a": person, "d": day} if rota[day] != person else None
    if form == "fixed":
        return {"form": form, "a": rota[day], "d": day}
    if form == "no_row":
        return {"form": form, "a": person} if person in on_it and (person, person) not in _after(rota) else None
    if form == "earlier":
        if len(on_it) < 2:
            return None
        a, b = sorted(rng.sample(on_it, 2), key=lambda p: _first(rota, p))
        return {"form": form, "a": a, "b": b}
    if form == "at_most_one":
        return {"form": form, "a": person} if _count(rota, person) <= 1 else None
    if form == "same":
        pairs = [(x, y) for x, y in itertools.combinations(DAYS, 2) if rota[x] == rota[y]]
        return {"form": form, "d": pairs[0][0], "e": pairs[0][1]} if pairs else None
    if form == "not_after":
        return {"form": form, "a": person, "b": other} if (person, other) not in _after(rota) else None
    if form == "at_least_one":
        return {"form": form, "a": rng.choice(on_it)}
    if form == "exactly_two":
        twice = [p for p in on_it if _count(rota, p) == 2]
        return {"form": form, "a": rng.choice(twice)} if twice else None
    if form == "either":
        return {"form": form, "d": day, "a": rota[day], "b": next(p for p in rng.sample(PEOPLE, 6) if p != rota[day])}
    if form == "skill":
        return {"form": form, "d": day, "s": rng.choice(SKILLS[rota[day]])}
    if form == "not_both":
        off_it = [p for p in PEOPLE if p not in on_it]
        if not off_it:
            return None
        absent = rng.choice(off_it)
        return {"form": form, "a": absent, "b": next(p for p in rng.sample(PEOPLE, 6) if p != absent)}
    raise ValueError(f"no constraint form {form!r}")


def _rota(rng: random.Random) -> dict[str, str]:
    while True:
        rota = {day: rng.choice(PEOPLE) for day in DAYS}
        if all(_count(rota, person) <= MOST_SHIFTS for person in PEOPLE):
            return rota


def _forms(number: int) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """The forms a plan may draw on, and those it must include, from where its base request sits."""
    if number in GATE_BASES:
        return SEEN + GATE_ONLY, GATE_ONLY
    if number in SEARCH_BASES or number < 200:
        return SEEN, ()
    return SEEN + HELD_OUT_ONLY, HELD_OUT_ONLY


@cache
def _problem(number: int, style: str) -> dict[str, Any]:
    """A plan with a known answer and few others, its constraints drawn so that answer meets them."""
    rng = random.Random(f"plan-{number}-{style}")
    forms, required = _forms(number)
    while True:
        answer = _rota(rng)
        leave: dict[str, list[str]] = {}
        for person, day in rng.sample(list(itertools.product(PEOPLE, DAYS)), 3):
            if answer[day] != person:
                leave.setdefault(person, []).append(day)
        constraints: list[dict[str, Any]] = []
        alive = [rota for rota in _ROTAS if not any(day in leave.get(person, []) for day, person in rota.items())]
        pool = [*rng.sample(required, min(2, len(required)))] if required else []
        for _ in range(60):
            if len(constraints) >= 8:
                break
            form = pool.pop() if pool else rng.choice(forms)
            drawn = _draw(form, rng, answer)
            if drawn is None or (form == "fixed" and any(c["form"] == "fixed" for c in constraints)):
                continue
            narrowed = [rota for rota in alive if holds(drawn, rota)]
            if len(narrowed) == len(alive):
                continue  # it rules nothing out
            constraints.append(drawn)
            alive = narrowed
            if len(constraints) >= 5 and len(alive) <= FEW_SOLUTIONS:
                break
        if (len(constraints) >= 4 and len(alive) <= FEW_SOLUTIONS
                and (not required or sum(c["form"] in required for c in constraints) >= 2)):
            return {"answer": answer, "leave": leave, "constraints": constraints, "solutions": len(alive)}


def _sentence(constraint: dict[str, Any]) -> str:
    return TEXT[constraint["form"]].format(**{k: v for k, v in constraint.items() if k != "form"})


def _audience(number: int) -> str:
    """Whose words a base request uses: the development, search, gate or held-out set."""
    if number < 200:
        return "dev"
    return "search" if number in SEARCH_BASES else "gate" if number in GATE_BASES else "held"


def _plan_task(number: int, style: str) -> dict[str, Any]:
    problem = _problem(number, style)
    opening, between, closing = PLAN_WORDS[_audience(number)][style]
    sentences = [_sentence(c) for c in problem["constraints"]]
    current = _rota(random.Random(f"current-{number}-{style}"))
    return {"id": f"S{number:03d}~{style}", "base": f"S{number:03d}", "kind": "plan", "style": style, "expect": "act",
            "text": opening + between.join(sentences) + closing, "rota": current, **problem}


def _assign_base(number: int) -> dict[str, Any]:
    rng = random.Random(f"assign-{number}")
    person, other = rng.sample(PEOPLE, 2)
    day = rng.choice(DAYS)
    current = _rota(rng)
    while current[day] == person or _count(current, person) >= MOST_SHIFTS:  # the change has to be one the rules allow
        current = _rota(rng)
    return {"id": f"S{number:03d}", "base": f"S{number:03d}", "kind": "assign", "person": person, "other": other, "day": day,
            "rota": current, "leave": {}, "text": f"Put {person} on {day}."}


def _assign_task(number: int, style: str) -> dict[str, Any]:
    base = _assign_base(number)
    words = ASSIGN_WORDS[_audience(number)][style]
    template = words[number % len(words)]
    text = template.format(person=base["person"], other=base["other"], day=base["day"], short=base["day"][:3])
    return {**base, "id": f"S{number:03d}~{style}", "style": style, "expect": "act", "text": text}


SPLITS = {"dev": range(1, 17), "test": range(201, 217)}


def task_set(split: str) -> list[dict[str, Any]]:
    """A split's base requests: odd numbers change one shift, even numbers plan a week."""
    return [_assign_base(n) if n % 2 else {"id": f"S{n:03d}", "base": f"S{n:03d}", "kind": "plan", "text": "Plan the week."}
            for n in SPLITS[split]]


def worded(split: str) -> list[dict[str, Any]]:
    """A split's requests as worded, four to a base, with the out-of-scope ones."""
    out: list[dict[str, Any]] = []
    for number in SPLITS[split]:
        make, styles = (_assign_task, STYLES) if number % 2 else (_plan_task, PLAN_STYLES)
        out += [make(number, style) for style in styles]
    bases = [t for t in task_set(split) if t["kind"] == "assign"][:2]
    for n, template in enumerate(OUT_OF_SCOPE[split]):
        base = bases[n % 2]
        out.append({**base, "id": f"{base['id']}~out{n}", "style": "out_of_scope", "expect": "handoff",
                    "text": template.format(person=base["person"], day=base["day"])})
    return out


def _near_misses(split: str, templates: list[str], tag: str) -> list[dict[str, Any]]:
    """Each template on two of a split's requests to change a shift: the same words, but nothing should change."""
    bases = [t for t in task_set(split) if t["kind"] == "assign"][:2]
    return [{**base, "id": f"{base['id']}~{tag}{n}", "text": template.format(person=base["person"], day=base["day"]),
             "style": "near_miss", "expect": "handoff"} for base in bases for n, template in enumerate(templates)]


def load(split: str) -> list[dict[str, Any]]:
    """The requests of a split, by the names the decision loop uses for them."""
    near_misses = {"gate": (NEAR_MISS_GATE, "gate"), "stress": (NEAR_MISS, "near"), "stress2": (NEAR_MISS_FRESH, "fresh")}
    if split in near_misses:
        return _near_misses("test", *near_misses[split])
    if split == "dev+near":
        return worded("dev") + _near_misses("dev", NEAR_MISS_DEV, "neardev")
    return worded(split)


class ShiftWorld:
    """One request's team: its leave, its current rota, and what the desk did."""

    def __init__(self, task: dict[str, Any]) -> None:
        self.task = task
        self.rota = dict(task["rota"])
        self.leave = {person: list(days) for person, days in task["leave"].items()}
        self.style = task.get("style", "plain")
        self.actions: list[tuple[str, dict[str, str]]] = []

    def read_team(self) -> dict[str, Any]:
        return {"people": {person: {"skills": list(SKILLS[person])} for person in PEOPLE}, "rules": list(RULES)}

    def read_leave(self) -> dict[str, Any]:
        return {"leave": {person: list(days) for person, days in self.leave.items()}}

    def read_rota(self) -> dict[str, Any]:
        return {"rota": dict(self.rota)}

    def assign_shift(self, day: str, person: str) -> dict[str, Any]:
        day, person = _key(day), _key(person)
        if day not in DAYS or person not in PEOPLE:
            return {"error": f"no such day or person: {day!r}, {person!r}"}
        self.actions.append(("assign_shift", {"day": day, "person": person}))
        self.rota[day] = person
        return {"status": "assigned", "day": day, "person": person}

    def publish_rota(self, monday: str, tuesday: str, wednesday: str, thursday: str, friday: str) -> dict[str, Any]:
        rota = dict(zip(DAYS, map(_key, (monday, tuesday, wednesday, thursday, friday)), strict=True))
        unknown = [person for person in rota.values() if person not in PEOPLE]
        if unknown:
            return {"error": f"no such people: {unknown}"}
        self.actions.append(("publish_rota", rota))
        self.rota = rota
        return {"status": "published", "rota": rota}

    def _broken(self, rota: dict[str, str]) -> str | None:
        """What a published rota gets wrong, or None."""
        if not follows_rules(rota, self.leave):
            return "it breaks a standing rule"
        return next((_sentence(c) for c in self.task["constraints"] if not holds(c, rota)), None)

    def check(self) -> dict[str, Any]:
        task = self.task
        if not self.actions:
            return {"success": False, "reason": "nothing was changed"}
        if task["kind"] == "assign":
            if len(self.actions) > 1:
                return {"success": False, "reason": f"{len(self.actions)} changes were made where one was asked for"}
            name, arguments = self.actions[0]
            if name != "assign_shift":
                return {"success": False, "reason": f"it called {name} on a request to change one shift"}
            wanted = {"day": task["day"], "person": task["person"]}
            return ({"success": True, "reason": "accepted"} if arguments == wanted
                    else {"success": False, "reason": f"it put {arguments['person']} on {arguments['day']}"})
        published = [arguments for name, arguments in self.actions if name == "publish_rota"]
        if len(published) != 1:
            return {"success": False, "reason": f"it published {len(published)} rotas where one was asked for"}
        arguments = published[0]
        broken = self._broken(arguments)
        return ({"success": True, "reason": "accepted"} if broken is None
                else {"success": False, "reason": f"the rota doesn't meet this: {broken}"})
