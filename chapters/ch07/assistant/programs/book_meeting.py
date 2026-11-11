"""Books one meeting with no language model. Code finds who, which days and how long; Jev reads the limits."""

WEEK = ["mon", "tue", "wed", "thu", "fri"]
DAY_KEYS = {"monday": "mon", "mon": "mon", "tuesday": "tue", "tue": "tue", "tues": "tue", "wednesday": "wed",
            "wed": "wed", "weds": "wed", "thursday": "thu", "thu": "thu", "thur": "thu", "thurs": "thu",
            "friday": "fri", "fri": "fri"}
DAY_NAMES = {"mon": "Monday", "tue": "Tuesday", "wed": "Wednesday", "thu": "Thursday", "fri": "Friday"}
DAY = r"(?i)\b(?:monday|tuesday|wednesday|thursday|friday|mon|tues?|weds?|thu(?:rs?)?|fri)\b"
SPAN = (r"(?i)\b(?:monday|tuesday|wednesday|thursday|friday|mon|tues?|weds?|thu(?:rs?)?|fri)\b"
        r"\s*(?:to|through|thru|until|-)\s*"
        r"\b(?:monday|tuesday|wednesday|thursday|friday|mon|tues?|weds?|thu(?:rs?)?|fri)\b")
LENGTH = r"\b\d{2,3}(?=[- ]?min)"
CLOCK = r"\b\d{1,2}:\d{2}\b"
NAME = r"\b[A-Z][a-z]+\b"
SURE = 0.7
ALLOWS = "Does `request` allow the meeting to be held on "
LIMIT = {"type": "choice",
         "instructions": "How does `request` use the clock time in `time` to limit when the meeting happens?",
         "criteria": {"earliest": "The meeting may not start before `time`.",
                      "latest_end": "The meeting must be over by `time`.",
                      "neither": "`time` does not limit when the meeting happens."}}
# What the agent reads when this program cannot finish the work itself.
MEMORY = "Call find_shared_starts once for the starts everyone can make, then book the earliest with book_time."


def people_named(request, kit):
    user = kit.read("read_calendar", person="", day="mon").get("person")
    people = []
    for word in kit.find(NAME, request):
        page = kit.read("read_calendar", person=word, day="mon")
        if isinstance(page, dict) and page.get("person") not in (None, user) and page["person"] not in people:
            people.append(page["person"])
    return people


def days_named(request, kit):
    days = []
    for span in kit.find(SPAN, request):
        ends = [DAY_KEYS[word.lower()] for word in kit.find(DAY, span)]
        if len(ends) == 2 and WEEK.index(ends[0]) <= WEEK.index(ends[1]):
            days = days + WEEK[WEEK.index(ends[0]):WEEK.index(ends[1]) + 1]
    days = days + [DAY_KEYS[word.lower()] for word in kit.find(DAY, request)]
    return [day for day in WEEK if day in days]


async def decide(request, kit):
    people, days = people_named(request, kit), days_named(request, kit)
    lengths, clocks = kit.find(LENGTH, request), kit.find(CLOCK, request)
    if len(people) == 0 or len(days) == 0 or len(lengths) != 1 or len(clocks) > 1:
        return {"handoff": "could not read who, which days or how long", "route": "none", "memory": MEMORY}
    questions = {}
    for day in days:
        questions["on_" + day] = {"type": "noul", "instructions": ALLOWS + DAY_NAMES[day] + "?"}
    if len(clocks) == 1:
        questions["limit"] = LIMIT
    judged = await kit.ask({"request": request, "time": clocks[0] if len(clocks) == 1 else ""}, questions)
    days = [day for day in days if judged["on_" + day]["noul"] >= SURE]
    limits = {"earliest": "", "latest_end": ""}
    if len(clocks) == 1 and judged["limit"]["choice"] in limits:
        limits[judged["limit"]["choice"]] = clocks[0]
    found = await kit.procedure("find_shared_starts", attendees=people, days=days, minutes=int(lengths[0]),
                                earliest=limits["earliest"], latest_end=limits["latest_end"])
    starts = found.get("starts") if isinstance(found, dict) else None
    if not isinstance(starts, dict):
        return {"handoff": "the procedure gave no starts", "route": "none", "memory": MEMORY}
    for day in days:
        if starts.get(day):
            return {"action": "book_time",
                    "arguments": {"title": "Meeting with " + " and ".join(people), "day": day, "start": starts[day][0],
                                  "minutes": int(lengths[0]), "attendees": people},
                    "reply": "Booked " + lengths[0] + " minutes with " + ", ".join(people) + " on "
                             + DAY_NAMES[day] + " at " + starts[day][0] + "."}
    return {"handoff": "nobody is free together on the days asked for", "route": "none", "memory": MEMORY}
