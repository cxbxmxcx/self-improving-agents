def find_shared_starts(
    api,
    attendees: list[str],
    days: list[str],
    minutes: int,
    earliest: str = "",
    latest_end: str = "",
) -> dict:
    """Chapter 6's hand-written memory, extended by hand to read the four colleagues' notes.

    A check, never offered to the agent: it shows that the checker and the generator agree
    with a function that honors every colleague's rule.
    """

    def to_minutes(hhmm):
        sign = -1 if hhmm.startswith("-") else 1
        hours, mins = hhmm.lstrip("+-").split(":")
        return sign * (int(hours) * 60 + int(mins))

    def to_hhmm(total):
        return f"{total // 60:02d}:{total % 60:02d}"

    names = {"monday": "mon", "tuesday": "tue", "wednesday": "wed", "thursday": "thu", "friday": "fri"}

    def day_of(word):
        word = word.strip(".,;:!").lower()
        if word.endswith("s"):
            word = word[:-1]
        return names.get(word)

    def rules_from(note):
        words = (note or "").replace(",", " ").split()
        clean = [w.strip(".;!").lower() for w in words]
        rule = {"buffer": 0, "not_before": None, "free": [], "no_days": [], "end_by": {}}
        for i, word in enumerate(clean):
            if word.startswith("minute") and i > 0 and clean[i - 1].isdigit():
                rule["buffer"] = int(clean[i - 1])
            if word == "before" and i + 1 < len(clean) and ":" in clean[i + 1]:
                rule["not_before"] = to_minutes(clean[i + 1])
            if word == "to" and 0 < i < len(clean) - 1 and ":" in clean[i - 1] and ":" in clean[i + 1] \
                    and "free" in clean:
                rule["free"].append((to_minutes(clean[i - 1]), to_minutes(clean[i + 1])))
            if word == "no-meeting" and i + 1 < len(clean) and day_of(clean[i + 1]):
                rule["no_days"].append(day_of(clean[i + 1]))
            if word == "offline" and i + 2 < len(clean) and clean[i + 1] == "from":
                cutoff = to_minutes(clean[i + 2])
                for later in clean[i + 3:]:
                    if day_of(later):
                        rule["end_by"][day_of(later)] = cutoff
        return rule

    low = to_minutes(earliest) if earliest else 0
    high = to_minutes(latest_end) if latest_end else 24 * 60
    starts = {}
    for day in days:
        mine = api.read_calendar("", day)
        if "error" in mine:
            return mine
        key = mine["days"][0]["day"]
        user_offset = to_minutes(mine["utc_offset"])
        pages = [mine]
        for person in attendees:
            page = api.read_calendar(person, day)
            if "error" in page:
                return page
            pages.append(page)
        windows, busy, refuse_day = [], [], False
        for page in pages:
            shift = to_minutes(page["utc_offset"]) - user_offset
            first, last = page["working_hours"].split("-")
            first, last = to_minutes(first), to_minutes(last)
            rule = rules_from(page.get("note"))
            if key in rule["no_days"]:
                refuse_day = True
            if rule["not_before"] is not None:
                first = max(first, rule["not_before"])
            if key in rule["end_by"]:
                last = min(last, rule["end_by"][key])
            windows.append((first - shift, last - shift))
            for a, b in rule["free"]:
                busy.append((a - shift, b - shift))
            for event in page["days"][0]["events"]:
                if event.get("show_as") == "free":
                    continue
                busy.append((to_minutes(event["start"]) - shift - rule["buffer"],
                             to_minutes(event["end"]) - shift + rule["buffer"]))
        if refuse_day:
            continue
        open_at = max([low] + [a for a, b in windows])
        close_at = min([high] + [b for a, b in windows])
        found = []
        start = open_at + (15 - open_at % 15) % 15
        while start + minutes <= close_at:
            if all(start + minutes <= a or b <= start for a, b in busy):
                found.append(to_hhmm(start))
            start += 15
        if found:
            starts[key] = found
    return {"starts": starts}
