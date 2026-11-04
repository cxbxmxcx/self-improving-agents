def find_shared_starts(
    api,
    attendees: list[str],
    days: list[str],
    minutes: int,
    earliest: str = "",
    latest_end: str = "",
) -> dict:
    """Every quarter-hour start, by day, when everyone can meet.

    Hand-written bootstrap skill for chapter 6: the target a learner must
    rebuild. `api` is the read-only calendar the harness binds. Times are
    taken and returned on the user's clock; each calendar is read on its
    owner's clock and converted.
    """

    def to_minutes(hhmm):
        sign = -1 if hhmm.startswith("-") else 1
        hours, mins = hhmm.lstrip("+-").split(":")
        return sign * (int(hours) * 60 + int(mins))

    def to_hhmm(total):
        return f"{total // 60:02d}:{total % 60:02d}"

    def buffer_from(note):
        # "Please leave 15 minutes between any meeting ..." -> 15
        words = (note or "").split()
        for i, word in enumerate(words[1:], 1):
            if word.startswith("minute") and words[i - 1].isdigit():
                return int(words[i - 1])
        return 0

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
        # Each person blocks time outside their working hours and during
        # their busy events, widened by any buffer their note asks for.
        windows = []
        busy = []
        for page in pages:
            shift = to_minutes(page["utc_offset"]) - user_offset
            first, last = page["working_hours"].split("-")
            windows.append(
                (to_minutes(first) - shift, to_minutes(last) - shift)
            )
            buffer = buffer_from(page.get("note"))
            for event in page["days"][0]["events"]:
                if event.get("show_as") == "free":
                    continue
                busy.append(
                    (
                        to_minutes(event["start"]) - shift - buffer,
                        to_minutes(event["end"]) - shift + buffer,
                    )
                )
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
