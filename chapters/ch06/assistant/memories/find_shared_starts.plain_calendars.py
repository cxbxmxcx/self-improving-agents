def find_shared_starts(api, attendees: list[str], days: list[str], minutes: int,
                       earliest: str = "", latest_end: str = "") -> dict:
    """Every quarter-hour start, by day, when the user and all attendees are free.

    Hand-written bootstrap skill for chapter 6: the target a learner must
    rebuild. `api` is the read-only calendar the harness binds.
    """

    def to_minutes(hhmm):
        hours, mins = hhmm.split(":")
        return int(hours) * 60 + int(mins)

    def to_hhmm(total):
        return f"{total // 60:02d}:{total % 60:02d}"

    low = to_minutes(earliest) if earliest else 0
    high = to_minutes(latest_end) if latest_end else 24 * 60
    starts = {}
    for day in days:
        shared = None
        key = day
        for person in [""] + list(attendees):  # "" is the user
            page = api.read_calendar(person, day)
            if "error" in page:
                return page
            key = page["days"][0]["day"]
            free = [tuple(to_minutes(t) for t in span.split("-")) for span in page["days"][0]["free"]]
            if shared is None:
                shared = free
            else:
                shared = [(max(a, c), min(b, d)) for a, b in shared for c, d in free if max(a, c) < min(b, d)]
        found = []
        for a, b in shared or []:
            start = max(a, low)
            start += (15 - start % 15) % 15
            while start + minutes <= min(b, high):
                found.append(to_hhmm(start))
                start += 15
        if found:
            starts[key] = found
    return {"starts": starts}
