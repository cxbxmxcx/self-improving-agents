"""What a model writes from the contract alone, with no experience.

The no-experience control of the learning search produced this function; it
matches none of the 16 validation tasks. Chapter 6 uses it as the seed
the context search starts from, the way chapters 2 to 4 started from a
genesis prompt.
"""


def find_shared_starts(api, attendees: list[str], days: list[str], minutes: int,
                       earliest: str = "", latest_end: str = "") -> dict:
    def parse_time(value):
        if isinstance(value, int):
            if value < 0:
                raise ValueError
            if value <= 24:
                return value * 60
            return value
        text = str(value).strip()
        if text == "":
            raise ValueError
        low = text.lower()
        if ":" in text:
            parts = text.split(":")
            hour_text = ""
            for ch in parts[0]:
                if ch.isdigit():
                    hour_text += ch
            minute_text = ""
            for ch in parts[1]:
                if ch.isdigit():
                    minute_text += ch
                else:
                    break
            if hour_text == "" or minute_text == "":
                raise ValueError
            hour = int(hour_text)
            minute = int(minute_text[:2])
            if "pm" in low and hour < 12:
                hour += 12
            if "am" in low and hour == 12:
                hour = 0
            return hour * 60 + minute
        if text.isdigit():
            if len(text) <= 2:
                return int(text) * 60
            if len(text) == 3:
                return int(text[0]) * 60 + int(text[1:])
            return int(text[:2]) * 60 + int(text[2:4])
        raise ValueError

    def parse_text_times(text):
        times = []
        n = len(text)
        low = text.lower()
        i = 0
        while i < n:
            if i > 0 and i + 2 < n and text[i] == ":" and text[i - 1].isdigit() and text[i + 1].isdigit() and text[i + 2].isdigit():
                if not (i + 3 < n and text[i + 3] == ":") and not (i - 3 >= 0 and text[i - 3] == ":"):
                    left = i - 1
                    while left >= 0 and text[left].isdigit():
                        left -= 1
                    end = i + 3
                    while end < n and text[end] == " ":
                        end += 1
                    if end + 1 < n:
                        tail = low[end:end + 2]
                        if tail == "am" or tail == "pm":
                            end += 2
                    token = text[left + 1:end]
                    try:
                        times.append(parse_time(token))
                    except Exception:
                        pass
                    i = end
                    continue
            i += 1
        return times

    def event_bounds(event):
        if isinstance(event, dict):
            start = None
            end = None
            for key in ("start", "start_time", "startTime", "begin", "from"):
                if key in event:
                    start = event[key]
                    break
            for key in ("end", "end_time", "endTime", "finish", "to"):
                if key in event:
                    end = event[key]
                    break
            if start is not None and end is not None:
                try:
                    return parse_time(start), parse_time(end)
                except Exception:
                    pass
            times = parse_text_times(str(event))
            if len(times) >= 2:
                return times[0], times[1]
        elif isinstance(event, list) or isinstance(event, tuple):
            if len(event) >= 2:
                try:
                    return parse_time(event[0]), parse_time(event[1])
                except Exception:
                    pass
            times = parse_text_times(str(event))
            if len(times) >= 2:
                return times[0], times[1]
        elif isinstance(event, str):
            times = parse_text_times(event)
            if len(times) >= 2:
                return times[0], times[1]
        return None

    def format_time(total_minutes):
        hour = total_minutes // 60
        minute = total_minutes % 60
        if hour < 10:
            hour_text = "0" + str(hour)
        else:
            hour_text = str(hour)
        if minute < 10:
            minute_text = "0" + str(minute)
        else:
            minute_text = str(minute)
        return hour_text + ":" + minute_text

    people = [""]
    for name in attendees:
        people.append(name)

    start_results = {}

    if earliest != "":
        start_limit = parse_time(earliest)
        if start_limit < 0:
            start_limit = 0
        start_limit = ((start_limit + 14) // 15) * 15
    else:
        start_limit = 0

    if latest_end != "":
        end_limit = parse_time(latest_end)
        if end_limit < 0:
            end_limit = 0
        if end_limit > 24 * 60:
            end_limit = 24 * 60
    else:
        end_limit = 24 * 60

    for day in days:
        busy = []
        for person in people:
            cal = api.read_calendar(person, day)
            items = cal
            if isinstance(cal, dict):
                if "events" in cal:
                    items = cal["events"]
                else:
                    items = [cal]
            elif isinstance(cal, str):
                items = [cal]
            if items is None:
                continue
            if isinstance(items, dict):
                items = [items]
            for event in items:
                bounds = event_bounds(event)
                if bounds is None:
                    continue
                s = bounds[0]
                e = bounds[1]
                if e < 0 or s > 24 * 60:
                    continue
                if s < 0:
                    s = 0
                if e > 24 * 60:
                    e = 24 * 60
                if e > s:
                    busy.append((s, e))

        if len(busy) > 0:
            busy = sorted(busy)
            merged = [list(busy[0])]
            for s, e in busy[1:]:
                if s <= merged[-1][1]:
                    if e > merged[-1][1]:
                        merged[-1][1] = e
                else:
                    merged.append([s, e])
        else:
            merged = []

        upper = end_limit - minutes
        if upper < start_limit:
            continue
        upper = (upper // 15) * 15
        if upper < start_limit:
            continue

        slots = []
        current = start_limit
        while current <= upper:
            meeting_end = current + minutes
            free = True
            for s, e in merged:
                if s >= meeting_end:
                    break
                if e > current and s < meeting_end:
                    free = False
                    break
            if free:
                slots.append(format_time(current))
            current += 15

        if len(slots) > 0:
            start_results[day] = slots

    return {"starts": start_results}
