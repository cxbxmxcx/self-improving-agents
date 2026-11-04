# Found by the context search (v15), promoted by a person.
def find_shared_starts(api, attendees: list[str], days: list[str], minutes: int,
                       earliest: str = "", latest_end: str = "") -> dict:
    def parseTime(value):
        if isinstance(value, dict):
            for key in ("time", "dateTime", "datetime", "value", "start", "end", "from", "to"):
                if key in value:
                    try:
                        return parseTime(value[key])
                    except Exception:
                        pass
            for item in value.values():
                try:
                    return parseTime(item)
                except Exception:
                    pass
            raise ValueError

        if isinstance(value, list) or isinstance(value, tuple):
            if len(value) > 0:
                return parseTime(value[0])
            raise ValueError

        if isinstance(value, int):
            if value < 0:
                raise ValueError
            if value < 24:
                return value * 60
            if value == 2400:
                return 24 * 60
            if value <= 2359:
                hour = value // 100
                minute = value % 100
                if 0 <= hour < 24 and 0 <= minute < 60:
                    return hour * 60 + minute
            raise ValueError

        text = str(value).strip().lower()
        if text == "":
            raise ValueError

        suffix = ""
        if text.endswith("am") or text.endswith("pm"):
            suffix = text[-2:]
            text = text[:-2].strip()

        if ":" in text:
            parts = text.split(":")
            if len(parts) < 2:
                raise ValueError
            hourText = ""
            for ch in parts[0]:
                if ch.isdigit():
                    hourText += ch
            minuteText = ""
            for ch in parts[1]:
                if ch.isdigit():
                    minuteText += ch
                else:
                    break
            if hourText == "" or minuteText == "":
                raise ValueError
            hour = int(hourText)
            minute = int(minuteText[:2])
        elif text.isdigit():
            if len(text) <= 2:
                hour = int(text)
                minute = 0
            elif len(text) == 3:
                hour = int(text[0])
                minute = int(text[1:])
            else:
                hour = int(text[:-2])
                minute = int(text[-2:])
        else:
            raise ValueError

        if suffix == "pm" and hour < 12:
            hour += 12
        if suffix == "am" and hour == 12:
            hour = 0

        if hour < 0 or minute < 0 or minute >= 60 or hour > 24 or (hour == 24 and minute > 0):
            raise ValueError

        return hour * 60 + minute

    def parseOffset(value):
        if value is None:
            return 0
        if isinstance(value, int):
            if abs(value) <= 24:
                return value * 60
            return value

        text = str(value).strip()
        if text == "":
            return 0

        sign = 1
        if text[0] == "+":
            text = text[1:].strip()
        elif text[0] == "-":
            sign = -1
            text = text[1:].strip()

        if text == "":
            return 0

        try:
            if ":" in text:
                parts = text.split(":")
                if len(parts) < 2:
                    return 0
                hourText = ""
                for ch in parts[0]:
                    if ch.isdigit():
                        hourText += ch
                minuteText = ""
                for ch in parts[1]:
                    if ch.isdigit():
                        minuteText += ch
                    else:
                        break
                if hourText == "" or minuteText == "":
                    return 0
                hour = int(hourText)
                minute = int(minuteText[:2])
            elif text.isdigit():
                if len(text) <= 2:
                    hour = int(text)
                    minute = 0
                else:
                    hour = int(text[:-2])
                    minute = int(text[-2:])
            else:
                return 0

            if hour < 0 or minute < 0 or minute >= 60:
                return 0
            return sign * (hour * 60 + minute)
        except Exception:
            return 0

    def parseRange(value):
        if isinstance(value, dict):
            if "start" in value and "end" in value:
                try:
                    return parseTime(value["start"]), parseTime(value["end"])
                except Exception:
                    return None
            if "from" in value and "to" in value:
                try:
                    return parseTime(value["from"]), parseTime(value["to"])
                except Exception:
                    return None
            return None

        if isinstance(value, list) or isinstance(value, tuple):
            if len(value) >= 2:
                try:
                    return parseTime(value[0]), parseTime(value[1])
                except Exception:
                    return None
            return None

        text = str(value).strip()
        if text == "" or "-" not in text:
            return None
        parts = text.split("-", 1)
        try:
            return parseTime(parts[0]), parseTime(parts[1])
        except Exception:
            return None

    def ceilingQuarter(value):
        if value < 0:
            value = 0
        return ((value + 14) // 15) * 15

    def floorQuarter(value):
        if value < 0:
            return 0
        return (value // 15) * 15

    def formatTime(totalMinutes):
        hour = totalMinutes // 60
        minute = totalMinutes % 60
        if hour < 10:
            hourText = "0" + str(hour)
        else:
            hourText = str(hour)
        if minute < 10:
            minuteText = "0" + str(minute)
        else:
            minuteText = str(minute)
        return hourText + ":" + minuteText

    def noteBuffer(raw):
        best = 0
        if isinstance(raw, dict):
            for key in ("note", "notes", "description"):
                if key in raw and raw[key] is not None:
                    text = str(raw[key]).lower()
                    idx = text.find("minute")
                    if idx == -1:
                        idx = text.find("mins")
                    if idx != -1:
                        j = idx - 1
                        while j >= 0 and text[j].isspace():
                            j -= 1
                        digits = ""
                        while j >= 0 and text[j].isdigit():
                            digits = text[j] + digits
                            j -= 1
                        if digits != "":
                            try:
                                best = max(best, int(digits))
                            except Exception:
                                pass
        return best

    def eventBounds(event):
        if isinstance(event, dict):
            show = ""
            if "show_as" in event:
                show = str(event["show_as"]).strip().lower()
            elif "showAs" in event:
                show = str(event["showAs"]).strip().lower()
            elif "status" in event:
                show = str(event["status"]).strip().lower()
            if show == "free":
                return None

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
                    s = parseTime(start)
                    e = parseTime(end)
                    if e > s:
                        return s, e
                except Exception:
                    pass

            text = str(event).strip().lower()
            if text != "":
                for ch in ",;()[]{}./":
                    text = text.replace(ch, " ")
                parts = text.split()
                found = []
                for part in parts:
                    try:
                        found.append(parseTime(part))
                    except Exception:
                        pass
                if len(found) >= 2 and found[1] > found[0]:
                    return found[0], found[1]

        elif isinstance(event, list) or isinstance(event, tuple):
            if len(event) >= 2:
                try:
                    s = parseTime(event[0])
                    e = parseTime(event[1])
                    if e > s:
                        return s, e
                except Exception:
                    pass
            text = str(event).strip().lower()
            if text != "":
                for ch in ",;()[]{}./":
                    text = text.replace(ch, " ")
                parts = text.split()
                found = []
                for part in parts:
                    try:
                        found.append(parseTime(part))
                    except Exception:
                        pass
                if len(found) >= 2 and found[1] > found[0]:
                    return found[0], found[1]

        elif isinstance(event, str):
            text = event.strip()
            if text != "":
                if "-" in text:
                    interval = parseRange(text)
                    if interval is not None and interval[1] > interval[0]:
                        return interval[0], interval[1]
                lowered = text.lower()
                for ch in ",;()[]{}./":
                    lowered = lowered.replace(ch, " ")
                parts = lowered.split()
                found = []
                for part in parts:
                    try:
                        found.append(parseTime(part))
                    except Exception:
                        pass
                if len(found) >= 2 and found[1] > found[0]:
                    return found[0], found[1]

        return None

    def getDayInfo(raw, day):
        events = []
        workStart = None
        workEnd = None
        offset = 0
        bufferMinutes = 0
        dayText = str(day).strip().lower()

        if isinstance(raw, dict):
            if "utc_offset" in raw:
                offset = parseOffset(raw["utc_offset"])
            elif "utcOffset" in raw:
                offset = parseOffset(raw["utcOffset"])
            elif "offset" in raw:
                offset = parseOffset(raw["offset"])

            bufferMinutes = max(bufferMinutes, noteBuffer(raw))

            if "working_hours" in raw:
                interval = parseRange(raw["working_hours"])
                if interval is not None:
                    workStart, workEnd = interval

            if "days" in raw and (isinstance(raw["days"], list) or isinstance(raw["days"], tuple)):
                found = None
                for item in raw["days"]:
                    if isinstance(item, dict):
                        itemDay = str(item.get("day", "")).strip().lower()
                        if itemDay == dayText or itemDay[:3] == dayText[:3]:
                            found = item
                            break
                if found is None and len(raw["days"]) == 1:
                    found = raw["days"][0]

                if isinstance(found, dict):
                    bufferMinutes = max(bufferMinutes, noteBuffer(found))

                    if "utc_offset" in found:
                        offset = parseOffset(found["utc_offset"])
                    elif "utcOffset" in found:
                        offset = parseOffset(found["utcOffset"])
                    elif "offset" in found:
                        offset = parseOffset(found["offset"])

                    if "working_hours" in found:
                        interval = parseRange(found["working_hours"])
                        if interval is not None:
                            workStart, workEnd = interval

                    if "events" in found and found["events"] is not None:
                        if isinstance(found["events"], list) or isinstance(found["events"], tuple):
                            events = list(found["events"])
                        else:
                            events = [found["events"]]
                    elif "start" in found and "end" in found:
                        events = [found]

            if len(events) == 0 and "events" in raw:
                if isinstance(raw["events"], list) or isinstance(raw["events"], tuple):
                    events = list(raw["events"])
                else:
                    events = [raw["events"]]
            elif len(events) == 0 and "start" in raw and "end" in raw:
                events = [raw]

        elif isinstance(raw, list) or isinstance(raw, tuple):
            events = list(raw)
        elif raw is not None:
            events = [raw]

        if workStart is None:
            workStart = 0
        if workEnd is None:
            workEnd = 24 * 60

        if workStart < 0:
            workStart = 0
        if workEnd < 0:
            workEnd = 0
        if workStart > 24 * 60:
            workStart = 24 * 60
        if workEnd > 24 * 60:
            workEnd = 24 * 60

        return events, workStart, workEnd, offset, bufferMinutes

    userOffset = 0
    if len(days) > 0:
        userRaw = api.read_calendar("", days[0])
        if isinstance(userRaw, dict):
            if "utc_offset" in userRaw:
                userOffset = parseOffset(userRaw["utc_offset"])
            elif "utcOffset" in userRaw:
                userOffset = parseOffset(userRaw["utcOffset"])
            elif "offset" in userRaw:
                userOffset = parseOffset(userRaw["offset"])

    globalStart = 0
    if earliest != "":
        globalStart = parseTime(earliest)
        if globalStart < 0:
            globalStart = 0
        if globalStart > 24 * 60:
            globalStart = 24 * 60

    globalEnd = 24 * 60
    if latest_end != "":
        globalEnd = parseTime(latest_end)
        if globalEnd < 0:
            globalEnd = 0
        if globalEnd > 24 * 60:
            globalEnd = 24 * 60

    people = [""]
    for name in attendees:
        people.append(name)

    startResults = {}

    for day in days:
        busy = []
        dayStart = globalStart
        dayEnd = globalEnd

        for person in people:
            cal = api.read_calendar(person, day)
            events, workStart, workEnd, personOffset, bufferMinutes = getDayInfo(cal, day)

            workStart = workStart + userOffset - personOffset
            workEnd = workEnd + userOffset - personOffset

            if workStart < 0:
                workStart = 0
            if workStart > 24 * 60:
                workStart = 24 * 60
            if workEnd < 0:
                workEnd = 0
            if workEnd > 24 * 60:
                workEnd = 24 * 60

            if workStart > dayStart:
                dayStart = workStart
            if workEnd < dayEnd:
                dayEnd = workEnd

            for event in events:
                bounds = eventBounds(event)
                if bounds is None:
                    continue

                s = bounds[0] + userOffset - personOffset
                e = bounds[1] + userOffset - personOffset

                if bufferMinutes > 0:
                    s -= bufferMinutes
                    e += bufferMinutes

                if e <= s:
                    continue
                if e < dayStart or s > dayEnd:
                    continue

                if s < dayStart:
                    s = dayStart
                if e > dayEnd:
                    e = dayEnd
                if e > s:
                    busy.append((s, e))

        dayStart = ceilingQuarter(dayStart)
        latestStart = dayEnd - minutes
        if latestStart > 24 * 60:
            latestStart = 24 * 60
        latestStart = floorQuarter(latestStart)

        if latestStart < dayStart:
            continue

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

        slots = []
        current = dayStart
        while current <= latestStart:
            meetingEnd = current + minutes
            if meetingEnd <= dayEnd:
                free = True
                for s, e in merged:
                    if s >= meetingEnd:
                        break
                    if e > current and s < meetingEnd:
                        free = False
                        break
                if free:
                    slots.append(formatTime(current))
            current += 15

        if len(slots) > 0:
            startResults[day] = slots

    return {"starts": startResults}