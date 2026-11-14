# find_common_starts, before and after day 17

Both versions as the builder wrote them, copied from the full system's recording. The first was learned on day 14 from one accepted run, and the second relearned on day 17, after Dana's note changed to three quarters of an hour, also from one accepted run. Both passed their gate, 6 of 6.

They differ in how they find a day in a calendar. The first compares the first three letters, so "monday" and "mon" both find Monday:

```python
want = str(day).strip().lower()[:3]
```

The second compares the day exactly as the agent passed it, so "monday" finds nothing, and the function sees nobody's events:

```python
if d.get("day", day) != day:
    continue
```

With that one line written as the first version wrote it, the second answers all 144 of the agent's calls in `judged_versions.json` right. SCALE_TEST.md section 11.3 tells the rest.

## Learned on day 14

```python
def find_common_starts(api, attendees: list[str], days: list[str], minutes: int,
                       earliest: str = "", latest_end: str = "") -> dict:
    """Find every start time that works for the user and all the listed attendees on every one of the given days, for a meeting of the given length, so it can be held at the same time each day. days are mon to fri; earliest is the first allowed start and latest_end the time the meeting must end by, both HH:MM or empty. Returns the common starts, earliest first."""

    def to_min(s):
        parts = str(s).strip().split(":")
        return int(parts[0]) * 60 + (int(parts[1]) if len(parts) > 1 else 0)

    def offset_min(s):
        s = str(s or "").strip()
        if not s or s in ("Z", "z"):
            return 0
        sign = -1 if s[0] == "-" else 1
        s = s.lstrip("+-")
        try:
            parts = s.split(":")
            h = int(parts[0])
            m = int(parts[1]) if len(parts) > 1 else 0
        except (ValueError, TypeError):
            return 0
        return sign * (h * 60 + m)

    def buffer_from_note(note):
        # Gap the person asks for between this meeting and their other meetings.
        text = str(note or "").lower().replace("-", " ")
        if not text:
            return 0
        if "half an hour" in text or "half hour" in text:
            return 30
        if "quarter of an hour" in text or "quarter hour" in text:
            return 15
        words = {"five": 5, "ten": 10, "fifteen": 15, "twenty": 20, "thirty": 30,
                 "forty five": 45, "sixty": 60, "one": 1, "an": 1, "two": 2}
        toks = [t.strip(".,;:!()") for t in text.split()]
        for i, t in enumerate(toks[:-1]):
            nxt = toks[i + 1]
            val = None
            if t.isdigit():
                val = int(t)
            elif t in words:
                val = words[t]
            if val is None:
                continue
            if nxt.startswith("min"):
                if t in ("one", "an", "two"):
                    continue
                return val
            if nxt.startswith("hour") and val <= 4:
                return val * 60
        return 0

    def day_info(cal, day, user_off):
        # Returns (lo, hi, blocks) on the user's clock.
        lo, hi, blocks = 0, 24 * 60, []
        if not isinstance(cal, dict):
            return lo, hi, blocks
        shift = user_off - offset_min(cal.get("utc_offset", ""))
        wh = str(cal.get("working_hours", "") or "")
        if "-" in wh:
            try:
                a, b = wh.split("-")[:2]
                lo, hi = to_min(a) + shift, to_min(b) + shift
            except (ValueError, TypeError):
                lo, hi = 0, 24 * 60
        buf = buffer_from_note(cal.get("note", ""))
        entries = cal.get("days", []) or []
        want = str(day).strip().lower()[:3]
        matched = [d for d in entries
                   if str(d.get("day", "")).strip().lower()[:3] == want]
        if not matched:
            matched = entries
        for d in matched:
            for ev in d.get("events", []) or []:
                if str(ev.get("show_as", "busy")).strip().lower() == "free":
                    continue
                try:
                    s = to_min(ev["start"]) + shift
                    e = to_min(ev["end"]) + shift
                except (KeyError, ValueError, TypeError):
                    continue
                blocks.append((s - buf, e + buf))
        return lo, hi, blocks

    if not days:
        return {"starts": []}

    people = []
    for p in attendees or []:
        name = str(p).strip()
        if name and name.lower() != "maya rivera" and name not in people:
            people.append(name)

    min_start = to_min(earliest) if earliest else 0
    max_end = to_min(latest_end) if latest_end else 24 * 60

    common = None
    for day in days:
        user_cal = api.read_calendar("", day)
        user_off = offset_min(user_cal.get("utc_offset", "")) if isinstance(user_cal, dict) else 0
        infos = [day_info(user_cal, day, user_off)]
        for p in people:
            infos.append(day_info(api.read_calendar(p, day), day, user_off))
        ok = set()
        for start in range(0, 24 * 60, 15):
            end = start + minutes
            if start < min_start or end > max_end:
                continue
            good = True
            for lo, hi, blocks in infos:
                if start < lo or end > hi:
                    good = False
                    break
                if any(start < be and end > bs for bs, be in blocks):
                    good = False
                    break
            if good:
                ok.add(start)
        common = ok if common is None else set([s for s in common if s in ok])

    result = sorted(common or [])
    return {"starts": [str(s // 60).rjust(2, "0") + ":" + str(s % 60).rjust(2, "0")
                       for s in result]}
```

## Relearned on day 17

```python
def find_common_starts(api, attendees: list[str], days: list[str], minutes: int,
                       earliest: str = "", latest_end: str = "") -> dict:
    """Find every start time that works for the user and all the listed attendees on every one of the given days, for a meeting of the given length, so it can be held at the same time each day. days are mon to fri; earliest is the first allowed start and latest_end the time the meeting must end by, both HH:MM or empty. Returns the common starts, earliest first."""

    def to_min(text):
        parts = str(text).strip().split(":")
        return int(parts[0]) * 60 + (int(parts[1]) if len(parts) > 1 else 0)

    def parse_offset(text):
        t = str(text or "").strip().upper().replace("UTC", "").replace("GMT", "").strip()
        if not t or t == "Z":
            return 0
        sign = 1
        if t[0] == "-":
            sign = -1
            t = t[1:]
        elif t[0] == "+":
            t = t[1:]
        try:
            if ":" in t:
                return sign * to_min(t)
            if len(t) == 4 and t.isdigit():
                return sign * (int(t[:2]) * 60 + int(t[2:]))
            return sign * int(round(float(t) * 60))
        except Exception:
            return 0

    words = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
             "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "fifteen": 15,
             "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
             "seventy": 70, "eighty": 80, "ninety": 90}
    tens = ["twenty", "thirty", "forty", "fifty"]

    def parse_buffer(note):
        n = str(note or "").lower()
        if not n:
            return 0
        keys = ["between", "buffer", "gap", "leave", "before and after", "either side",
                "back-to-back", "back to back", "breathing", "padding", "around"]
        if not any(k in n for k in keys):
            return 0
        n = n.replace("-", " ")
        for ch in ",.;:!?()":
            n = n.replace(ch + " ", " ")
        n = n.strip(".!?;,")
        if "three quarters of an hour" in n or "three quarter" in n:
            return 45
        if "hour and a half" in n or "one and a half hour" in n:
            return 90
        if "quarter of an hour" in n or "quarter hour" in n:
            return 15
        if "half an hour" in n or "half hour" in n or "half of an hour" in n:
            return 30
        toks = n.split()

        def num_at(i):
            if i < 0:
                return None
            tok = toks[i]
            try:
                return float(tok)
            except Exception:
                pass
            if tok in words:
                v = words[tok]
                if i > 0 and toks[i - 1] in tens and v < 10 and tok not in ("a", "an"):
                    v += words[toks[i - 1]]
                return float(v)
            return None

        for i, tok in enumerate(toks):
            if tok.startswith("min"):
                v = num_at(i - 1)
                if v is not None:
                    return int(round(v))
            elif tok.startswith("hour") or tok in ("hr", "hrs"):
                v = num_at(i - 1)
                if v is not None:
                    return int(round(v * 60))
            else:
                for suf in ["minutes", "mins", "min", "m"]:
                    if tok.endswith(suf) and tok[:-len(suf)].isdigit():
                        return int(tok[:-len(suf)])
                for suf in ["hours", "hour", "hrs", "hr", "h"]:
                    if tok.endswith(suf) and tok[:-len(suf)].isdigit():
                        return int(tok[:-len(suf)]) * 60
        return 0

    people = [""]
    for a in attendees or []:
        if a and a != "Maya Rivera" and a not in people:
            people.append(a)

    lo_bound = to_min(earliest) if earliest else 0
    hi_bound = to_min(latest_end) if latest_end else 24 * 60

    common = None
    for day in days:
        user_off = 0
        blocks = []
        lo = lo_bound
        hi = hi_bound
        for person in people:
            cal = api.read_calendar(person, day)
            if not isinstance(cal, dict):
                cal = {}
            off = parse_offset(cal.get("utc_offset", ""))
            if person == "":
                user_off = off
            shift = user_off - off
            wh = str(cal.get("working_hours", "") or "")
            if "-" in wh:
                try:
                    ws, we = wh.split("-")[0], wh.split("-")[1]
                    ws_m = to_min(ws) + shift
                    we_m = to_min(we) + shift
                    if we_m <= ws_m:
                        we_m += 24 * 60
                    lo = max(lo, ws_m)
                    hi = min(hi, we_m)
                except Exception:
                    pass
            buf = parse_buffer(cal.get("note", ""))
            for d in cal.get("days", []) or []:
                if not isinstance(d, dict):
                    continue
                if d.get("day", day) != day:
                    continue
                for ev in d.get("events", []) or []:
                    if str(ev.get("show_as", "busy")).lower() == "free":
                        continue
                    try:
                        s = to_min(ev.get("start")) + shift
                        e = to_min(ev.get("end")) + shift
                    except Exception:
                        continue
                    if e < s:
                        e += 24 * 60
                    blocks.append((s - buf, e + buf))
        ok = set()
        for s in range(0, 24 * 60, 15):
            e = s + minutes
            if s < lo or e > hi:
                continue
            if all(e <= b[0] or s >= b[1] for b in blocks):
                ok.add(s)
        common = ok if common is None else set([x for x in common if x in ok])

    if common is None:
        common = set()
    out = []
    for s in sorted(common):
        h, m = divmod(s, 60)
        out.append(("0" if h < 10 else "") + str(h) + ":" + ("0" if m < 10 else "") + str(m))
    return {"starts": out}
```
