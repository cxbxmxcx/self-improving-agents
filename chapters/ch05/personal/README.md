# Personal calendar agent

The example in [Chapter 5, draft 0.16](../chapter5.draft0.16.md) creates a 45-minute Writing event on the requested date. A local JSON calendar persists the action, while a scripted user review evaluates whether the chosen time suits the user's current work style. Episodic memory retains earlier requests and final reports, including observed feedback.

Run `python chapters/ch05/03_memory_learning.py` from the repository root. A fresh output directory contains the calendar, archive, episodic store, trace, and transcript. No external calendar account is accessed.

## Files and evidence

- [requests.json](requests.json) supplies four dated tasks and private reviewer fixtures.
- [encoder.py](encoder.py) provides fixed lexical vectors without model downloads.
- [scripted.py](scripted.py) chooses actions from the visible request and recalled context, using the real agent loop and tools.
- [Recorded evidence](recorded/README.md) includes the resulting calendar and memory updates.

`read_calendar()` exposes current availability, `book_time()` validates and persists an event, and `review_booking()` supplies feedback only after creation. Both 09:00 and 14:00 are available before each example booking. Private reviewer preferences are not included in the prompt, calendar availability, or episodic recall before they have been observed.

## Learning sequence

| Request | Action and review | Retained change |
|---------|-------------------|-----------------|
| CAL-201 | 14:00 accepted | Write E1 at 0.5; no earlier memory to credit |
| CAL-202 | 14:00 accepted after recalling E1 | E1 moves from 0.5 to 0.75; write E2 |
| CAL-203 | 14:00 rejected; user now requests mornings | E1 moves from 0.75 to 0.375; write correction episode E3 |
| CAL-204 | Follow-up recalls E3 and books 09:00, accepted | E3 moves from 0.5 to 0.75; write E4 |

The fourth request refers to the rejected booking's feedback without repeating 09:00. This cue matters for the small lexical retriever; a bare request selected E2 in the initial check. The stored correction supplies the preferred time, while a current explicit instruction always takes precedence over history.

Valid accepted bookings receive reward one; invalid attempts and explicitly rejected choices receive zero. A valid booking awaiting review, or a request with no attempted outcome, supplies no learning target. The measurement reports valid bookings, invalid attempts, user rejections, and missing outcomes separately, retaining all assigned requests in the acceptance-rate denominator.

All feedback in this fixture arrives before `Agent.run()` completes. Feedback arriving later would require a separate update path linked to the original event and recalled versions, with duplicate-credit protection. The example does not implement that later lifecycle or extract a permanent preference profile.

The final handoff opens the same store in a new read-only agent. Maya can retrieve her earlier episode under a new session ID; Leo cannot retrieve Maya's user-scoped records. The handoff creates no event or new memory.

Run `python -m pytest tests/test_calendar_memory_walkthrough.py -q` to check calendar persistence, constraints, feedback timing, current-instruction precedence, learned values, archived versions, and identity boundaries. The scripted run establishes mechanics; a controlled held-out comparison is needed to measure improvement from learned utilities.
