Scheduling a meeting with several people:
1. List the request's constraints: the length, how many meetings, which days, the earliest start or latest end, and whether meetings must fall on different days.
2. Read the calendar of every attendee, not only the user's, for each allowed day. book_time checks only the user's calendar, so it accepts times when an attendee is busy.
3. For each allowed day, intersect the free intervals of the user and every attendee, then trim them to the time limit. A start works only when start plus length fits inside one shared interval.
4. Take the earliest start that works. For two meetings on different days, take the earliest start on each of two different days.
5. Book with every attendee's full name, then reply with the day, start, and attendees of each booking.
