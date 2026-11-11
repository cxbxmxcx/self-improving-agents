"""Picks the memory that serves a request. Jev judges whether it asks for a booking now, and for how many."""

SURE = 0.7
ASKS_TO_BOOK = {"type": "noul", "instructions": "Does `request` ask to put a meeting on the calendar now?"}
SEVERAL = {"type": "noul", "instructions": "Does `request` ask for more than one meeting?"}


async def decide(request, kit):
    judged = await kit.ask({"request": request}, {"books": ASKS_TO_BOOK, "several": SEVERAL})
    if judged["books"]["noul"] < SURE:
        return {"serve": "not_a_booking"}
    if judged["several"]["noul"] >= SURE:
        return {"serve": "several_meetings"}
    return {"serve": "one_meeting"}
