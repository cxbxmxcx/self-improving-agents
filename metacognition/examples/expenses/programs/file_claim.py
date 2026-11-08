"""Files a trip's expense claim with no language model. Jev only confirms the request asks for a claim."""

TRIP_ID = r"\bT-\d+[A-Z]?\b"
ASKS_FOR_A_CLAIM = {"type": "noul", "instructions": "Does `request` ask to file an expense claim for a business trip?"}
SURE = 0.7


async def decide(request, kit):
    trips = kit.find(TRIP_ID, request)
    if len(trips) != 1:
        return {"handoff": "the request names " + str(len(trips)) + " trips, not one"}
    answers = await kit.ask({"request": request}, {"asks_for_a_claim": ASKS_FOR_A_CLAIM})
    if answers["asks_for_a_claim"]["noul"] < SURE:
        return {"handoff": "Jev isn't sure the request asks for a claim"}
    total = await kit.procedure("reimbursable_total", trip_id=trips[0])
    if not isinstance(total, dict) or "total_usd" not in total:
        return {"handoff": "the procedure gave no total"}
    return {"action": "file_claim", "arguments": {"trip_id": trips[0], "amount_usd": total["total_usd"]},
            "reply": "I filed the claim for trip " + trips[0] + "."}
