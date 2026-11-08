badWords = ["status", "cancel", "policy", "allowance", "split", "how much", "what is the daily meal allowance", "daily meal allowance"]
fileWords = ["file", "submit", "expense claim", "claim"]

async def decide(request, kit):
    text = request.lower()

    if any(word in text for word in badWords):
        return {"handoff": "This request is not a new expense-claim filing."}

    if not any(word in text for word in fileWords):
        return {"handoff": "I can only file expense claims."}

    ids = kit.find("(?i)(?:T-)?\\d{3}", request)
    seen = set()
    choices = []
    for item in ids:
        item = item.upper()
        if len(item) == 3 and item.isdigit():
            item = "T-" + item
        if item not in seen:
            seen.add(item)
            choices.append(item)

    if len(choices) == 0:
        return {"handoff": "I couldn't identify a trip id to file."}

    trip = choices[0]
    if len(choices) > 1:
        options = {}
        for item in choices:
            options[item] = "the trip to file"
        options["none"] = "no clear trip"

        answer = await kit.ask(
            {"request": request, "trip_ids": choices},
            {
                "trip": {
                    "type": "choice",
                    "instructions": "Choose the single trip id the user wants filed. If there is no clear trip to file, choose none.",
                    "criteria": options,
                }
            },
        )

        picked = answer["trip"]["choice"]
        confidence = answer["trip"]["confidence"]
        if picked == "none" or confidence < 0.7:
            return {"handoff": "I couldn't determine which trip to file."}
        trip = picked

    variants = [trip]
    if trip.startswith("T-"):
        variants.append(trip[2:])
    else:
        if len(trip) == 3 and trip.isdigit():
            variants.append("T-" + trip)

    verified = None
    last_error = None
    for candidate in variants:
        try:
            info = kit.read("read_trip", trip_id=candidate)
            if not (isinstance(info, dict) and "error" in info):
                verified = candidate
                break
        except Exception as e:
            last_error = e

    if verified is None:
        return {"handoff": "I couldn't verify that trip."}

    trip = verified

    try:
        total = await kit.procedure("reimbursable_total", trip_id=trip)
        amount = round(float(total["total_usd"]), 2)
    except Exception:
        return {"handoff": "I couldn't compute the claim total."}

    return {
        "action": "file_claim",
        "arguments": {"trip_id": trip, "amount_usd": amount},
        "reply": "Filing your expense claim now.",
    }
