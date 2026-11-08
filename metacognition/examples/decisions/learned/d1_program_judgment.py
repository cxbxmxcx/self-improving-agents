tripPattern = "trip\\s+T-\\d+|trip\\s+\\d+|T-\\d+"
noneChoice = "none"
fileChoice = "file_claim"
otherChoice = "other"

async def decide(request, kit):
    def normalizeTrip(raw):
        text = raw.strip()
        if text.startswith("trip "):
            text = text[5:].strip()
        if text.startswith("T-"):
            return text
        if text.isdigit():
            return "T-" + text
        return text

    def moneyText(amount):
        cents = int(round(float(amount) * 100))
        negative = cents < 0
        if negative:
            cents = -cents
        dollars = cents // 100
        remainder = cents % 100
        text = str(dollars) + "." + ("0" if remainder < 10 else "") + str(remainder)
        if negative:
            return "-" + text
        return text

    try:
        found = kit.find(tripPattern, request)
    except Exception:
        return {"handoff": "could not identify any trip id"}

    rawIds = []
    normalizedByRaw = {}
    seenRaw = set()
    seenNorm = set()

    for raw in found:
        raw = raw.strip()
        norm = normalizeTrip(raw)
        if raw not in seenRaw and norm not in seenNorm:
            seenRaw.add(raw)
            seenNorm.add(norm)
            rawIds.append(raw)
            normalizedByRaw[raw] = norm

    if len(rawIds) == 0:
        return {"handoff": "no trip id identified"}

    state = {"request": request, "trip_ids": rawIds}
    criteria = {}
    for raw in rawIds:
        criteria[raw] = "the trip that should be filed now"
    criteria[noneChoice] = "none of these should be filed"

    try:
        judged = await kit.ask(
            state,
            {
                "intent": {
                    "type": "choice",
                    "instructions": "Classify the request. Choose file_claim only if the user wants this assistant to file an expense claim now for one trip. Choose other for status, cancellation, policy questions, multi-step requests, or anything that should not file a claim now.",
                    "criteria": {
                        fileChoice: "the user wants an expense claim filed now",
                        otherChoice: "the user wants something else or is not asking to file now",
                    },
                },
                "which_trip": {
                    "type": "choice",
                    "instructions": "Choose the single trip id that should be filed now. If the request is not asking to file a claim for any trip, or a trip mention is only background, choose none.",
                    "criteria": criteria,
                },
            },
        )
    except Exception:
        return {"handoff": "could not judge the request"}

    intent = judged["intent"]["choice"]
    selectedRaw = judged["which_trip"]["choice"]

    if intent != fileChoice:
        return {"handoff": "request is not asking to file a claim now"}

    if selectedRaw == noneChoice:
        return {"handoff": "no trip id selected"}

    if selectedRaw not in normalizedByRaw:
        return {"handoff": "no valid trip id selected"}

    selected = normalizedByRaw[selectedRaw]

    try:
        total = await kit.procedure("reimbursable_total", trip_id=selected)
    except Exception:
        return {"handoff": "could not complete the filing request"}

    amount = None
    if isinstance(total, dict) and "total_usd" in total:
        amount = total["total_usd"]
    else:
        amount = total

    try:
        amount = float(amount)
    except Exception:
        return {"handoff": "could not complete the filing request"}

    reply = "Filed your claim for " + selected + " for $" + moneyText(amount) + "."
    return {
        "action": "file_claim",
        "arguments": {"trip_id": selected, "amount_usd": amount},
        "reply": reply,
    }
