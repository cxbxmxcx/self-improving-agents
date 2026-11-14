def reimbursable_total(api, trip_id: str) -> dict:
    trip = api.read_trip(trip_id)
    policy = api.read_policy()
    city = trip.get("city")
    start = str(trip.get("start", ""))
    month = start[:7]
    allow = policy.get("allowances_usd", {}).get(city, {})
    meal_cap = allow.get("meals_per_day")
    hotel_cap = allow.get("hotel_per_night")
    rates = policy.get("company_rates_usd_per_unit", {}).get(month, {})

    def to_usd(r):
        cur = r.get("currency", "USD")
        amt = float(r.get("amount", 0))
        if cur == "USD":
            return amt
        if cur not in rates:
            raise KeyError("no rate for " + str(cur) + " in " + month)
        return amt * rates[cur]

    meals = {}
    total = 0.0
    for r in trip.get("receipts", []):
        note = str(r.get("note") or "")
        if "alcohol" in note.lower():
            continue
        cat = r.get("category")
        usd = to_usd(r)
        if cat == "meal":
            d = r.get("date")
            meals[d] = meals.get(d, 0.0) + usd
        elif cat == "hotel":
            if hotel_cap is not None:
                usd = min(usd, hotel_cap)
            total += usd
        elif cat == "taxi":
            if usd > 40 and not note.strip():
                continue
            total += usd
        else:
            total += usd
    for d in meals:
        v = meals[d]
        if meal_cap is not None:
            v = min(v, meal_cap)
        total += v
    return {"total_usd": round(total + 1e-9, 2)}
