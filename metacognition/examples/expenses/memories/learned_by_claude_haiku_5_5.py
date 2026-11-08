def reimbursable_total(api, trip_id: str) -> dict:
    trip = api.read_trip(trip_id)
    policy = api.read_policy()

    # Convert every foreign receipt at the company rate for the month the trip starts.
    month = trip["start"][:7]
    rates = policy["company_rates_usd_per_unit"][month]
    allowances = policy["allowances_usd"][trip["city"]]
    meal_cap = float(allowances["meals_per_day"])
    hotel_cap = float(allowances["hotel_per_night"])

    def to_usd(receipt):
        amount = float(receipt["amount"])
        if receipt["currency"] == "USD":
            return amount
        return amount * float(rates[receipt["currency"]])

    total = 0.0
    meals_by_day = {}
    hotels_by_night = {}

    for receipt in trip["receipts"]:
        note = receipt.get("note") or ""

        # Alcohol is never reimbursed.
        if "alcohol" in note.lower():
            continue

        usd = to_usd(receipt)
        category = receipt["category"]

        if category == "meal":
            meals_by_day[receipt["date"]] = meals_by_day.get(receipt["date"], 0.0) + usd
        elif category == "hotel":
            hotels_by_night[receipt["date"]] = hotels_by_night.get(receipt["date"], 0.0) + usd
        elif category == "taxi":
            # Taxis over 40 USD need a note explaining why.
            if usd > 40.0 and note.strip() == "":
                continue
            total += usd
        else:
            total += usd

    # Meals are capped per calendar day.
    for day in meals_by_day:
        total += min(meals_by_day[day], meal_cap)

    # Hotels are capped per night.
    for night in hotels_by_night:
        total += min(hotels_by_night[night], hotel_cap)

    return {"total_usd": round(total, 2)}
