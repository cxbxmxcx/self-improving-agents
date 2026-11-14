def reimbursable_total(api, trip_id: str) -> dict:
    trip = api.read_trip(trip_id)
    policy = api.read_policy()

    city = trip.get("city", "")
    start = trip.get("start", "")
    month = start[:7]

    allowances = policy.get("allowances_usd", {})
    city_allowances = allowances.get(city, {})
    meal_allowance = float(city_allowances.get("meals_per_day", 0))
    hotel_allowance = float(city_allowances.get("hotel_per_night", 0))

    rates_by_month = policy.get("company_rates_usd_per_unit", {})
    month_rates = rates_by_month.get(month, {})

    receipts = trip.get("receipts", [])

    meal_by_day = {}
    hotel_by_day = {}
    total = 0.0

    for receipt in receipts:
        amount = float(receipt.get("amount", 0))
        currency = str(receipt.get("currency", "USD"))
        if currency == "USD":
            rate = 1.0
        else:
            rate = float(month_rates.get(currency, 1.0))
        usd = amount * rate

        note = str(receipt.get("note", ""))
        note_text = note.lower()
        if "alcohol" in note_text:
            continue

        category = str(receipt.get("category", "")).lower()
        date = str(receipt.get("date", ""))

        if category == "meal":
            meal_by_day[date] = meal_by_day.get(date, 0.0) + usd
        elif category == "hotel":
            hotel_by_day[date] = hotel_by_day.get(date, 0.0) + usd
        elif category == "taxi":
            if usd > 40 and len(note.strip()) == 0:
                continue
            total += usd
        else:
            total += usd

    for day_total in meal_by_day.values():
        total += min(day_total, meal_allowance)

    for day_total in hotel_by_day.values():
        total += min(day_total, hotel_allowance)

    return {"total_usd": round(total, 2)}
