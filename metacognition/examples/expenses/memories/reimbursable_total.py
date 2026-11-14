# Written by hand to the contract, as the reference a learned function is matched against.
def reimbursable_total(api, trip_id: str) -> dict:
    """Work out how much of a trip's receipts the company reimburses, in US dollars, under its expense policy."""
    trip = api.read_trip(trip_id)
    policy = api.read_policy()
    allowance = policy["allowances_usd"][trip["city"]]
    rates = policy["company_rates_usd_per_unit"]
    by_receipt_month = any("month on the receipt" in note for note in policy["notes"])
    start_month = trip["start"][:7]
    meals = {}
    total = 0.0
    for receipt in trip["receipts"]:
        month = receipt["date"][:7] if by_receipt_month else start_month
        usd = receipt["amount"] * rates[month][receipt["currency"]]
        note = receipt["note"].strip()
        if "alcohol" in note.lower():
            continue
        if receipt["category"] == "meal":
            meals[receipt["date"]] = meals.get(receipt["date"], 0.0) + usd
        elif receipt["category"] == "hotel":
            total += min(usd, allowance["hotel_per_night"])
        elif receipt["category"] == "taxi":
            if usd > 40 and not note:
                continue
            total += usd
    for spent in meals.values():
        total += min(spent, allowance["meals_per_day"])
    return {"total_usd": round(total, 2)}
