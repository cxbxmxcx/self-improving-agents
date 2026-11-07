def refund_decision(api, return_id: str) -> dict:
    """The procedure as written before the holiday policy: what a return gets under the regular policy."""

    def day_number(text):
        # Days since a fixed epoch, for "YYYY-MM-DD" (the civil-calendar algorithm; procedures can't import).
        year, month, day = int(text[0:4]), int(text[5:7]), int(text[8:10])
        if month <= 2:
            year -= 1
        era = year // 400
        year_of_era = year - era * 400
        day_of_year = (153 * ((month + 9) % 12) + 2) // 5 + day - 1
        return era * 146097 + year_of_era * 365 + year_of_era // 4 - year_of_era // 100 + day_of_year

    ret = api.read_return(return_id)
    order = api.read_order(ret["order_id"])
    customer = api.read_customer(order["customer_id"])
    policy = api.read_policy()
    item = [i for i in order["items"] if i["sku"] == ret["sku"]][0]
    delivered = day_number(order["delivered_on"])
    requested = day_number(ret["requested_on"])
    since = requested - delivered

    if ret["reason"] == "arrived damaged":
        if since <= policy["damaged_refund_days"]:
            return {"action": "refund", "amount_usd": round(item["price"] + item["shipping"], 2)}
        return {"action": "decline", "amount_usd": 0.0}
    if item["final_sale"] or (item["category"] == "beauty" and ret["opened"]):
        return {"action": "decline", "amount_usd": 0.0}

    amount = item["price"]
    if item["category"] == "electronics" and ret["opened"]:
        amount -= policy["restocking_fee_opened_electronics"] * item["price"]
    amount -= sum(a["value"] for a in item["accessories"] if a["name"] in ret["missing_accessories"])
    amount = round(max(amount, 0.0), 2)

    window = policy["return_window_days"][item["category"]]
    if customer["tier"] == "plus":
        window += policy["member_extra_days"]
    if since <= window:
        return {"action": "refund", "amount_usd": amount}
    if since <= window + policy["store_credit_days_after_window"]:
        return {"action": "credit", "amount_usd": amount}
    return {"action": "decline", "amount_usd": 0.0}
