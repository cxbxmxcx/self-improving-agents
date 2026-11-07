def amount_payable(api, invoice_id: str) -> dict:
    """The hand-written payable: each client's written terms, and the convention each keeps unwritten."""

    def add_days(text, days):
        year, month, day = int(text[0:4]), int(text[5:7]), int(text[8:10])
        while days > 0:
            leap = year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)
            length = [31, 29 if leap else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][month - 1]
            day, days = day + 1, days - 1
            if day > length:
                day, month = 1, month + 1
                if month > 12:
                    month, year = 1, year + 1
        return str(year).zfill(4) + "-" + str(month).zfill(2) + "-" + str(day).zfill(2)

    invoice = api.read_invoice(invoice_id)
    po = api.read_purchase_order(invoice["po_number"])
    terms = api.read_client_terms(invoice["client_id"])
    delivered = {line["sku"]: line["delivered"] for line in po["lines"]}
    goods = taxed = materials = 0.0
    for line in invoice["lines"]:
        amount = min(line["quantity"], delivered.get(line["sku"], 0)) * line["unit_price"]
        goods += amount
        if line["category"] in terms["taxed_categories"]:
            taxed += amount
        if line["category"] == "materials":
            materials += amount
    vendor = invoice["terms"]
    take = vendor["discount_percent"] > 0
    discount = goods * vendor["discount_percent"] / 100 if take else 0.0
    freight = invoice["freight_usd"]
    total = goods - discount + taxed * terms["sales_tax_rate"] + freight
    client = invoice["client_id"]
    if client == "ALDER":  # Alder & Finch's logistics team pays carriers itself
        total -= freight
    if client == "COPPER":  # Copperline withholds 5% retention on construction materials
        total -= 0.05 * materials
    total = round(total, 2)
    if client == "BRAMBLE":  # Bramblewood's treasury pays whole dollars, rounding down
        total = float(int(total))
    days = vendor["discount_days"] if take else vendor["net_days"]
    return {"invoice_id": invoice_id, "amount_usd": total, "pay_on": add_days(invoice["invoice_date"], days)}
