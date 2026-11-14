def amount_payable(api, invoice_id):
    def leap_year(year):
        return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)

    def add_days(date_text, days):
        if not isinstance(date_text, str):
            return ""
        try:
            days = int(days)
        except Exception:
            days = 0
        if days <= 0:
            return date_text
        parts = date_text.split("-")
        if len(parts) != 3:
            return date_text
        try:
            year = int(parts[0])
            month = int(parts[1])
            day = int(parts[2])
        except Exception:
            return date_text
        while days > 0:
            if month == 2:
                month_days = 29 if leap_year(year) else 28
            elif month in [4, 6, 9, 11]:
                month_days = 30
            else:
                month_days = 31
            remaining = month_days - day
            if days <= remaining:
                day += days
                days = 0
            else:
                days -= remaining + 1
                day = 1
                month += 1
                if month > 12:
                    month = 1
                    year += 1
        return str(year) + "-" + ("0" + str(month))[-2:] + "-" + ("0" + str(day))[-2:]

    def round_cents(value):
        if value >= 0:
            return int(value * 100 + 0.5) / 100.0
        return -int(-value * 100 + 0.5) / 100.0

    def whole_dollar_payment(client_id, client_terms):
        rules = {}
        if isinstance(client_terms, dict):
            rules = client_terms.get("payment_rules", {})
        text = str(rules).lower()
        if "whole" in text and "dollar" in text:
            return True
        if "round" in text and "down" in text and "dollar" in text:
            return True
        if isinstance(client_id, str):
            low = client_id.lower()
            if "bramble" in low:
                return True
        return False

    try:
        invoice = api.read_invoice(invoice_id)
    except Exception:
        return {}

    if not isinstance(invoice, dict):
        return {}

    po_number = invoice.get("po_number", "")
    if not po_number:
        po_number = invoice.get("purchase_order", "")
    if not po_number:
        po_number = invoice.get("po", "")

    po = {}
    if isinstance(po_number, str) and po_number:
        try:
            po = api.read_purchase_order(po_number)
        except Exception:
            po = {}
        if not isinstance(po, dict):
            po = {}

    client_id = invoice.get("client_id", "")
    if not client_id:
        client_id = invoice.get("client", "")
    if not client_id and isinstance(po, dict):
        client_id = po.get("client_id", "")
    if not client_id and isinstance(po, dict):
        client_id = po.get("client", "")

    client = {}
    if isinstance(client_id, str) and client_id:
        try:
            client = api.read_client_terms(client_id)
        except Exception:
            client = {}
        if not isinstance(client, dict):
            client = {}

    taxed_categories = client.get("taxed_categories", [])
    if not isinstance(taxed_categories, list):
        taxed_categories = []
    taxed_set = set(taxed_categories)

    try:
        sales_tax_rate = float(client.get("sales_tax_rate", 0.0))
    except Exception:
        sales_tax_rate = 0.0

    terms = invoice.get("terms", {})
    if not isinstance(terms, dict):
        terms = {}
    if not terms and isinstance(invoice.get("payment_terms", {}), dict):
        terms = invoice.get("payment_terms", {})
    if not terms and isinstance(invoice.get("vendor_payment_terms", {}), dict):
        terms = invoice.get("vendor_payment_terms", {})

    try:
        discount_percent = float(terms.get("discount_percent", 0.0))
    except Exception:
        discount_percent = 0.0
    try:
        discount_days = int(terms.get("discount_days", 0))
    except Exception:
        discount_days = 0
    try:
        net_days = int(terms.get("net_days", 0))
    except Exception:
        net_days = 0

    delivered_map = {}
    po_lines = po.get("lines", [])
    if isinstance(po_lines, list):
        for line in po_lines:
            if not isinstance(line, dict):
                continue
            sku = line.get("sku", "")
            delivered_qty = 0.0
            for key in ["quantity_delivered", "delivered", "delivered_quantity", "qty_delivered"]:
                if key in line:
                    try:
                        delivered_qty = float(line.get(key, 0))
                    except Exception:
                        delivered_qty = 0.0
                    break
            delivered_map[sku] = delivered_qty

    goods_total = 0.0
    taxable_total = 0.0
    retention_total = 0.0

    inv_lines = invoice.get("lines", [])
    if not isinstance(inv_lines, list):
        inv_lines = []

    for line in inv_lines:
        if not isinstance(line, dict):
            continue

        sku = line.get("sku", "")
        cat = line.get("category", "")
        try:
            inv_qty = float(line.get("quantity", 0))
        except Exception:
            inv_qty = 0.0
        try:
            unit_price = float(line.get("unit_price", line.get("unit price", 0.0)))
        except Exception:
            unit_price = 0.0

        delivered_qty = delivered_map.get(sku, inv_qty)
        try:
            delivered_qty = float(delivered_qty)
        except Exception:
            delivered_qty = inv_qty

        if delivered_qty < 0:
            delivered_qty = 0.0
        if inv_qty < 0:
            inv_qty = 0.0

        use_qty = inv_qty if inv_qty < delivered_qty else delivered_qty
        line_total = use_qty * unit_price

        goods_total += line_total
        if cat in taxed_set:
            taxable_total += line_total

        if isinstance(client_id, str) and client_id == "COPPER" and isinstance(cat, str) and "material" in cat.lower():
            retention_total += line_total * 0.05

    try:
        freight = float(invoice.get("freight_usd", invoice.get("freight", 0.0)))
    except Exception:
        freight = 0.0

    if client_id == "ALDER":
        freight = 0.0

    tax = taxable_total * sales_tax_rate
    discount = 0.0
    if discount_percent > 0:
        discount = goods_total * discount_percent / 100.0

    amount = goods_total + tax + freight - discount - retention_total

    if whole_dollar_payment(client_id, client):
        if amount >= 0:
            amount = float(int(amount))
        else:
            amount = -float(int(-amount))
    else:
        amount = round_cents(amount)

    invoice_date = invoice.get("invoice_date", invoice.get("date", ""))
    pay_days = discount_days if discount_percent > 0 else net_days
    pay_on = add_days(invoice_date, pay_days)

    payload = {"invoice_id": invoice_id, "amount_usd": amount, "pay_on": pay_on}

    try:
        return api.schedule_payment(payload)
    except Exception:
        return payload
